"""Independent GPU health monitoring and persistent email delivery to subscribers."""

from __future__ import annotations

import argparse
import datetime
import email.utils
import fcntl
import json
import logging
import os
import pathlib
import smtplib
import socket
import ssl
import sys
import tempfile
import time
import uuid
from email.headerregistry import Address
from email.message import EmailMessage

import yaml

from libero_eval.gpu_health import GPU_CHECK_INTERVAL, GpuHealth, check_gpu_health

logger = logging.getLogger("gpu_monitor")
RETRY_INTERVAL = 300


def mailbox(value: object) -> str:
    if not isinstance(value, str) or not value.isascii() or any(c.isspace() for c in value):
        raise ValueError("email address must be a plain ASCII mailbox without whitespace")
    try:
        parsed = Address(addr_spec=value)
    except ValueError as exc:
        raise ValueError("invalid email address") from exc
    if not parsed.username or not parsed.domain or parsed.addr_spec != value:
        raise ValueError("email address must include a username and domain")
    return value


def load_config(path: pathlib.Path) -> dict:
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict) or not isinstance(config.get("subscribers"), list):
        raise ValueError("config must contain a subscribers list")
    seen = set()
    for subscriber in config["subscribers"]:
        if not isinstance(subscriber, dict):
            raise ValueError("each subscriber must contain email and enabled fields")
        address = mailbox(subscriber.get("email"))
        if address.casefold() in seen:
            raise ValueError("duplicate subscriber email")
        seen.add(address.casefold())
        if not isinstance(subscriber.get("enabled"), bool):
            raise ValueError("subscriber enabled must be true or false")
    if not isinstance(config.get("smtp"), dict):
        raise ValueError("config must contain an smtp mapping")
    return config


def active_recipients(config: dict) -> list[str]:
    return [s["email"] for s in config["subscribers"] if s["enabled"]]


def smtp_settings(config: dict) -> dict:
    settings = config["smtp"]
    host = settings.get("host")
    if not isinstance(host, str) or not host or any(c.isspace() for c in host):
        raise ValueError("SMTP is not configured: set smtp.host")
    mailbox(settings.get("sender"))
    if settings.get("security") not in ("ssl", "starttls"):
        raise ValueError("smtp.security must be ssl or starttls")
    port = settings.get("port")
    if type(port) is not int or not 1 <= port <= 65535:
        raise ValueError("smtp.port must be an integer between 1 and 65535")
    username = settings.get("username", "")
    if not isinstance(username, str):
        raise ValueError("smtp.username must be a string")
    password_env = settings.get("password_env", "GPU_ALERT_SMTP_PASSWORD")
    if not isinstance(password_env, str) or not password_env.isidentifier():
        raise ValueError("smtp.password_env must name an environment variable")
    if username and not os.environ.get(password_env):
        raise ValueError(f"SMTP password is not configured: set {password_env} in the service environment file")
    return settings


def timestamp(now: float) -> str:
    return datetime.datetime.fromtimestamp(now, datetime.UTC).isoformat(timespec="seconds")


def new_event(kind: str, incident: dict, now: float, detail: str, *, hostname: str) -> dict:
    return {
        "id": str(uuid.uuid4()),
        "kind": kind,
        "hostname": hostname,
        "occurred_at": timestamp(now),
        "incident_started_at": incident["started_at"],
        "detail": detail,
        "delivered_to": [],
        "next_attempt_at": 0,
    }


def record_health(state: dict, health: GpuHealth, now: float, *, hostname: str) -> None:
    state["last_check_at"] = timestamp(now)
    state["healthy"] = health.healthy
    state["detail"] = health.detail
    state.setdefault("pending", [])
    incident = state.get("incident")
    if not health.healthy and incident is None:
        incident = {"started_at": timestamp(now)}
        state["incident"] = incident
        state["pending"].append(new_event("failure", incident, now, health.detail, hostname=hostname))
        logger.error("GPU unavailable: %s", health.detail)
    elif health.healthy and incident is not None:
        state["pending"].append(new_event("recovery", incident, now, health.detail, hostname=hostname))
        state["incident"] = None
        logger.info("GPU recovered")


def save_state(path: pathlib.Path, state: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, delete=False, encoding="utf-8") as stream:
        temporary = pathlib.Path(stream.name)
        try:
            json.dump(state, stream, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)


def read_state(path: pathlib.Path) -> dict:
    if not path.exists():
        return {"incident": None, "pending": []}
    state = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(state, dict) or not isinstance(state.get("pending"), list):
        raise ValueError("invalid GPU monitor state; preserve it for inspection before recovery")
    return state


def send_email(config: dict, event: dict, recipient: str) -> None:
    settings = smtp_settings(config)
    status = {"failure": "GPU unavailable", "recovery": "GPU recovered", "test": "Test notification"}[event["kind"]]
    message = EmailMessage()
    message["From"] = settings["sender"]
    message["To"] = recipient
    message["Subject"] = f"[OpenRoboto] {status} on {event['hostname']}"
    message["Date"] = email.utils.formatdate(localtime=False)
    message["Message-ID"] = f"<{event['id']}@{settings['sender'].split('@')[1]}>"
    message.set_content(
        f"{status}\n\nHost: {event['hostname']}\n"
        f"Detected at (UTC): {event['occurred_at']}\n"
        f"Incident started (UTC): {event['incident_started_at']}\n\n"
        f"GPU check: {event['detail']}\n\n"
        "This notification reports host GPU health, not a model evaluation result.\n"
        "Manage recipients in the GPU monitor configuration's subscribers list.\n"
    )
    context = ssl.create_default_context()
    if settings["security"] == "ssl":
        client = smtplib.SMTP_SSL(settings["host"], settings["port"], timeout=10, context=context)
    else:
        client = smtplib.SMTP(settings["host"], settings["port"], timeout=10)
    try:
        if settings["security"] == "starttls":
            client.starttls(context=context)
        if settings.get("username"):
            client.login(settings["username"], os.environ[settings.get("password_env", "GPU_ALERT_SMTP_PASSWORD")])
        refused = client.send_message(message)
        if refused:
            raise smtplib.SMTPRecipientsRefused(refused)
    finally:
        # A failed QUIT after successful DATA must not cause a duplicate retry.
        client.close()


def deliver_pending(config: dict, state: dict, path: pathlib.Path, now: float) -> None:
    recipients = active_recipients(config)
    for event in list(state["pending"]):
        if now < event["next_attempt_at"]:
            break
        event["next_attempt_at"] = now + RETRY_INTERVAL
        if not recipients:
            event["last_error"] = "no enabled subscribers; notification not sent"
            logger.error(event["last_error"])
            save_state(path, state)
            break
        try:
            smtp_settings(config)
        except ValueError as exc:
            event["last_error"] = str(exc)
            logger.error("notification not sent: %s", exc)
            save_state(path, state)
            break
        delivered = {address.casefold() for address in event["delivered_to"]}
        for recipient in recipients:
            if recipient.casefold() in delivered:
                continue
            try:
                send_email(config, event, recipient)
            except (OSError, smtplib.SMTPException) as exc:
                # SMTP server error text can include credentials or message data.
                event["last_error"] = f"SMTP delivery failed ({type(exc).__name__}); retry pending"
                logger.error(event["last_error"])
            else:
                event["delivered_to"].append(recipient)
                delivered.add(recipient.casefold())
                event.pop("last_error", None)
                logger.info("%s notification accepted by SMTP for %s", event["kind"], recipient)
            save_state(path, state)
        if all(recipient.casefold() in delivered for recipient in recipients):
            state["last_delivery"] = {
                "event_id": event["id"],
                "kind": event["kind"],
                "at": timestamp(now),
                "recipients": event["delivered_to"],
            }
            state["pending"].remove(event)
            save_state(path, state)
        else:
            break  # Preserve failure/recovery ordering when delivery is delayed.


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=pathlib.Path, required=True)
    parser.add_argument(
        "--state-file", type=pathlib.Path, default=pathlib.Path.home() / ".local/state/openroboto/gpu-monitor.json"
    )
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--once", action="store_true", help="Check GPU health and attempt pending delivery once")
    modes.add_argument("--list-subscribers", action="store_true")
    modes.add_argument("--test-email", action="store_true", help="Send a test email to all enabled subscribers")
    modes.add_argument("--status", action="store_true", help="Show last GPU check and pending email delivery")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    try:
        config = load_config(args.config)
        if args.list_subscribers:
            for subscriber in config["subscribers"]:
                print(f"{'enabled' if subscriber['enabled'] else 'disabled'}\t{subscriber['email']}")
            return 0
        if args.status:
            print(json.dumps(read_state(args.state_file), indent=2))
            return 0
        if args.test_email:
            smtp_settings(config)
            if not active_recipients(config):
                raise ValueError("no enabled subscribers; test email not sent")
            now = time.time()
            event = new_event(
                "test",
                {"started_at": timestamp(now)},
                now,
                "SMTP delivery test; no GPU failure is implied.",
                hostname=socket.gethostname(),
            )
            for recipient in active_recipients(config):
                send_email(config, event, recipient)
                print(f"Test email accepted by SMTP for {recipient}; verify receipt in the inbox.")
            return 0
        args.state_file.parent.mkdir(parents=True, exist_ok=True)
        with args.state_file.with_suffix(".lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            state = read_state(args.state_file)
            while True:
                started = time.monotonic()
                health = check_gpu_health()
                now = time.time()
                record_health(state, health, now, hostname=socket.gethostname())
                save_state(args.state_file, state)
                try:
                    config = load_config(args.config)
                    deliver_pending(config, state, args.state_file, now)
                except (ValueError, yaml.YAMLError, OSError) as exc:
                    logger.error("notification configuration/state error (%s); will retry", type(exc).__name__)
                if args.once:
                    return 0 if health.healthy else 1
                time.sleep(max(0, GPU_CHECK_INTERVAL - (time.monotonic() - started)))
    except ValueError as exc:
        logger.error("%s", exc)
    except (OSError, smtplib.SMTPException, yaml.YAMLError) as exc:
        logger.error("monitor failed (%s); check configuration, permissions and SMTP connectivity", type(exc).__name__)
    return 2


if __name__ == "__main__":
    sys.exit(main())
