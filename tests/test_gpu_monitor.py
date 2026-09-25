import pathlib
import smtplib
import sys
from unittest import mock

import pytest
import yaml

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from benchmark_worker import gpu_monitor as monitor
from libero_eval.gpu_health import GpuHealth


def config():
    return {
        "smtp": {
            "host": "smtp.example.com",
            "port": 587,
            "security": "starttls",
            "sender": "alerts@example.com",
            "username": "alerts",
            "password_env": "TEST_SMTP_PASSWORD",
        },
        "subscribers": [
            {"email": "first@example.com", "enabled": True},
            {"email": "second@example.com", "enabled": True},
            {"email": "disabled@example.com", "enabled": False},
        ],
    }


@pytest.fixture(autouse=True)
def password(monkeypatch):
    monkeypatch.setenv("TEST_SMTP_PASSWORD", "test-secret-never-log")


def failure_state():
    state = {"incident": None, "pending": []}
    monitor.record_health(state, GpuHealth(False, "driver blocked"), 100, hostname="validator-test")
    return state


def test_failure_restart_dedup_and_recovery(tmp_path):
    path = tmp_path / "state.json"
    state = failure_state()
    with mock.patch.object(monitor, "send_email") as send:
        monitor.deliver_pending(config(), state, path, 100)
        assert send.call_count == 2
        state = monitor.read_state(path)
        monitor.record_health(state, GpuHealth(False, "still blocked"), 200, hostname="validator-test")
        monitor.deliver_pending(config(), state, path, 200)
        assert send.call_count == 2
        monitor.record_health(state, GpuHealth(True, "GPU 0: Test"), 300, hostname="validator-test")
        monitor.deliver_pending(config(), state, path, 300)
        assert send.call_count == 4
        assert send.call_args.args[1]["kind"] == "recovery"
        assert state["incident"] is None
        assert state["pending"] == []
        monitor.record_health(state, GpuHealth(False, "new fault"), 400, hostname="validator-test")
        assert len(state["pending"]) == 1


def test_failed_recipient_retries_without_resending_to_accepted_recipient(tmp_path, caplog):
    state = failure_state()
    path = tmp_path / "state.json"

    def send(_config, _event, recipient):
        if recipient == "second@example.com":
            raise smtplib.SMTPAuthenticationError(535, b"test-secret-never-log")

    with mock.patch.object(monitor, "send_email", side_effect=send) as sender:
        monitor.deliver_pending(config(), state, path, 100)
        assert sender.call_count == 2
    assert "test-secret-never-log" not in caplog.text
    state = monitor.read_state(path)
    assert state["pending"][0]["delivered_to"] == ["first@example.com"]
    with mock.patch.object(monitor, "send_email") as sender:
        monitor.deliver_pending(config(), state, path, 399)
        sender.assert_not_called()
        monitor.deliver_pending(config(), state, path, 400)
        sender.assert_called_once()
        assert sender.call_args.args[2] == "second@example.com"
    assert state["pending"] == []


def test_unconfigured_smtp_keeps_notification_pending(tmp_path):
    settings = config()
    settings["smtp"]["host"] = ""
    state = failure_state()
    path = tmp_path / "state.json"
    with mock.patch.object(monitor, "send_email") as send:
        monitor.deliver_pending(settings, state, path, 100)
    send.assert_not_called()
    persisted = monitor.read_state(path)
    assert len(persisted["pending"]) == 1
    assert "not configured" in persisted["pending"][0]["last_error"]


def test_recovery_waits_for_failure_delivery(tmp_path):
    state = failure_state()
    monitor.record_health(state, GpuHealth(True, "GPU 0: Test"), 200, hostname="validator-test")
    with mock.patch.object(monitor, "send_email", side_effect=OSError("network down")) as send:
        monitor.deliver_pending(config(), state, tmp_path / "state.json", 200)
    assert send.call_count == 2
    assert all(call.args[1]["kind"] == "failure" for call in send.call_args_list)
    assert len(state["pending"]) == 2


def test_recovery_does_not_overtake_failure_retry_backoff(tmp_path):
    state = failure_state()
    path = tmp_path / "state.json"
    with mock.patch.object(monitor, "send_email", side_effect=OSError("network down")):
        monitor.deliver_pending(config(), state, path, 100)
    monitor.record_health(state, GpuHealth(True, "GPU 0: Test"), 200, hostname="validator-test")
    with mock.patch.object(monitor, "send_email") as send:
        monitor.deliver_pending(config(), state, path, 200)
        send.assert_not_called()
        monitor.deliver_pending(config(), state, path, 400)
    assert [call.args[1]["kind"] for call in send.call_args_list] == ["failure", "failure", "recovery", "recovery"]


@pytest.mark.parametrize(
    "change",
    [
        lambda c: c["subscribers"].append({"email": "FIRST@example.com", "enabled": True}),
        lambda c: c["subscribers"][0].update(email="bad\nBcc: victim@example.com"),
        lambda c: c["subscribers"][0].update(enabled="false"),
    ],
)
def test_invalid_subscriptions_fail_before_sending(tmp_path, change):
    settings = config()
    change(settings)
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(settings))
    with pytest.raises(ValueError):
        monitor.load_config(path)


@pytest.mark.parametrize("security", ["ssl", "starttls"])
def test_smtp_uses_tls_and_one_private_recipient(security):
    settings = config()
    settings["smtp"]["security"] = security
    event = failure_state()["pending"][0]
    client = mock.Mock()
    client.send_message.return_value = {}
    factory = "SMTP_SSL" if security == "ssl" else "SMTP"
    with mock.patch.object(monitor.smtplib, factory, return_value=client) as constructor:
        monitor.send_email(settings, event, "first@example.com")
    assert constructor.call_args.kwargs["timeout"] == 10
    if security == "starttls":
        client.starttls.assert_called_once()
    client.login.assert_called_once_with("alerts", "test-secret-never-log")
    message = client.send_message.call_args.args[0]
    assert message["To"] == "first@example.com"
    assert "second@example.com" not in str(message)
    assert "driver blocked" in message.get_content()
    client.close.assert_called_once()


def test_subscriber_listing_does_not_probe_or_send(tmp_path, monkeypatch, capsys):
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config()))
    monkeypatch.setattr(sys, "argv", ["monitor", "--config", str(path), "--list-subscribers"])
    with mock.patch.object(monitor, "check_gpu_health") as check, mock.patch.object(monitor, "send_email") as send:
        assert monitor.main() == 0
    assert "enabled\tfirst@example.com" in capsys.readouterr().out
    check.assert_not_called()
    send.assert_not_called()
