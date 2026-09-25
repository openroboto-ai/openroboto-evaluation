"""Explicitly release one static LingBot evaluation lane."""

from __future__ import annotations

import argparse

from openpi_client import websocket_client_policy

from lingbot_eval_protocol import POLICY_BATCH_LANE_FIELD, POLICY_BATCH_LANE_RELEASE_FIELD


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--lane", type=int, required=True)
    args = parser.parse_args()

    client = websocket_client_policy.WebsocketClientPolicy(args.host, args.port)
    result = client.infer({
        POLICY_BATCH_LANE_FIELD: args.lane,
        POLICY_BATCH_LANE_RELEASE_FIELD: True,
    })
    if not isinstance(result, dict) or result.get("released_lane") != args.lane:
        raise RuntimeError(f"policy server did not acknowledge static lane {args.lane}: {result!r}")


if __name__ == "__main__":
    main()
