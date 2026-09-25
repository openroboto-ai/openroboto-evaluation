"""Static LingBot batch scheduling regression tests."""

import asyncio
import importlib.util
import pathlib
import sys
import types
import unittest


_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "libero_eval"))

# The scheduler tests do not need LingBot's msgpack implementation; provide the
# import boundary expected by lingbot_batch_server in the lightweight test env.
previous_deploy = sys.modules.get("deploy")
previous_msgpack_numpy = sys.modules.get("deploy.msgpack_numpy")
websocket_module_names = ("websockets", "websockets.asyncio", "websockets.asyncio.server", "websockets.frames")
previous_websocket_modules = {name: sys.modules.get(name) for name in websocket_module_names}
deploy = types.ModuleType("deploy")
msgpack_numpy = types.ModuleType("deploy.msgpack_numpy")
msgpack_numpy.Packer = object
msgpack_numpy.unpackb = lambda value: value
sys.modules["deploy"] = deploy
sys.modules["deploy.msgpack_numpy"] = msgpack_numpy
deploy.msgpack_numpy = msgpack_numpy
websockets = types.ModuleType("websockets")
websockets.ConnectionClosed = ConnectionError
websockets_asyncio = types.ModuleType("websockets.asyncio")
websockets_server = types.ModuleType("websockets.asyncio.server")
websockets_frames = types.ModuleType("websockets.frames")
websockets_frames.CloseCode = types.SimpleNamespace(INTERNAL_ERROR=1011)
websockets.asyncio = websockets_asyncio
websockets.frames = websockets_frames
websockets_asyncio.server = websockets_server
sys.modules["websockets"] = websockets
sys.modules["websockets.asyncio"] = websockets_asyncio
sys.modules["websockets.asyncio.server"] = websockets_server
sys.modules["websockets.frames"] = websockets_frames

module_path = _ROOT / "libero_eval" / "lingbot_batch_server.py"
spec = importlib.util.spec_from_file_location("_tested_lingbot_batch_server", module_path)
assert spec is not None and spec.loader is not None
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
try:
    spec.loader.exec_module(module)
finally:
    if previous_deploy is None:
        sys.modules.pop("deploy", None)
    else:
        sys.modules["deploy"] = previous_deploy
    if previous_msgpack_numpy is None:
        sys.modules.pop("deploy.msgpack_numpy", None)
    else:
        sys.modules["deploy.msgpack_numpy"] = previous_msgpack_numpy
    for name, previous in previous_websocket_modules.items():
        if previous is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = previous

LingbotStaticBatchServer = module.LingbotStaticBatchServer
_LaneStateChange = module._LaneStateChange
_PendingRequest = module._PendingRequest


class _FakePolicy:
    def __init__(self):
        self.calls = []

    def infer_batch(self, observations):
        self.calls.append(observations)
        return [{"action": observation["value"]} for observation in observations]

    def infer(self, observation):
        return {"action": None}


class TestLingbotStaticBatchServer(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.policy = _FakePolicy()
        self.server = LingbotStaticBatchServer(
            self.policy,
            host="127.0.0.1",
            port=9000,
            batch_size=4,
            lane_count=8,
        )
        self.server._policy_lock = asyncio.Lock()
        self.worker = asyncio.create_task(self.server._infer_worker())

    async def asyncTearDown(self):
        self.worker.cancel()
        try:
            await self.worker
        except asyncio.CancelledError:
            pass

    async def _activate(self, *lanes):
        for lane in lanes:
            await self.server._events.put(_LaneStateChange(lane, True))
        await asyncio.sleep(0)

    async def _deactivate(self, *lanes):
        for lane in lanes:
            await self.server._events.put(_LaneStateChange(lane, False))
        await asyncio.sleep(0)

    async def _request(self, lane):
        future = asyncio.get_running_loop().create_future()
        await self.server._events.put(
            _PendingRequest(
                observation={"value": lane},
                future=future,
                enqueued_at=0.0,
                lane=lane,
            )
        )
        return future

    async def test_active_cohort_waits_for_every_fixed_lane(self):
        await self._activate(0, 1, 2, 3)
        futures = [await self._request(lane) for lane in (0, 1, 2)]
        await asyncio.sleep(0.01)
        self.assertEqual(self.policy.calls, [])

        futures.append(await self._request(3))
        results = await asyncio.gather(*futures)
        self.assertEqual([item["action"] for item in results], [0, 1, 2, 3])
        self.assertEqual([item["value"] for item in self.policy.calls[0]], [0, 1, 2, 3])
        self.assertTrue(all(item["server_timing"]["padded_to"] == 4 for item in results))

    async def test_inactive_tail_lane_is_filled_without_changing_shape(self):
        await self._deactivate(3)
        futures = [await self._request(lane) for lane in (0, 1, 2)]
        results = await asyncio.gather(*futures)

        self.assertEqual(len(self.policy.calls), 1)
        self.assertEqual(len(self.policy.calls[0]), 4)
        self.assertEqual([item["value"] for item in self.policy.calls[0]][:3], [0, 1, 2])
        self.assertEqual([item["action"] for item in results], [0, 1, 2])
        self.assertTrue(all(item["server_timing"]["batch_size"] == 3 for item in results))

    async def test_release_is_applied_before_a_ready_batch(self):
        futures = [await self._request(lane) for lane in (0, 1, 2)]
        await self.server._events.put(_LaneStateChange(3, False))
        results = await asyncio.gather(*futures)

        self.assertEqual(len(self.policy.calls), 1)
        self.assertEqual([item["action"] for item in results], [0, 1, 2])
        self.assertTrue(all(item["server_timing"]["batch_size"] == 3 for item in results))

    def test_lane_count_must_form_whole_cohorts(self):
        with self.assertRaisesRegex(ValueError, "positive multiple"):
            LingbotStaticBatchServer(
                self.policy,
                host="127.0.0.1",
                port=9000,
                batch_size=4,
                lane_count=6,
            )


if __name__ == "__main__":
    unittest.main()
