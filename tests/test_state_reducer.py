import importlib.util
import pathlib
import sys
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]

def load_module(name, relative_path):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module

anchor_name = "vlnce_baselines.common.opennav_ext.anchor_chain"
load_module(anchor_name, "vlnce_baselines/common/opennav_ext/anchor_chain.py")
state_reducer_module = load_module(
    "state_reducer_under_test",
    "vlnce_baselines/common/opennav_ext/state_reducer.py",
)
RouteStateReducer = state_reducer_module.RouteStateReducer


class FakeProgress:
    def __init__(self, calls):
        self.calls = calls

    def reset_episode(self, anchors):
        self.calls.append(("progress_reset", len(anchors)))

    def update(self, step, position, heading, tags, geometry):
        self.calls.append(("progress_update", step))
        return {"j": 1, "n_anchors": 1, "complete": True, "ever_satisfied": [0]}

    def completion_text(self):
        return "verified"


class FakeSpatial:
    def __init__(self, calls):
        self.calls = calls

    def reset_episode(self):
        self.calls.append(("spatial_reset",))

    def update(self, step, position, heading, candidates):
        self.calls.append(("spatial_update", step))
        return {"node_id": 0}


class FakePool:
    def __init__(self, calls):
        self.calls = calls

    def reset_episode(self, vocabulary):
        self.calls.append(("pool_reset", tuple(vocabulary)))

    def update(self, step, position, heading, tags, geometry):
        self.calls.append(("pool_update", step))
        return {"pool_size": 1}


class FakeGate:
    def __init__(self, calls):
        self.calls = calls

    def evaluate(self, locator, pool, final_term):
        self.calls.append(("gate", final_term))
        return {"allow_stop": True}


class StateReducerTest(unittest.TestCase):
    def build_reducer(self):
        calls = []
        reducer = RouteStateReducer(
            anchor_chain_enabled=True,
            progress_locator=FakeProgress(calls),
            spatial_graph=FakeSpatial(calls),
            landmark_pool=FakePool(calls),
            terminal_gate=FakeGate(calls),
        )
        return reducer, calls

    def test_reset_owns_all_episode_scoped_modules_and_is_idempotent(self):
        reducer, calls = self.build_reducer()
        plan = reducer.reset_episode("ep-1", "Go through the door", "door")
        self.assertEqual(plan["n_anchors"], 1)
        self.assertEqual(
            [name for name, *_ in calls],
            ["progress_reset", "spatial_reset", "pool_reset"],
        )
        reducer.reset_episode("ep-1", "changed", "changed")
        self.assertEqual(len(calls), 3)
        reducer.reset_episode("ep-2", "Turn left", "")
        self.assertEqual(len(calls), 6)

    def test_updates_delegate_to_owned_modules(self):
        reducer, calls = self.build_reducer()
        reducer.reset_episode("ep-1", "Go through the door", "door")
        spatial = reducer.update_spatial(2, [0, 0, 0], 0.0, [])
        progress = reducer.update_progress(2, [0, 0, 0], 0.0, {}, {})
        pool = reducer.update_landmarks(2, [0, 0, 0], 0.0, {}, {})
        gate = reducer.evaluate_stop(progress, object(), "door")
        self.assertEqual(spatial["node_id"], 0)
        self.assertTrue(progress["complete"])
        self.assertEqual(pool["pool_size"], 1)
        self.assertTrue(gate["allow_stop"])
        self.assertEqual(reducer.completion_text(), "verified")
        self.assertEqual(
            [name for name, *_ in calls][-4:],
            ["spatial_update", "progress_update", "pool_update", "gate"],
        )

    def test_update_before_reset_fails_loudly(self):
        reducer, _ = self.build_reducer()
        with self.assertRaises(RuntimeError):
            reducer.update_progress(0, None, None, {}, {})

    def test_trainer_has_no_direct_state_module_mutations(self):
        source = (
            ROOT / "vlnce_baselines/common/base_il_trainer_llm.py"
        ).read_text(encoding="utf-8")
        for forbidden in (
            "memory_diagnostic.update",
            "progress_locator.update",
            "landmark_pool.update",
            "terminal_gate.evaluate",
            "progress_locator.reset_episode",
            "landmark_pool.reset_episode",
        ):
            self.assertNotIn(forbidden, source)
        self.assertIn("route_state_reducer.reset_episode", source)
        self.assertIn("route_state_reducer.update_progress", source)
        self.assertIn("route_state_reducer.update_landmarks", source)
        self.assertIn("route_state_reducer.evaluate_stop", source)


if __name__ == "__main__":
    unittest.main()
