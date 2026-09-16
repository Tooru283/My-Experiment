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


contracts_name = "vlnce_baselines.common.opennav_ext.pipeline_contracts"
load_module(contracts_name, "vlnce_baselines/common/opennav_ext/pipeline_contracts.py")
provider_module = load_module(
    "progress_provider_under_test",
    "vlnce_baselines/common/opennav_ext/progress_provider.py",
)
ACNL1ProgressProvider = provider_module.ACNL1ProgressProvider
terminal_target_is_confirmed = provider_module.terminal_target_is_confirmed
TerminalEvidenceMemory = provider_module.TerminalEvidenceMemory
TerminalInstanceTracker = provider_module.TerminalInstanceTracker
direction_aligned_terminal_result = provider_module.direction_aligned_terminal_result
weak_generic_close_is_confirmed = provider_module.weak_generic_close_is_confirmed
is_weak_generic_terminal = provider_module.is_weak_generic_terminal


class FakeReducer:
    def completion_text(self):
        return "Verified from ACN L1."


class ProgressProviderTest(unittest.TestCase):
    def setUp(self):
        self.provider = ACNL1ProgressProvider(FakeReducer())
        self.provider.reset_episode("ep-1")

    def test_l1_is_authoritative_and_structured(self):
        update = self.provider.update(
            {
                "j": 1,
                "n_anchors": 3,
                "complete": False,
                "current_raw": "Turn left",
                "ever_satisfied": [0],
                "abstained_this_step": False,
                "degenerate": False,
            },
            evidence_refs=("progress_locator:2",),
        )
        self.assertEqual(update.provider, "acn_l1")
        self.assertTrue(update.authoritative)
        self.assertEqual(update.current_index, 1)
        self.assertEqual(update.total, 3)
        self.assertEqual(update.transition, "advance")
        self.assertEqual(update.display_text, "Verified from ACN L1.")
        self.assertIsNone(update.shadow)

    def test_action_evidence_is_exposed_in_progress_contract(self):
        self.provider.state_reducer.plan = {
            "anchors": [
                {"raw": "Exit the bedroom"},
                {"raw": "Turn right"},
            ],
            "terminal_policy": {"present": False},
        }
        update = self.provider.update(
            {
                "j": 1,
                "n_anchors": 2,
                "complete": False,
                "current_raw": "Turn right",
                "current_action": [{"type": "turn", "direction": "right"}],
                "ever_satisfied": [0],
                "last_evaluation": {
                    "anchor_index": 1,
                    "status": "incomplete",
                    "missing_evidence": ["required_turn_not_reached"],
                },
                "completion_events": [
                    {
                        "anchor_index": 0,
                        "status": "completed",
                        "event_interval": [0, 1],
                        "confirmed_at": 1,
                    }
                ],
                "degenerate": False,
            }
        )
        self.assertEqual(update.current_action["raw"], "Turn right")
        self.assertEqual(
            update.current_action["requirements"],
            [{"type": "turn", "direction": "right"}],
        )
        self.assertEqual(
            update.verified_actions,
            ({"anchor_index": 0, "raw": "Exit the bedroom"},),
        )
        self.assertEqual(
            update.missing_evidence, ("required_turn_not_reached",)
        )
        self.assertEqual(update.completion_events[0]["event_interval"], [0, 1])

    def test_hold_and_advance_are_derived_across_steps(self):
        first = self.provider.update(
            {"j": 0, "n_anchors": 2, "ever_satisfied": [], "degenerate": False}
        )
        held = self.provider.update(
            {
                "j": 0,
                "n_anchors": 2,
                "ever_satisfied": [],
                "abstained_this_step": False,
                "degenerate": False,
            }
        )
        advanced = self.provider.update(
            {
                "j": 1,
                "n_anchors": 2,
                "ever_satisfied": [0],
                "abstained_this_step": False,
                "degenerate": False,
            }
        )
        self.assertEqual(first.transition, "hold")
        self.assertEqual(held.transition, "hold")
        self.assertEqual(advanced.transition, "advance")

    def test_degenerate_chain_abstains_without_llm_fallback(self):
        update = self.provider.update(
            {"j": 0, "n_anchors": 0, "ever_satisfied": [], "degenerate": True}
        )
        self.assertEqual(update.transition, "abstain")
        self.assertTrue(update.abstained)
        self.assertIsNone(update.complete)
        self.assertEqual(update.provider, "acn_l1")

    def test_route_complete_does_not_imply_goal_complete(self):
        self.provider.state_reducer.plan = {
            "terminal_policy": {"present": True, "target": "doorway"}
        }
        route_update = self.provider.update(
            {
                "j": 3, "n_anchors": 3, "complete": True,
                "ever_satisfied": [0, 1, 2], "degenerate": False,
            }
        )
        self.assertTrue(route_update.complete)
        self.assertTrue(route_update.route_progress_complete)
        self.assertFalse(route_update.terminal_target_confirmed)
        self.assertFalse(route_update.goal_complete)

        unconfirmed = self.provider.apply_terminal_evidence(
            route_update, terminal_target_confirmed=False
        )
        confirmed = self.provider.apply_terminal_evidence(
            route_update, terminal_target_confirmed=True,
            evidence_refs=("visual_persistence:9",),
        )
        self.assertFalse(unconfirmed.goal_complete)
        self.assertTrue(confirmed.terminal_target_confirmed)
        self.assertTrue(confirmed.goal_complete)
        self.assertIn("visual_persistence:9", confirmed.evidence_refs)

    def test_direction_aligned_terminal_rejects_cross_direction_aggregate(self):
        raw = {
            "verdict": "allow",
            "final_target_visible": True,
            "arrival_evidence": True,
            "required_landmark_terms": ["archway"],
            "selected_candidate_verdict": {
                "candidate_id": "current-pano",
                "target_direction_id": "1",
                "final_target_visible": False,
                "arrival_evidence": False,
                "matched_final_landmarks": [],
            },
        }
        aligned = direction_aligned_terminal_result(raw)
        self.assertEqual(aligned["verdict"], "reject")
        self.assertFalse(aligned["final_target_visible"])
        self.assertTrue(aligned["aggregate_final_target_visible"])
        self.assertFalse(
            aligned["direction_aligned_terminal_support"]["aligned"]
        )

    def test_direction_aligned_terminal_accepts_same_candidate_evidence(self):
        raw = {
            "verdict": "allow",
            "final_target_visible": True,
            "arrival_evidence": True,
            "required_landmark_terms": ["archway"],
            "selected_candidate_verdict": {
                "candidate_id": "current-pano",
                "target_direction_id": "1",
                "final_target_visible": True,
                "arrival_evidence": True,
                "matched_final_landmarks": ["archway"],
            },
        }
        aligned = direction_aligned_terminal_result(raw)
        self.assertEqual(aligned["verdict"], "allow")
        self.assertTrue(aligned["final_target_visible"])
        self.assertEqual(
            aligned["current_view_corroboration"]["scope"],
            "selected_candidate_direction",
        )

    def test_direction_aligned_terminal_requires_direction_id(self):
        raw = {
            "verdict": "allow",
            "required_landmark_terms": ["sofa"],
            "selected_candidate_verdict": {
                "candidate_id": "current-pano",
                "target_direction_id": None,
                "final_target_visible": True,
                "arrival_evidence": True,
                "matched_final_landmarks": ["sofa"],
            },
        }
        aligned = direction_aligned_terminal_result(raw)
        self.assertEqual(aligned["verdict"], "reject")

    def test_terminal_confirmation_requires_formal_v2_allow(self):
        raw = {"final_target_visible": True, "arrival_evidence": True}
        self.assertFalse(terminal_target_is_confirmed(
            dict(raw, verdict="not_applicable"), persistent_visual_confirm=True, spatial_distance_confirmed=True
        ))
        self.assertFalse(terminal_target_is_confirmed(
            dict(raw, verdict="reject"), persistent_visual_confirm=True, spatial_distance_confirmed=True
        ))
        self.assertFalse(terminal_target_is_confirmed(
            dict(raw, verdict="allow"), persistent_visual_confirm=False, spatial_distance_confirmed=True
        ))
        self.assertFalse(terminal_target_is_confirmed(
            dict(raw, verdict="allow"), persistent_visual_confirm=True, spatial_distance_confirmed=False
        ))
        self.assertTrue(terminal_target_is_confirmed(
            dict(raw, verdict="allow"), persistent_visual_confirm=True, spatial_distance_confirmed=True
        ))

    def test_generic_terminal_needs_explicit_current_view_corroboration(self):
        generic = {
            "verdict": "uncertain",
            "allow_blockers": ["generic_final_terms:archway"],
            "final_target_visible": True,
            "arrival_evidence": True,
            "current_view_corroboration": {"uncorroborated_terms": ["archway"]},
        }
        self.assertFalse(terminal_target_is_confirmed(
            generic, persistent_visual_confirm=True, spatial_distance_confirmed=True
        ))
        generic["current_view_corroboration"]["uncorroborated_terms"] = []
        self.assertTrue(terminal_target_is_confirmed(
            generic, persistent_visual_confirm=True, spatial_distance_confirmed=True,
            weak_generic_route_matured=True,
        ))

    def test_weak_generic_terminal_requires_route_maturity(self):
        weak = {
            "required_landmark_terms": ["archway"],
            "all_landmark_terms": ["floor", "archway"],
        }
        distinctive = {
            "required_landmark_terms": ["doorway"],
            "all_landmark_terms": ["stairs", "white double doors", "doorway"],
        }
        self.assertTrue(is_weak_generic_terminal(weak))
        weak_evidence = dict(
            weak, verdict="uncertain",
            allow_blockers=["generic_final_terms:archway"],
            final_target_visible=True, arrival_evidence=True,
            current_view_corroboration={"uncorroborated_terms": []},
        )
        self.assertFalse(terminal_target_is_confirmed(
            weak_evidence, persistent_visual_confirm=True,
            spatial_distance_confirmed=True,
            weak_generic_route_matured=False,
        ))
        self.assertTrue(terminal_target_is_confirmed(
            weak_evidence, persistent_visual_confirm=True,
            spatial_distance_confirmed=True,
            weak_generic_route_matured=True,
        ))
        self.assertFalse(is_weak_generic_terminal(distinctive))

    def test_terminal_memory_is_short_lived_and_ego_local(self):
        memory = TerminalEvidenceMemory(max_age_steps=2, max_displacement_m=2.25)
        first = memory.update(
            step_id=4, position=(0, 0, 0), direct_confirmed=True,
            current_visual_support=True, instance_key="target-1",
        )
        self.assertTrue(first["confirmed"])
        reused = memory.update(
            step_id=5, position=(1.5, 0, 0), direct_confirmed=False,
            current_visual_support=True, instance_key="target-1",
        )
        self.assertTrue(reused["confirmed"])
        self.assertTrue(reused["reused"])
        lost_visual = memory.update(
            step_id=5, position=(1.5, 0, 0), direct_confirmed=False,
            current_visual_support=False, instance_key="target-1",
        )
        self.assertFalse(lost_visual["confirmed"])
        too_far = memory.update(
            step_id=6, position=(3.0, 0, 0), direct_confirmed=False,
            current_visual_support=True, instance_key="target-1",
        )
        self.assertFalse(too_far["confirmed"])

    def test_terminal_memory_clears_on_instance_change(self):
        memory = TerminalEvidenceMemory(max_age_steps=2, max_displacement_m=2.25)
        memory.update(
            step_id=4, position=(0, 0, 0), direct_confirmed=True,
            current_visual_support=True, instance_key="target-1",
        )
        changed = memory.update(
            step_id=5, position=(0, 0, 0), direct_confirmed=False,
            current_visual_support=True, instance_key="target-2",
            confirmed_instance_switch=True,
        )
        self.assertFalse(changed["confirmed"])
        self.assertEqual(changed["cache_cleared_reason"], "confirmed_instance_changed")
        self.assertIsNone(changed["cached_instance_key"])

    def test_terminal_memory_retains_cache_on_uncertain_instance_switch(self):
        memory = TerminalEvidenceMemory(max_age_steps=2, max_displacement_m=2.25)
        memory.update(
            step_id=4, position=(0, 0, 0), direct_confirmed=True,
            current_visual_support=True, instance_key="target-1",
        )
        uncertain = memory.update(
            step_id=5, position=(1, 0, 0), direct_confirmed=False,
            current_visual_support=True, instance_key="target-2",
            confirmed_instance_switch=False,
        )
        self.assertFalse(uncertain["confirmed"])
        self.assertEqual(
            uncertain["cache_cleared_reason"],
            "uncertain_instance_switch_cache_retained",
        )
        self.assertEqual(uncertain["cached_instance_key"], "target-1")

    def test_distinctive_context_bridges_projection_jitter(self):
        tracker = TerminalInstanceTracker(max_position_delta_m=2.25)
        tracker.reset("ep-1")
        first = tracker.update(
            step_id=1, position=(0, 0, 0), heading=0, direction_id=0,
            depth_m=2.0, target_terms=("entrance to dining room",),
            supporting_landmarks=("white sofas", "fireplace"),
            visually_supported=True,
        )
        second = tracker.update(
            step_id=2, position=(0, 0, 0), heading=0, direction_id=6,
            depth_m=2.0, target_terms=("entrance to dining room",),
            supporting_landmarks=("white sofas", "fireplace"),
            visually_supported=True,
        )
        self.assertEqual(first["instance_key"], second["instance_key"])
        self.assertTrue(second["distinctive_context_match"])
        self.assertEqual(second["consecutive_observations"], 2)

    def test_weak_generic_does_not_use_context_to_relax_spatial_limit(self):
        tracker = TerminalInstanceTracker(max_position_delta_m=2.25)
        tracker.reset("ep-1")
        first = tracker.update(
            step_id=1, position=(0, 0, 0), heading=0, direction_id=0,
            depth_m=2.0, target_terms=("archway",),
            supporting_landmarks=("pool table",), visually_supported=True,
        )
        second = tracker.update(
            step_id=2, position=(0, 0, 0), heading=0, direction_id=6,
            depth_m=2.0, target_terms=("archway",),
            supporting_landmarks=("pool table",), visually_supported=True,
        )
        self.assertNotEqual(first["instance_key"], second["instance_key"])
        self.assertTrue(second["weak_generic_target"])

    def test_weak_generic_close_confirm_is_strictly_gated(self):
        result = {
            "required_landmark_terms": ["archway"],
            "all_landmark_terms": ["floor", "archway"],
            "arrival_evidence": True,
            "confidence": 0.95,
            "direction_aligned_terminal_support": {"aligned": True},
            "aggregate_current_view_corroboration": {
                "uncorroborated_terms": []
            },
        }
        self.assertTrue(weak_generic_close_is_confirmed(
            result, route_progress_complete=True,
            route_maturity_satisfied=True, local_surface_depth_m=1.4,
        ))
        self.assertFalse(weak_generic_close_is_confirmed(
            result, route_progress_complete=False,
            route_maturity_satisfied=True, local_surface_depth_m=1.4,
        ))
        self.assertFalse(weak_generic_close_is_confirmed(
            result, route_progress_complete=True,
            route_maturity_satisfied=True, local_surface_depth_m=1.6,
        ))
        result["aggregate_current_view_corroboration"] = {
            "uncorroborated_terms": ["archway"]
        }
        self.assertFalse(weak_generic_close_is_confirmed(
            result, route_progress_complete=True,
            route_maturity_satisfied=True, local_surface_depth_m=1.4,
        ))
        result["aggregate_current_view_corroboration"] = {
            "uncorroborated_terms": []
        }
        result["terminal_relation_evidence"] = {
            "required": True, "satisfied": False,
        }
        self.assertFalse(weak_generic_close_is_confirmed(
            result, route_progress_complete=True,
            route_maturity_satisfied=True, local_surface_depth_m=1.4,
        ))

    def test_terminal_instance_tracker_matches_spatial_target(self):
        tracker = TerminalInstanceTracker(max_position_delta_m=0.5)
        tracker.reset("ep-1")
        first = tracker.update(
            step_id=1, position=(0, 0, 0), heading=0, direction_id=0,
            depth_m=2.0, target_terms=("archway",),
            supporting_landmarks=("sofa",), visually_supported=True,
        )
        second = tracker.update(
            step_id=2, position=(0, 0, 0.5), heading=0, direction_id=0,
            depth_m=1.5, target_terms=("archway",),
            supporting_landmarks=("sofa",), visually_supported=True,
        )
        changed = tracker.update(
            step_id=3, position=(0, 0, 0.5), heading=0, direction_id=6,
            depth_m=1.5, target_terms=("archway",),
            supporting_landmarks=("sofa",), visually_supported=True,
        )
        self.assertEqual(first["instance_key"], second["instance_key"])
        self.assertEqual(second["consecutive_observations"], 2)
        self.assertNotEqual(second["instance_key"], changed["instance_key"])
        self.assertTrue(changed["instance_changed"])
        self.assertEqual(changed["consecutive_observations"], 1)

    def test_terminal_instance_tracker_abstains_without_geometry(self):
        tracker = TerminalInstanceTracker()
        tracker.reset("ep-1")
        result = tracker.update(
            step_id=1, position=(0, 0, 0), heading=0, direction_id=None,
            depth_m=None, target_terms=("sofa",),
            supporting_landmarks=(), visually_supported=True,
        )
        self.assertIsNone(result["instance_key"])
        self.assertEqual(result["reason"], "insufficient_instance_geometry")

    def test_regression_fails_loudly(self):
        self.provider.update(
            {"j": 2, "n_anchors": 3, "ever_satisfied": [0, 1], "degenerate": False}
        )
        with self.assertRaises(ValueError):
            self.provider.update(
                {"j": 1, "n_anchors": 3, "ever_satisfied": [0], "degenerate": False}
            )

    def test_update_before_reset_fails(self):
        provider = ACNL1ProgressProvider(FakeReducer())
        with self.assertRaises(RuntimeError):
            provider.update({"j": 0, "n_anchors": 1, "degenerate": False})

    def test_trainer_has_no_llm_completion_call_or_legacy_provider(self):
        source = (
            ROOT / "vlnce_baselines/common/base_il_trainer_llm.py"
        ).read_text(encoding="utf-8")
        self.assertNotIn("estimate_completion(", source)
        self.assertNotIn("build_legacy_progress_update", source)
        self.assertIn("progress_provider.update(", source)
        self.assertIn('provider="acn_l1"', source)

    def test_instruction_plan_failure_is_episode_local_forced_termination(self):
        source = (
            ROOT / "vlnce_baselines/common/base_il_trainer_llm.py"
        ).read_text(encoding="utf-8")
        handler = source.split("except ValueError as plan_exc:", 1)[1].split(
            "if instruction_plan_failure is None:", 1
        )[0]
        self.assertNotIn("envs.close()", handler)
        self.assertNotIn("raise", handler)
        self.assertIn('failure_scope="episode"', handler)
        self.assertIn(
            'forced_termination_source = "instruction_plan_failed"', source
        )
        self.assertIn("stop_termination_reason = (", source)
        self.assertIn(
            "# A plan-validation failure is an evaluation failure", source
        )


if __name__ == "__main__":
    unittest.main()
