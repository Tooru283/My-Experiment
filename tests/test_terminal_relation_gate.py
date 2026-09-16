import unittest

from vlnce_baselines.common.opennav_ext.visual_target_verifier import (
    VisualTargetVerifier,
)


class TerminalRelationGateTest(unittest.TestCase):
    def evidence(self, visible):
        return {
            "parsed": {"candidates": [{
                "candidate_id": "__current_view__",
                "visible_landmarks": visible,
                "matched_instruction_terms": visible,
                "missing_instruction_terms": [],
                "final_target_visible": True,
                "arrival_evidence": True,
                "target_direction_id": 0,
                "confidence": 0.95,
            }]},
            "requested_candidate_ids": ["__current_view__"],
            "total_candidate_ids": ["__current_view__"],
            "sampled_all": True,
            "parse_error": None,
            "schema_error": None,
        }

    def policy(self):
        return {
            "target_kind": "relational_region",
            "relations": [
                {"type": "between", "reference": "two white sofas"},
                {"type": "next_to", "reference": "entrance to the dining room"},
            ],
        }

    def verify(self, visible):
        return VisualTargetVerifier(confidence_threshold=0.9).verify(
            "selector_stop_gate",
            "Stop in the area between the two white sofas, next to the entrance to the dining room.",
            "Stop in the area between the two white sofas, next to the entrance to the dining room",
            "two white sofas\nentrance to the dining room\narea between the two white sofas",
            "completed",
            "history",
            [],
            self.evidence(visible),
            True,
            "selector proposed STOP",
            "__current_view__",
            8,
            "current_pano",
            self.policy(),
        )

    def test_missing_reference_blocks_relational_stop(self):
        result = self.verify([
            "area between the two white sofas", "two white sofas",
        ])
        self.assertEqual(result["verdict"], "uncertain")
        report = result["terminal_relation_evidence"]
        self.assertFalse(report["satisfied"])
        self.assertEqual(
            report["missing_references"], ["entrance to the dining room"]
        )
        self.assertTrue(any(
            item.startswith("terminal_relation_evidence_missing:")
            for item in result["allow_blockers"]
        ))

    def test_all_references_in_same_view_allow_relation_gate(self):
        result = self.verify([
            "area between the two white sofas",
            "two white sofas",
            "entrance to the dining room",
        ])
        self.assertEqual(result["verdict"], "allow")
        self.assertTrue(result["terminal_relation_evidence"]["satisfied"])


if __name__ == "__main__":
    unittest.main()
