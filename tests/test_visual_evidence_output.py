import json
from types import SimpleNamespace
import unittest

from PIL import Image
from vlnce_baselines.common.opennav_ext.visual_evidence import (
    VisualEvidenceLogger,
    _validated_stop_evidence,
)


class StopVisualEvidenceOutputTest(unittest.TestCase):
    def valid(self):
        return {
            "candidates": [{
                "candidate_id": "__current_view__",
                "visible_landmarks": ["archway"] * 20 + ["floor"],
                "matched_instruction_terms": ["archway"] * 4,
                "missing_instruction_terms": [],
                "final_target_visible": True,
                "arrival_evidence": True,
                "target_direction_id": "6",
                "spatial_notes": "near",
                "confidence": 0.9,
            }]
        }

    def test_repeated_fields_are_deduplicated_and_bounded(self):
        candidate = _validated_stop_evidence(self.valid())["candidates"][0]
        self.assertEqual(candidate["visible_landmarks"], ["archway", "floor"])
        self.assertEqual(candidate["matched_instruction_terms"], ["archway"])
        self.assertEqual(candidate["target_direction_id"], 6)

    def test_invalid_single_candidate_contract_is_rejected(self):
        for value in (
            {"candidates": []},
            {"candidates": [{}, {}]},
        ):
            with self.subTest(value=value), self.assertRaises(ValueError):
                _validated_stop_evidence(value)

    def test_visibility_false_cannot_keep_arrival_true(self):
        value = self.valid()
        item = value["candidates"][0]
        item["final_target_visible"] = False
        item["arrival_evidence"] = True
        normalized = _validated_stop_evidence(value)["candidates"][0]
        self.assertFalse(normalized["arrival_evidence"])

    def test_bad_direction_is_rejected(self):
        value = self.valid()
        value["candidates"][0]["target_direction_id"] = 12
        with self.assertRaises(ValueError):
            _validated_stop_evidence(value)

    def test_malformed_current_view_output_gets_one_short_retry(self):
        valid = self.valid()
        valid["candidates"][0]["visible_landmarks"] = ["archway"]
        replies = iter(["```json\n{truncated", json.dumps(valid)])

        class Completions:
            def __init__(self):
                self.calls = []

            def create(self, **kwargs):
                self.calls.append(kwargs)
                return SimpleNamespace(choices=[
                    SimpleNamespace(message=SimpleNamespace(content=next(replies)))
                ])

        completions = Completions()
        logger = VisualEvidenceLogger.__new__(VisualEvidenceLogger)
        logger.base_url = "local"
        logger.model = "test"
        logger.max_candidates = 4
        logger.max_image_edge = 32
        logger.max_tokens = 768
        logger.image_jpeg_quality = 70
        logger.metadata_observation_chars = 300
        logger.compact_json = True
        logger.client = SimpleNamespace(
            chat=SimpleNamespace(completions=completions)
        )
        result = logger.run(
            instruction="Wait by the archway",
            actions="Wait by the archway",
            landmarks="archway",
            candidates=[],
            images_dict={"0": {"rgb": Image.new("RGB", (16, 16), "black")}},
            observe_dict={"0": "Scene Objects: archway"},
            stop_current_view_evidence=True,
        )
        self.assertIsNone(result["parse_error"])
        self.assertEqual(result["retry_count"], 1)
        self.assertEqual(
            [item["valid"] for item in result["attempt_diagnostics"]],
            [False, True],
        )
        self.assertEqual(len(completions.calls), 2)
        self.assertEqual(completions.calls[1]["max_tokens"], 256)



if __name__ == "__main__":
    unittest.main()
