import base64
import io
import json
import re
from typing import Any, Dict, Iterable, List, Optional

from openai import OpenAI
from PIL import Image

from vlnce_baselines.common.opennav_ext.agent_state import CandidateState
from vlnce_baselines.common.opennav_ext.visual_evidence_schema import (
    normalize_visual_evidence_parsed,
)


DEFAULT_VISUAL_EVIDENCE_BASE_URL = "http://127.0.0.1:23333/v1"
DEFAULT_VISUAL_EVIDENCE_MODEL = "/root/models/Qwen3.5-4B"


def _candidate_id(candidate: Any) -> str:
    return str(getattr(candidate, "candidate_id", candidate))


def _image_to_data_url(
    image: Image.Image,
    max_edge: int,
    jpeg_quality: int,
) -> str:
    if image.mode != "RGB":
        image = image.convert("RGB")
    if max_edge > 0:
        width, height = image.size
        scale = min(float(max_edge) / float(max(width, height)), 1.0)
        if scale < 1.0:
            image = image.resize(
                (max(1, int(width * scale)), max(1, int(height * scale))),
                Image.BICUBIC,
            )
    buffer = io.BytesIO()
    quality = max(1, min(int(jpeg_quality), 95))
    image.save(buffer, format="JPEG", quality=quality)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return "data:image/jpeg;base64,{}".format(encoded)


def _extract_json(text: str) -> Dict[str, Any]:
    raw = str(text or "").strip()
    fenced = re.search(r"```(?:json)?\s*(.*?)```", raw, flags=re.S | re.I)
    if fenced:
        raw = fenced.group(1).strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", raw, flags=re.S)
        if match:
            return json.loads(match.group(0))
        raise


class VisualEvidenceLogger:
    """Logging-only Qwen-VL visual evidence extractor.

    This tool must not alter candidate ranking, selector output, stop decisions,
    or env actions. It only writes structured visual evidence into trace logs.
    """

    def __init__(
        self,
        base_url: str = DEFAULT_VISUAL_EVIDENCE_BASE_URL,
        model: str = DEFAULT_VISUAL_EVIDENCE_MODEL,
        api_key: str = "not-needed",
        max_candidates: int = 4,
        max_image_edge: int = 384,
        max_tokens: int = 384,
        timeout_seconds: float = 60.0,
        image_jpeg_quality: int = 80,
        metadata_observation_chars: int = 700,
        compact_json: bool = False,
    ) -> None:
        self.base_url = base_url
        self.model = model
        self.max_candidates = max_candidates
        self.max_image_edge = max_image_edge
        self.max_tokens = max_tokens
        self.image_jpeg_quality = image_jpeg_quality
        self.metadata_observation_chars = metadata_observation_chars
        self.compact_json = compact_json
        self.client = OpenAI(
            api_key=api_key or "not-needed",
            base_url=base_url,
            timeout=timeout_seconds,
        )

    def run(
        self,
        instruction: str,
        actions: str,
        landmarks: str,
        candidates: Iterable[CandidateState],
        images_dict: Dict[str, Dict[str, Image.Image]],
        observe_dict: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        all_candidates = list(candidates)
        total_candidate_ids = [_candidate_id(candidate) for candidate in all_candidates]
        selected = []
        selection_reason_by_candidate: Dict[str, str] = {}
        for candidate in all_candidates:
            candidate_id = _candidate_id(candidate)
            if candidate_id not in images_dict:
                selection_reason_by_candidate[candidate_id] = "no_image"
                continue
            selected.append(candidate)
            selection_reason_by_candidate[candidate_id] = "first_available"
            if len(selected) >= self.max_candidates:
                break
        for candidate in all_candidates:
            candidate_id = _candidate_id(candidate)
            if candidate_id not in selection_reason_by_candidate:
                selection_reason_by_candidate[candidate_id] = "max_candidates_limit"

        if not selected:
            return {
                "skipped": True,
                "reason": "no_candidate_images",
                "model": self.model,
                "results": [],
                "total_candidate_ids": total_candidate_ids,
                "requested_candidate_ids": [],
                "selection_reason_by_candidate": selection_reason_by_candidate,
                "sampled_all": len(total_candidate_ids) == 0,
            }

        content: List[Dict[str, Any]] = [
            {
                "type": "text",
                "text": self._build_prompt(
                    instruction,
                    actions,
                    landmarks,
                    selected,
                    observe_dict or {},
                ),
            }
        ]
        requested_candidate_ids = []
        for index, candidate in enumerate(selected, start=1):
            candidate_id = _candidate_id(candidate)
            requested_candidate_ids.append(candidate_id)
            rgb_image = images_dict[candidate_id]["rgb"]
            content.append(
                {
                    "type": "text",
                    "text": "\nCandidate image {} maps to candidate_id={}.\n".format(
                        index, candidate_id
                    ),
                }
            )
            content.append(
                {
                    "type": "image_url",
                    "image_url": {
                        "url": _image_to_data_url(
                            rgb_image,
                            self.max_image_edge,
                            self.image_jpeg_quality,
                        )
                    },
                }
            )

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": content}],
            temperature=0,
            max_tokens=self.max_tokens,
        )
        raw_response = response.choices[0].message.content
        parse_error = None
        parsed_root_type = None
        parsed: Dict[str, Any] = {}
        try:
            parsed = _extract_json(raw_response)
            parsed_root_type = type(parsed).__name__
        except Exception as exc:
            parse_error = "{}: {}".format(type(exc).__name__, exc)
        parsed = normalize_visual_evidence_parsed(parsed)

        return {
            "skipped": False,
            "model": self.model,
            "base_url": self.base_url,
            "total_candidate_ids": total_candidate_ids,
            "requested_candidate_ids": requested_candidate_ids,
            "selection_reason_by_candidate": selection_reason_by_candidate,
            "sampled_all": set(requested_candidate_ids) == set(total_candidate_ids),
            "sampled_candidate_count": len(requested_candidate_ids),
            "total_candidate_count": len(total_candidate_ids),
            "image_jpeg_quality": self.image_jpeg_quality,
            "metadata_observation_chars": self.metadata_observation_chars,
            "compact_json": self.compact_json,
            "parsed_root_type": parsed_root_type,
            "raw_response": raw_response,
            "parsed": parsed,
            "parse_error": parse_error,
        }

    def _build_prompt(
        self,
        instruction: str,
        actions: str,
        landmarks: str,
        candidates: Iterable[CandidateState],
        observe_dict: Dict[str, str],
    ) -> str:
        candidate_lines = []
        for candidate in candidates:
            candidate_id = _candidate_id(candidate)
            candidate_lines.append(
                "- candidate_id={}; angle_deg={}; distance={}; text_observation={}".format(
                    candidate_id,
                    getattr(candidate, "angle_deg", None),
                    getattr(candidate, "distance", None),
                    observe_dict.get(candidate_id, "")[
                        : max(0, int(self.metadata_observation_chars))
                    ],
                )
            )

        if self.compact_json:
            return (
                "Extract compact visual evidence for VLN candidate images. "
                "Do not choose actions or STOP. Use exact candidate_id values.\n"
                "Instruction: {}\nActions: {}\nLandmarks: {}\n"
                "Candidate metadata:\n{}\n"
                "For every candidate image, compare the image and metadata "
                "against the instruction, actions, and landmarks. Fill "
                "matched_instruction_terms with concrete required landmarks "
                "or objects that are visible/supported. Fill "
                "missing_instruction_terms with concrete required landmarks "
                "or objects that are not visible. Prefer terms copied from "
                "Instruction/Landmarks/Actions. Use [] only when truly none.\n"
                "Set final_target_visible=true only when the likely final "
                "destination/STOP target is visible. Set arrival_evidence=true "
                "only when the view appears at or immediately beside that "
                "target.\n"
                "The top-level JSON value must be an object with a "
                "\"candidates\" key. Do not return a bare array.\n"
                "Return minified JSON only, no markdown, no extra text. Schema: "
                '{{"candidates":[{{"candidate_id":"string",'
                '"visible_landmarks":["string"],'
                '"matched_instruction_terms":["string"],'
                '"missing_instruction_terms":["string"],'
                '"final_target_visible":false,'
                '"arrival_evidence":false,'
                '"spatial_notes":"<=40 chars",'
                '"confidence":0.0}}]}}\n'
                "Keep arrays short: at most 5 visible, 5 matched, 5 missing."
            ).format(
                instruction,
                actions,
                landmarks,
                "\n".join(candidate_lines),
            )

        return (
            "You are a visual evidence extraction tool for a VLN navigation "
            "harness. Do not choose an action and do not decide STOP.\n"
            "Use the provided candidate images only as evidence.\n\n"
            "Instruction:\n{}\n\n"
            "Decomposed actions:\n{}\n\n"
            "Landmarks:\n{}\n\n"
            "Candidate metadata:\n{}\n\n"
            "The top-level JSON value must be an object with a "
            "\"candidates\" key. Do not return a bare array.\n"
            "Return JSON only with this schema:\n"
            "{{\n"
            '  "candidates": [\n'
            "    {{\n"
            '      "candidate_id": "string",\n'
            '      "visible_landmarks": ["string"],\n'
            '      "matched_instruction_terms": ["string"],\n'
            '      "missing_instruction_terms": ["string"],\n'
            '      "final_target_visible": false,\n'
            '      "arrival_evidence": false,\n'
            '      "spatial_notes": "short string",\n'
            '      "confidence": 0.0\n'
            "    }}\n"
            "  ],\n"
            '  "global_notes": "short string"\n'
            "}}\n"
        ).format(
            instruction,
            actions,
            landmarks,
            "\n".join(candidate_lines),
        )
