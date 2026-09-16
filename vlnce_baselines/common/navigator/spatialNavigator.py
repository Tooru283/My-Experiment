import re
import math
import random
import json
import time
from collections import Counter
from vlnce_baselines.common.navigator.api import *
from vlnce_baselines.common.navigator.prompts import *
from vlnce_baselines.common.opennav_ext.landmark_matching import (
    final_landmark_terms,
    matched_terms as match_landmark_terms,
    missing_terms as missing_landmark_terms,
    split_landmark_terms,
    term_present,
)
from vlnce_baselines.common.opennav_ext.backtrack_policy import MOVE_BACK_CANDIDATE
from vlnce_baselines.common.opennav_ext.navigation_guidance import (
    format_navigation_feedback,
    build_navigation_history_item, format_navigation_history,
)
from vlnce_baselines.common.opennav_ext.instruction_parsing import parse_actions, parse_landmarks

STOP_CANDIDATE = "STOP"
STOP_WORDS = ("stop", "wait", "stay", "stand", "pause")
WEAK_FINAL_TARGET_TERMS = {
    "area",
    "archway",
    "doorway",
    "entry way",
    "entryway",
    "floor",
    "hall",
    "hallway",
    "room",
    "stair",
    "stairs",
    "staircase",
}

class Open_Nav():
    def __init__(self, device, llm_type, api_key):
        self.device = device
        self.llm = llmClient(llm_type, api_key)
        self.spatial = spatialClient(self.device)
        self.last_test_decision_metadata = {}
        self.last_stop_gate_metadata = {}
        self.block_weak_final_target_completion_stop = True
        self.min_steps_for_weak_final_target_completion_stop = 8
        
    # =====================================
    # ===== Instruction Comprehension =====
    # =====================================
    def get_plan(self, instruction):
        self.last_plan_metadata = {"llm_calls": 0, "stages": []}
        actions = self._extract_plan_stage(
            "action_extraction", ACTION_DETECTION,
            ACTION_DETECTION['user'].format(instruction), parse_actions, instruction,
        )
        landmarks = self._extract_plan_stage(
            "landmark_extraction", LANDMARK_DETECTION,
            LANDMARK_DETECTION['user'].format(instruction, json.dumps(actions, ensure_ascii=False)),
            parse_landmarks, instruction,
        )
        return "\n".join(actions), "\n".join(landmarks)

    def _extract_plan_stage(self, stage, prompt, user, parser, instruction):
        # One correction attempt per stage; invalid plans never enter the cache.
        correction = ""
        for attempt in range(2):
            started = time.perf_counter()
            self.last_plan_metadata["llm_calls"] += 1
            response = self.llm.gpt_infer(
                prompt['system'], user + correction, max_tokens=1536, temperature=0,
            )
            record = {"stage": stage, "attempt": attempt + 1,
                      "elapsed_seconds": round(time.perf_counter() - started, 4),
                      "raw_response": response}
            self.last_plan_metadata["stages"].append(record)
            try:
                parsed = parser(response, instruction)
            except ValueError as exc:
                record.update(valid=False, error=str(exc))
                if attempt == 1:
                    raise ValueError("{} failed after 2 attempts: {}".format(stage, exc)) from exc
                correction = "\nPrevious output failed validation: {}\nCorrect it using the original instruction; return only the required JSON.".format(exc)
            else:
                record.update(valid=True, parsed=parsed)
                return parsed

    # =============================
    # ===== Visual Perception =====
    # =============================
    def observe_environment(self, logger, current_step, images_list, distance_dict=None):
        observe_results = []
        observe_dict = {}
        for direction_idx, direction_image in images_list.items():
            observe_result = self.spatial.observe_view(logger, current_step, direction_idx, direction_image)
            observe_result = self._inject_waypoint_distance(observe_result, direction_idx, distance_dict)
            logger.info(observe_result)
            observe_results.append(observe_result)
            observe_dict[direction_idx] = observe_result
        return observe_results, observe_dict

    def _inject_waypoint_distance(self, observe_result, direction_idx, distance_dict):
        # P0: inject the WaypointBert sensor-measured distance to this direction's waypoint
        # so the navigator grounds on a measured value instead of the VLM-estimated distances
        # in the Scene Description (which are unreliable). The heading is already conveyed by
        # the direction id (see NAVIGATOR prompt), so only distance is added here.
        # See docs/architecture_optimization_20260701.md.
        if not distance_dict or direction_idx not in distance_dict:
            return observe_result
        try:
            wp_dist = round(float(distance_dict[direction_idx]), 1)
        except (TypeError, ValueError):
            return observe_result
        geo = f"[Waypoint distance: {wp_dist} m] "
        # Insert before "Scene Description" so the measured distance leads and the
        # "Scene Description" split used in save_history keeps working.
        marker = "Scene Description"
        if marker in observe_result:
            head, _, tail = observe_result.partition(marker)
            return f"{head}{geo}{marker}{tail}"
        return observe_result + " " + geo.strip()
    
    # ===================================
    # ===== Progress Estimation =========
    # ===================================
    def save_history(self, logger, current_step, next_vp, thought, curr_observe,
                     nav_history, action_receipt=None):
        nav_history.append(build_navigation_history_item(
            current_step, next_vp, thought, curr_observe, action_receipt,
        ))
        logger.info("The history at current step is %s", nav_history)
        return nav_history

    def review_history(self, logger, nav_history):
        text = format_navigation_history(nav_history)
        logger.info("History: " + text)
        return text

    def _normalize_action_text(self, text):
        text = str(text or "").lower()
        text = re.sub(r"[^a-z0-9]+", " ", text)
        return re.sub(r"\s+", " ", text).strip()

    def _split_actions(self, actions):
        chunks = []
        for line in str(actions or "").replace(";", "\n").splitlines():
            line = line.strip()
            if not line:
                continue
            chunks.extend(part.strip() for part in line.split(","))

        cleaned = []
        for chunk in chunks:
            chunk = re.sub(r"^\s*(?:[-*]|\d+[\.\)]|action\s*\d+\s*:)\s*", "", chunk, flags=re.I).strip()
            normalized = self._normalize_action_text(chunk)
            if normalized and normalized not in {"none", "no action", "not executed"}:
                cleaned.append(chunk)
        return cleaned

    def _is_stop_action(self, action):
        normalized = self._normalize_action_text(action)
        return any(word in normalized.split() for word in STOP_WORDS)

    def _split_landmarks(self, landmarks):
        return split_landmark_terms(landmarks)

    def _final_landmark_gate(self, landmarks, observation):
        all_landmark_terms = self._split_landmarks(landmarks)
        final_terms = final_landmark_terms(landmarks)
        if not final_terms:
            return {
                "visible": False,
                "reason": "empty_landmarks",
                "all_landmark_terms": all_landmark_terms,
                "required_landmark_terms": [],
                "matched_landmark_terms": [],
                "missing_landmark_terms": [],
            }
        matched_landmarks = match_landmark_terms(final_terms, [observation])
        missing_landmarks = missing_landmark_terms(final_terms, matched_landmarks)
        visible = bool(matched_landmarks)
        return {
            "visible": visible,
            "reason": "matched_final_landmark" if visible else "missing_final_landmark",
            "all_landmark_terms": all_landmark_terms,
            "required_landmark_terms": final_terms,
            "matched_landmark_terms": matched_landmarks,
            "missing_landmark_terms": missing_landmarks,
        }

    def _final_landmark_visible(self, landmarks, observation):
        return self._final_landmark_gate(landmarks, observation)["visible"]

    def _all_final_landmarks_weak(self, landmarks):
        final_terms = final_landmark_terms(landmarks)
        if not final_terms:
            return False
        return all(
            self._normalize_action_text(term) in WEAK_FINAL_TARGET_TERMS
            for term in final_terms
        )

    def _parse_executed_actions(self, estimation):
        estimation = str(estimation or "")
        thought_marker = re.search(r"\bThought\s*:", estimation, flags=re.I)
        if thought_marker:
            estimation = estimation[:thought_marker.start()].strip()
        executed_items = self._split_actions(estimation)
        executed_text = self._normalize_action_text(estimation)
        completed_numbers = set()
        raw_estimation = estimation.lower()
        negative_markers = (
            "not executed",
            "not completed",
            "not done",
            "has not",
            "have not",
            "not yet",
            "incomplete",
        )
        for match in re.finditer(r"\b(?:action|step)\s*(\d+)\b", raw_estimation):
            window_start = max(0, match.start() - 80)
            window_end = min(len(raw_estimation), match.end() + 80)
            local_window = raw_estimation[window_start:window_end]
            if any(marker in local_window for marker in negative_markers):
                continue
            completed_numbers.add(int(match.group(1)))
        return executed_items, executed_text, completed_numbers

    def _positive_completion_context(self, action, text):
        normalized_action = self._normalize_action_text(action)
        if not normalized_action:
            return False
        normalized_text = self._normalize_action_text(text)
        action_index = normalized_text.find(normalized_action)
        if action_index < 0:
            return False
        window_start = max(0, action_index - 120)
        window_end = min(
            len(normalized_text),
            action_index + len(normalized_action) + 120,
        )
        local_window = normalized_text[window_start:window_end]
        negative_markers = (
            "not executed",
            "not completed",
            "not done",
            "has not",
            "have not",
            "not yet",
            "incomplete",
            "cannot be considered",
            "cannot be done",
        )
        if any(marker in local_window for marker in negative_markers):
            return False
        positive_markers = (
            "executed",
            "completed",
            "done",
            "has been",
            "have been",
            "successfully",
            "reached",
            "arrived",
        )
        return any(marker in local_window for marker in positive_markers)

    def _positive_numbered_action_context(self, text):
        normalized_text = self._normalize_action_text(text)
        if not normalized_text:
            return False
        negative_markers = (
            "not executed",
            "not completed",
            "not done",
            "has not",
            "have not",
            "not yet",
            "incomplete",
            "cannot be considered",
            "cannot be done",
        )
        if any(marker in normalized_text for marker in negative_markers):
            return False
        positive_markers = (
            "executed",
            "completed",
            "done",
            "has been",
            "have been",
            "successfully",
            "reached",
            "arrived",
        )
        return any(marker in normalized_text for marker in positive_markers)

    def _recover_executed_actions_from_response(self, actions, response):
        action_items = self._split_actions(actions)
        if not action_items:
            return ""
        response_text = str(response or "")
        recovered = []
        for idx, action in enumerate(action_items, start=1):
            numbered_pattern = re.compile(
                r"\b(?:action|step)\s*{}\b".format(idx),
                flags=re.I,
            )
            for match in numbered_pattern.finditer(response_text):
                window_start = match.start()
                next_match = re.search(
                    r"\b(?:action|step)\s*\d+\b",
                    response_text[match.end():],
                    flags=re.I,
                )
                window_end = (
                    match.end() + next_match.start()
                    if next_match
                    else min(len(response_text), match.end() + 180)
                )
                local_window = response_text[window_start:window_end]
                if self._positive_completion_context(
                    action,
                    local_window,
                ) or self._positive_numbered_action_context(local_window):
                    recovered.append(f"{idx}. {action}")
                    break
        return "\n".join(recovered)

    def _has_negative_action_context(self, action, estimation):
        normalized_action = self._normalize_action_text(action)
        if not normalized_action:
            return False
        normalized_estimation = self._normalize_action_text(estimation)
        negative_markers = (
            "not executed",
            "not completed",
            "not done",
            "has not",
            "have not",
            "not yet",
            "incomplete",
        )
        action_index = normalized_estimation.find(normalized_action)
        if action_index < 0:
            return False
        window_start = max(0, action_index - 80)
        window_end = min(
            len(normalized_estimation),
            action_index + len(normalized_action) + 80,
        )
        local_window = normalized_estimation[window_start:window_end]
        return any(marker in local_window for marker in negative_markers)

    def _action_completed(self, action, action_index, executed_items, completed_numbers, estimation):
        if self._has_negative_action_context(action, estimation):
            return False
        normalized_action = self._normalize_action_text(action)
        if action_index in completed_numbers:
            return True
        for executed_item in executed_items:
            normalized_executed = self._normalize_action_text(executed_item)
            if not normalized_executed:
                continue
            if normalized_action == normalized_executed:
                return True
            if term_present(normalized_action, [normalized_executed]):
                return True
        return False

    def should_stop(
        self,
        logger,
        actions,
        landmarks,
        estimation,
        history_traj,
        observation,
        current_step=None,
    ):
        try:
            current_step_number = int(current_step)
        except (TypeError, ValueError):
            current_step_number = 0
        self.last_stop_gate_metadata = {
            "decision": False,
            "rejection_reason": None,
            "current_step": current_step_number,
            "completed_actions": [],
            "final_action_completed": False,
            "all_actions_completed": False,
            "weak_final_target": False,
            "weak_final_target_min_steps": (
                self.min_steps_for_weak_final_target_completion_stop
            ),
            "landmark_gate": None,
        }
        if not history_traj or history_traj == "Step 0 start position. ":
            self.last_stop_gate_metadata["rejection_reason"] = "empty_history"
            return False, ""

        action_items = self._split_actions(actions)
        if not action_items:
            self.last_stop_gate_metadata["rejection_reason"] = "empty_actions"
            return False, ""

        executed_items, _, completed_numbers = self._parse_executed_actions(estimation)
        if not executed_items and not completed_numbers:
            logger.info(
                "Stop gate rejected: no structured executed actions parsed."
            )
            self.last_stop_gate_metadata[
                "rejection_reason"
            ] = "no_structured_executed_actions"
            return False, ""

        completed_actions = [
            action for idx, action in enumerate(action_items, start=1)
            if self._action_completed(
                action,
                idx,
                executed_items,
                completed_numbers,
                estimation,
            )
        ]
        final_action_completed = self._action_completed(
            action_items[-1],
            len(action_items),
            executed_items,
            completed_numbers,
            estimation,
        )
        all_actions_completed = len(completed_actions) == len(action_items)
        landmark_gate = self._final_landmark_gate(landmarks, observation)
        final_landmark_visible = landmark_gate["visible"]
        weak_final_target = self._all_final_landmarks_weak(landmarks)
        self.last_stop_gate_metadata.update(
            {
                "completed_actions": completed_actions,
                "final_action_completed": final_action_completed,
                "all_actions_completed": all_actions_completed,
                "weak_final_target": weak_final_target,
                "current_step": current_step_number,
                "weak_final_target_min_steps": (
                    self.min_steps_for_weak_final_target_completion_stop
                ),
                "landmark_gate": landmark_gate,
            }
        )

        if not final_landmark_visible:
            logger.info(
                "Stop gate rejected by landmark gate: {}".format(landmark_gate)
            )
            self.last_stop_gate_metadata[
                "rejection_reason"
            ] = "missing_final_landmark"
            return False, ""

        if (
            weak_final_target
            and self.block_weak_final_target_completion_stop
            and self.min_steps_for_weak_final_target_completion_stop > 0
            and current_step_number
            < self.min_steps_for_weak_final_target_completion_stop
        ):
            logger.info(
                "Stop gate rejected weak final target for completion auto-stop: {}".format(
                    landmark_gate
                )
            )
            self.last_stop_gate_metadata[
                "rejection_reason"
            ] = "weak_final_target_completion_auto_stop"
            return False, ""

        if all_actions_completed:
            reason = "Completion estimator indicates all decomposed actions have been executed and the final landmark is visible."
            logger.info(
                "Stop decision: {} gate={}".format(reason, landmark_gate)
            )
            self.last_stop_gate_metadata["decision"] = True
            return True, reason

        if self._is_stop_action(action_items[-1]) and final_action_completed:
            reason = "Completion estimator indicates the final stop/wait action has been executed and the final landmark is visible."
            logger.info(
                "Stop decision: {} gate={}".format(reason, landmark_gate)
            )
            self.last_stop_gate_metadata["decision"] = True
            return True, reason

        self.last_stop_gate_metadata[
            "rejection_reason"
        ] = "completion_requirements_not_met"
        return False, ""

    def _parse_prediction(self, prediction_text, candidate_ids):
        prediction_text = str(prediction_text or "").strip().replace("\"", "").replace("'", "")
        normalized_prediction = re.sub(r"[^A-Za-z0-9]+", " ", prediction_text).strip().upper()
        if normalized_prediction == STOP_CANDIDATE:
            return STOP_CANDIDATE
        # Backtracking: accept MOVE_BACK only when it was offered this step (present in
        # candidate_ids). Normalization turns the underscore into a space, so restore it
        # before comparing. When not offered, MOVE_BACK is absent -> unchanged behavior.
        if (
            normalized_prediction.replace(" ", "_") == MOVE_BACK_CANDIDATE
            and MOVE_BACK_CANDIDATE in candidate_ids
        ):
            return MOVE_BACK_CANDIDATE
        match = re.search(r"\d+", prediction_text)
        if not match:
            return None
        pred_vp = match.group()
        return pred_vp if pred_vp in candidate_ids else None

    def _fallback_candidate(self, logger, fused_pred_thought, observe_dict):
        valid_candidates = {str(key) for key in observe_dict.keys()}
        available_candidates = [str(key) for key in observe_dict.keys()]
        if not available_candidates:
            logger.info("Fallback failed because no observed candidates are available")
            return STOP_CANDIDATE, "", {
                "fallback_strategy": "navigator_internal",
                "fallback_reason": "no_available_candidates",
                "available_candidates": [],
                "selected_candidate": STOP_CANDIDATE,
                "ranked_candidates": [],
            }
        for key, thought in fused_pred_thought.items():
            if key == STOP_CANDIDATE:
                logger.info("Fallback selected STOP from fused predictions")
                return STOP_CANDIDATE, thought, {
                    "fallback_strategy": "navigator_internal",
                    "fallback_reason": "fused_stop_candidate",
                    "available_candidates": available_candidates,
                    "selected_candidate": STOP_CANDIDATE,
                    "ranked_candidates": [],
                }
            if key in valid_candidates:
                logger.info(f"Fallback selected valid fused candidate {key}")
                return key, thought, {
                    "fallback_strategy": "navigator_internal",
                    "fallback_reason": "first_valid_fused_candidate",
                    "available_candidates": available_candidates,
                    "selected_candidate": key,
                    "ranked_candidates": [
                        {"candidate_id": str(candidate), "source": "fused_prediction"}
                        for candidate in fused_pred_thought.keys()
                    ],
                }

        fallback = next(iter(observe_dict.keys()))
        logger.info(f"Fallback selected first observed candidate {fallback}")
        return str(fallback), observe_dict[fallback], {
            "fallback_strategy": "navigator_internal",
            "fallback_reason": "first_observed_candidate",
            "available_candidates": available_candidates,
            "selected_candidate": str(fallback),
            "ranked_candidates": [
                {
                    "candidate_id": str(candidate),
                    "score": [0, -idx],
                    "source": "observation_order",
                }
                for idx, candidate in enumerate(available_candidates)
            ],
        }
    
    # =================================
    # ===== Move to next position =====
    # =================================
    def move_to_next_vp(
        self,
        logger,
        current_step,
        instruction,
        actions,
        landmarks,
        history_traj,
        estimation,
        observation,
        observe_dict,
        max_tokens=None,
        k=1,
        temperature=0,
        return_prompt=False,
        offer_move_back=False,
        spatial_alert="",
        progress_feedback=None,
    ):
        # k=1 by default: one selector request. Optional k>1 samples are grouped
        # by deterministic vote, with no extra fusion/arbitration model requests.
        break_flag = True
        effective_prediction, thought_list = [], []
        candidate_ids = [str(key) for key in observe_dict.keys()]
        candidate_list = candidate_ids + [STOP_CANDIDATE]
        # Backtracking (default off -> byte-identical to prior behavior): when the trainer
        # offers a backtrack this step, expose MOVE_BACK as an extra action (candidate list,
        # parse whitelist, and a neutral prompt line). offer_move_back=False leaves all three
        # untouched, so the prompt and accepted-action set are unchanged.
        parse_candidate_ids = candidate_ids
        move_back_hint = ""
        if offer_move_back:
            candidate_list = candidate_list + [MOVE_BACK_CANDIDATE]
            parse_candidate_ids = candidate_ids + [MOVE_BACK_CANDIDATE]
            move_back_hint = MOVE_BACK_PROMPT_LINE
        user_prompt = NAVIGATOR['user'].format(
            ", ".join(candidate_list),
            current_step,
            instruction,
            actions,
            landmarks,
            history_traj,
            estimation,
            observation,
        ) + format_navigation_feedback(progress_feedback) + move_back_hint + (spatial_alert or "")
        # C5 (GTA S_alert): spatial_alert is "" unless MEMORY_DIAGNOSTIC is decision-active,
        # so the assembled prompt stays byte-identical when the switch is off.
        for _ in range(max(1, k)):
            decision_reasoning = self.llm.gpt_infer(
                NAVIGATOR['system'],
                user_prompt,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            logger.info(decision_reasoning)
            if "Prediction:" in decision_reasoning:
                pred_thought = decision_reasoning.split("Prediction:")[0].strip()
                pred_vp = self._parse_prediction(decision_reasoning.split("Prediction:")[1], parse_candidate_ids)
                if pred_vp is None:
                    logger.info("Ignore invalid predicted viewpoint")
                else:
                    effective_prediction.append(pred_vp)
                    thought_list.append(pred_thought)
        if return_prompt:
            return effective_prediction, thought_list, break_flag, user_prompt
        return effective_prediction, thought_list, break_flag

    @staticmethod
    def vote_dispersion(preds):
        # Normalized vote entropy in [0,1]. 0 = unanimous, 1 = all k samples differ. denom=log(k).
        if not preds:
            return None
        n = len(preds)
        counts = Counter(preds)
        H = -sum((v / n) * math.log(v / n) for v in counts.values())
        return H / math.log(n) if n > 1 else 0.0
    
    # =========================
    # ===== Test Decision =====
    # =========================
    def thought_fusion(self, logger, predictions, thoughts, max_tokens=None):
        # Legacy method name retained for trace/caller compatibility; no model call.
        # Stable majority order; a tie keeps the first valid sample's order.
        grouped = {}
        for pred, thought in zip(predictions, thoughts):
            grouped.setdefault(str(pred), []).append(str(thought or ""))
        ranked = sorted(grouped, key=lambda key: -len(grouped[key]))
        fused = {key: grouped[key][0] for key in ranked}
        logger.info("Deterministic candidate vote counts: %s",
                    {key: len(grouped[key]) for key in ranked})
        return fused

    def test_decisions(
        self, logger, fused_pred_thought, observation, instruction, error_number,
        observe_dict, max_tokens=None, offer_move_back=False,
    ):
        # Input order is the stable vote ranking from thought_fusion. Never let a
        # second, progress-blind model overwrite the evidence-aware selector.
        valid = {str(key) for key in observe_dict}
        valid.add(STOP_CANDIDATE)
        if offer_move_back:
            valid.add(MOVE_BACK_CANDIDATE)
        eligible = {str(key): value for key, value in fused_pred_thought.items()
                    if str(key) in valid}
        if eligible:
            next_vp = next(iter(eligible))
            self.last_test_decision_metadata = {
                "fallback_used": False,
                "decision_source": "single_fused_candidate" if len(eligible) == 1
                else "deterministic_vote",
                "llm_calls": 0,
                "selected_candidate": next_vp,
                "available_candidates": [str(key) for key in observe_dict],
                "fused_candidates": list(eligible),
            }
            return next_vp, eligible[next_vp], error_number

        error_number += 1
        next_vp, thought, metadata = self._fallback_candidate(logger, {}, observe_dict)
        self.last_test_decision_metadata = {
            **metadata, "fallback_used": True, "fallback_source": "test_decisions_no_valid_candidate",
            "error_number": error_number, "llm_calls": 0,
        }
        return next_vp, thought, error_number
