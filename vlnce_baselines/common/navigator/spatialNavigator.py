import re
import math
import random
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
    def get_actions(self, instruction):
        return self.llm.gpt_infer(ACTION_DETECTION['system'], ACTION_DETECTION['user'].format(instruction))

    def get_landmarks(self, actions):
        actions = actions.replace("\n", " ")
        return self.llm.gpt_infer(LANDMARK_DETECTION['system'], LANDMARK_DETECTION['user'].format(actions))
    
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
    def save_history(self, logger, current_step, next_vp, thought, curr_observe, nav_history): 
        # ===== get obervation summary =====
        direction_id = int(curr_observe.split("Direction Viewpoint")[0].replace("Direction","").strip())
        direction = DIRECTIONS[direction_id]
        curr_observe = "Scene Description"+curr_observe.split("Scene Description")[1]
        observation = f"Direction {direction} " + self.llm.gpt_infer(OBSERVATION_SUMMARY['system'], OBSERVATION_SUMMARY['user'].format(curr_observe))
        # ===== get thought summary =====
        thought = self.llm.gpt_infer(THOUGHT_SUMMARY['system'], THOUGHT_SUMMARY['user'].format(thought))
        # ===== get nav history =====
        nav_history.append({
            "step": current_step,
            "viewpoint": next_vp,
            "observation": observation,
            "thought": thought
        })
        logger.info(f"The history at current step is {nav_history}")
        return nav_history
    
    def review_history(self, logger, nav_history):
        nav_history_str = " -> ".join(["Step "+str(idx+1)+" Observation: "+item["observation"]+" Thought: "+item["thought"] for idx, item in enumerate(nav_history)])
        logger.info("History: " + nav_history_str)
        return nav_history_str
    
    def estimate_completion(
        self,
        logger,
        actions,
        landmarks,
        history_traj,
        max_tokens=None,
    ):
        response = self.llm.gpt_infer(
            COMPLETION_ESTIMATION['system'],
            COMPLETION_ESTIMATION['user'].format(history_traj, landmarks, actions),
            max_tokens=max_tokens,
        )
        executed_markers = list(
            re.finditer(r"\bExecuted Actions\s*:?", response, flags=re.I)
        )
        if executed_markers:
            logger.info("Executed Actions " + response)
            marker = executed_markers[-1]
            executed = response[marker.end():].strip()
            thought_marker = re.search(r"\bThought\s*:", executed, flags=re.I)
            if thought_marker:
                executed = executed[:thought_marker.start()].strip()
            return executed.strip()
        logger.info("Completion estimator response missing Executed Actions marker: " + response)
        recovered = self._recover_executed_actions_from_response(actions, response)
        if recovered:
            logger.info("Recovered executed actions from unstructured completion response: " + recovered)
        return recovered

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
    ):
        # P1: k>1 draws multiple samples so the (dormant) thought_fusion can arbitrate;
        # temperature>0 gives diversity. k=1, temperature=0 -> byte-identical to prior behavior.
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
        ) + move_back_hint
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
        matched_dict = dict()
        for pred, thought in zip(predictions, thoughts):
            if pred not in matched_dict.keys():
                matched_dict[pred] = []
            matched_dict[pred].append(thought)

        if len(matched_dict) <= 1:
            skipped_dict = {}
            for key, value in matched_dict.items():
                skipped_dict[key] = value[0] if value else ""
            logger.info("Skip thought fusion because there is no candidate disagreement")
            return skipped_dict

        for key, value in matched_dict.items():
            multiple_thoughts = "; ".join(["Thought "+str(idx+1)+": "+thought for idx, thought in enumerate(value)])
            one_thought = self.llm.gpt_infer(
                THOUGHT_FUSION['system'],
                THOUGHT_FUSION['user'].format(multiple_thoughts),
                max_tokens=max_tokens,
            )
            logger.info(f"Pred viewpoint ID: {key} Fused Thought: {one_thought}")
            matched_dict[key] = one_thought 
        return matched_dict 
    
    def test_decisions(
        self,
        logger,
        fused_pred_thought,
        observation,
        instruction,
        error_number,
        observe_dict,
        max_tokens=None,
    ):
        try:
            valid_candidates = {str(key) for key in observe_dict.keys()}
            valid_candidates.add(STOP_CANDIDATE)
            for fused_key in list(fused_pred_thought.keys()):
                if fused_key not in valid_candidates:
                    fused_pred_thought.pop(fused_key)
                    
            if not fused_pred_thought:
                raise ValueError("Error in fused_thought key")
                
            if len(fused_pred_thought.keys()) == 1:
                for key, value in fused_pred_thought.items():
                    self.last_test_decision_metadata = {
                        "fallback_used": False,
                        "decision_source": "single_fused_candidate",
                        "selected_candidate": key,
                        "available_candidates": [str(candidate) for candidate in observe_dict.keys()],
                    }
                    return key, value, error_number
            else:
                fused_pred_thought_ = "; ".join(["Direction Viewpoint ID: "+key+" Thought: "+value for key, value in fused_pred_thought.items()])
                next_vp = None
                for i in range(2): 
                    logger.info(f"========== {i} retry in test decision==========")
                    next_vp = self.llm.gpt_infer(
                        DECISION_TEST['system'],
                        DECISION_TEST['user'].format(
                            fused_pred_thought.keys(),
                            observation,
                            instruction,
                            fused_pred_thought_,
                        ),
                        max_tokens=max_tokens,
                    )
                    logger.info(f"Next predicted action is {next_vp}")
                    next_vp = self._parse_prediction(next_vp, valid_candidates)
                    if next_vp in fused_pred_thought:
                        break
                if next_vp not in fused_pred_thought:
                    raise ValueError("Decision test did not return a valid fused candidate")
        
            logger.info(f"In test decision the predicted direction: {next_vp}")
            logger.info(f"In test decision the predicted thought: {fused_pred_thought[next_vp]}")
            self.last_test_decision_metadata = {
                "fallback_used": False,
                "decision_source": "decision_test_llm",
                "selected_candidate": next_vp,
                "available_candidates": [str(candidate) for candidate in observe_dict.keys()],
                "fused_candidates": [str(candidate) for candidate in fused_pred_thought.keys()],
            }
            return next_vp, fused_pred_thought[next_vp], error_number
        except Exception as e:
            logger.info(f"Error in test decision {e}")
            error_number += 1
            logger.info(f"Error number is {error_number}")
            next_vp, thought, fallback_metadata = self._fallback_candidate(logger, fused_pred_thought, observe_dict)
            self.last_test_decision_metadata = {
                "fallback_used": True,
                "fallback_source": "test_decisions_exception",
                "error": str(e),
                "error_number": error_number,
                **fallback_metadata,
            }
            return next_vp, thought, error_number
