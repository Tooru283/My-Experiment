import re
import random
from vlnce_baselines.common.navigator.api import *
from vlnce_baselines.common.navigator.prompts import *

STOP_CANDIDATE = "STOP"
STOP_WORDS = ("stop", "wait", "stay", "stand", "pause")

class Open_Nav():
    def __init__(self, device, llm_type, api_key):
        self.device = device
        self.llm = llmClient(llm_type, api_key)
        self.spatial = spatialClient(self.device)
        
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
    def observe_environment(self, logger, current_step, images_list):        
        observe_results = []
        observe_dict = {}
        for direction_idx, direction_image in images_list.items(): 
            observe_result = self.spatial.observe_view(logger, current_step, direction_idx, direction_image)
            logger.info(observe_result)
            observe_results.append(observe_result) 
            observe_dict[direction_idx] = observe_result
        return observe_results, observe_dict
    
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
        if "Executed Actions" in response:
            logger.info("Executed Actions " + response)
            marker_idx = response.rfind("Executed Actions")
            executed = response[marker_idx + len("Executed Actions"):].strip()
            return executed.lstrip(":").strip()
        else:
            return response

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
        normalized = self._normalize_action_text(landmarks)
        skip_words = {"and", "or", "the", "a", "an", "near", "before", "after", "around", "to", "of"}
        return [
            word for word in normalized.split()
            if len(word) > 2 and word not in skip_words
        ]

    def _final_landmark_visible(self, landmarks, observation):
        landmark_words = list(dict.fromkeys(self._split_landmarks(landmarks)))
        if not landmark_words:
            return True
        normalized_observation = self._normalize_action_text(observation)
        observation_words = set(normalized_observation.split())
        matched_landmarks = [
            word for word in landmark_words
            if word in observation_words
        ]
        required_matches = 1 if len(landmark_words) == 1 else 2
        return len(matched_landmarks) >= required_matches

    def should_stop(self, logger, actions, landmarks, estimation, history_traj, observation):
        if not history_traj or history_traj == "Step 0 start position. ":
            return False, ""

        action_items = self._split_actions(actions)
        if not action_items:
            return False, ""

        normalized_actions = [self._normalize_action_text(action) for action in action_items]
        normalized_estimation = self._normalize_action_text(estimation)
        if not normalized_estimation:
            return False, ""

        completed_actions = [
            action for action in normalized_actions
            if action and action in normalized_estimation
        ]
        final_action = normalized_actions[-1]
        final_action_completed = final_action and final_action in normalized_estimation
        all_actions_completed = len(completed_actions) == len(normalized_actions)
        final_landmark_visible = self._final_landmark_visible(landmarks, observation)

        if not final_landmark_visible:
            return False, ""

        if all_actions_completed:
            reason = "Completion estimator indicates all decomposed actions have been executed and the final landmark is visible."
            logger.info(f"Stop decision: {reason}")
            return True, reason

        if self._is_stop_action(action_items[-1]) and final_action_completed:
            reason = "Completion estimator indicates the final stop/wait action has been executed and the final landmark is visible."
            logger.info(f"Stop decision: {reason}")
            return True, reason

        return False, ""

    def _parse_prediction(self, prediction_text, candidate_ids):
        prediction_text = str(prediction_text or "").strip().replace("\"", "").replace("'", "")
        normalized_prediction = re.sub(r"[^A-Za-z0-9]+", " ", prediction_text).strip().upper()
        if normalized_prediction == STOP_CANDIDATE:
            return STOP_CANDIDATE
        match = re.search(r"\d+", prediction_text)
        if not match:
            return None
        pred_vp = match.group()
        return pred_vp if pred_vp in candidate_ids else None

    def _fallback_candidate(self, logger, fused_pred_thought, observe_dict):
        valid_candidates = {str(key) for key in observe_dict.keys()}
        for key, thought in fused_pred_thought.items():
            if key == STOP_CANDIDATE:
                logger.info("Fallback selected STOP from fused predictions")
                return STOP_CANDIDATE, thought
            if key in valid_candidates:
                logger.info(f"Fallback selected valid fused candidate {key}")
                return key, thought

        fallback = next(iter(observe_dict.keys()))
        logger.info(f"Fallback selected first observed candidate {fallback}")
        return str(fallback), observe_dict[fallback]
    
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
    ):
        break_flag = True
        effective_prediction, thought_list = [], []
        candidate_ids = [str(key) for key in observe_dict.keys()]
        candidate_list = candidate_ids + [STOP_CANDIDATE]
        decision_reasoning = self.llm.gpt_infer(
            NAVIGATOR['system'],
            NAVIGATOR['user'].format(
                ", ".join(candidate_list),
                current_step,
                instruction,
                actions,
                landmarks,
                history_traj,
                estimation,
                observation,
            ),
            max_tokens=max_tokens,
        )
        logger.info(decision_reasoning)
        if "Prediction:" in decision_reasoning:
            pred_thought = decision_reasoning.split("Prediction:")[0].strip()
            pred_vp = self._parse_prediction(decision_reasoning.split("Prediction:")[1], candidate_ids)
            if pred_vp is None:
                logger.info("Ignore invalid predicted viewpoint")
            else:
                effective_prediction.append(pred_vp)
                thought_list.append(pred_thought)
        return effective_prediction, thought_list, break_flag
    
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
            return next_vp, fused_pred_thought[next_vp], error_number
        except Exception as e:
            logger.info(f"Error in test decision {e}")
            error_number += 1
            logger.info(f"Error number is {error_number}")
            next_vp, thought = self._fallback_candidate(logger, fused_pred_thought, observe_dict)
            return next_vp, thought, 0
