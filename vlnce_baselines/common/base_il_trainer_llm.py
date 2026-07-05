import json
import sys
import jsonlines
import os
import time
import warnings
from collections import defaultdict
from datetime import datetime
from typing import Any, Dict, List
from PIL import Image
import requests
from openai import OpenAI

# for navigator      
from vlnce_baselines.common.navigator.spatialNavigator import *
from vlnce_baselines.common.opennav_ext import (
    ContextBuilder,
    FailureDiagnostic,
    ArrivalGate,
    GeometryQueryLogger,
    GrounderDiagnostic,
    MetricsLogger,
    MultimodalSelectorContext,
    PhaseAwareEvidenceScaffolder,
    PhaseEvidenceTracker,
    RecoveryPolicy,
    StopEvidenceVerifier,
    VisualEvidenceFallbackRanker,
    VisualEvidenceMemory,
    VisualEvidenceLogger,
    STOP_CURRENT_VIEW_CANDIDATE_ID,
    VisualTargetVerifier,
    VisualGraphMemoryDiagnostic,
    arrival_gate_config,
    arrival_gate_enabled,
    proactive_stop_gate_config,
    proactive_stop_gate_enabled,
    build_candidate_records,
    decision_effect_enabled,
    fail_open_enabled,
    get_trace_dir,
    harness_logging_enabled,
    module_enabled,
    module_log_only,
    selected_distance_gain,
    summarize_step_outputs,
    build_decision_audit,
    u_decision_effect_unit,
    u_module_enabled,
    u_module_log_only,
    u_series_enabled,
    validate_a1_harness_config,
)
import torch
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as distr
import torch.multiprocessing as mp
import gzip
import math
from copy import deepcopy

import tqdm
from gym import Space
from habitat import Config, logger
from habitat.utils.visualizations.utils import append_text_to_image
from habitat_baselines.common.base_il_trainer import BaseILTrainer
from habitat_baselines.common.baseline_registry import baseline_registry
from habitat_baselines.common.environments import get_env_class
from habitat_baselines.common.obs_transformers import (
    apply_obs_transforms_batch,
    apply_obs_transforms_obs_space,
    get_active_obs_transforms,
)
from habitat_extensions.measures import Position
from habitat_baselines.common.tensorboard_utils import TensorboardWriter
from habitat_baselines.utils.common import batch_obs, generate_video
from habitat_baselines.utils.common import (
    get_checkpoint_id,
    poll_checkpoint_folder,
)

from habitat_extensions.utils import observations_to_image
from vlnce_baselines.common.aux_losses import AuxLosses
from vlnce_baselines.common.env_utils import (
    construct_envs_auto_reset_false,
    construct_envs,
    is_slurm_batch_job,
)
from vlnce_baselines.common.utils import *

from habitat_extensions.measures import NDTW
from fastdtw import fastdtw

from ..utils import get_camera_orientations
from ..models.utils import (
    length2mask, dir_angle_feature, dir_angle_feature_with_ele,
)


def _episode_group_name(episode_count) -> str:
    try:
        value = int(episode_count)
    except (TypeError, ValueError):
        return "ep_unknown"
    if value < 0:
        return "ep_all"
    return "ep{}".format(value)


with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=FutureWarning)
    import tensorflow as tf  # noqa: F401

class BaseVLNCETrainerLLM(BaseILTrainer):
    r"""A base trainer for VLN-CE imitation learning."""
    supported_tasks: List[str] = ["VLN-v0"]

    def __init__(self, config=None):
        super().__init__(config)
        self.policy = None
        self.device = (
            torch.device("cuda", self.config.TORCH_GPU_ID)
            if torch.cuda.is_available()
            else torch.device("cpu")
        )
        self.obs_transforms = []
        self.start_epoch = 0
        self.step_id = 0

    def _initialize_policy(
        self,
        config: Config,
        load_from_ckpt: bool,
        observation_space: Space,
        action_space: Space,
    ) -> None:
        policy = baseline_registry.get_policy(self.config.MODEL.policy_name)
        self.policy = policy.from_config(
            config=config,
            observation_space=observation_space,
            action_space=action_space,
        )
        ''' initialize the waypoint predictor here '''
        from waypoint_prediction.TRM_net import BinaryDistPredictor_TRM
        self.waypoint_predictor = BinaryDistPredictor_TRM(device=self.device)
        self.waypoint_predictor.load_state_dict(
            torch.load(
                './waypoint_prediction/checkpoints/check_val_best_avg_wayscore',
                map_location = torch.device('cpu'),
            )['predictor']['state_dict']
        )
        for param in self.waypoint_predictor.parameters():
            param.requires_grad = False

  
        self.policy.to(self.device)
        self.waypoint_predictor.to(self.device)
        self.num_recurrent_layers = self.policy.net.num_recurrent_layers

        logger.info("Finished setting up waypoint_predictor.")

    def load_checkpoint(self, checkpoint_path, *args, **kwargs) -> Dict:
        return torch.load(checkpoint_path, *args, **kwargs)

    @staticmethod
    def _pause_envs(
        envs_to_pause,
        envs,
        not_done_masks,
        prev_actions,
        batch,
        rgb_frames=None,
    ):
        if len(envs_to_pause) > 0:
            state_index = list(range(envs.num_envs))
            for idx in reversed(envs_to_pause):
                state_index.pop(idx)
                envs.pause_at(idx)
                
            not_done_masks = not_done_masks[state_index]
            prev_actions = prev_actions[state_index]

            for k, v in batch.items():
                batch[k] = v[state_index]

            if rgb_frames is not None:
                rgb_frames = [rgb_frames[i] for i in state_index]

        return (
            envs,
            not_done_masks,
            prev_actions,
            batch,
            rgb_frames,
        )
        
    def generate_input(self, observations):
        instruction = observations['instruction']['text']
        image_dict = {} 
        rgb_image_dict = {}
        depth_image_dict = {}
        rgb_index = 0
        depth_index = 0
        for key in observations.keys():
            image_path = "./image_show/"
            if 'rgb' in key:
                image_path += f"{key}.jpg"
                image = Image.fromarray(observations[key], mode="RGB")
                dir_name = os.path.dirname(image_path)
                if not os.path.exists(dir_name):
                    os.makedirs(dir_name)
                image.save(image_path, format="JPEG")
                rgb_image_dict[str(rgb_index)] = Image.open(image_path)
                rgb_index += 1
            if 'depth' in key:
                image_path += f"{key}.jpg"
                if observations[key].ndim == 3 and observations[key].shape[-1] == 1:
                    depth_map = observations[key].squeeze(-1)
                depth_img = (255 * (depth_map - np.min(depth_map)) / (np.max(depth_map) - np.min(depth_map))).astype(np.uint8)
                image = Image.fromarray(depth_img)
                dir_name = os.path.dirname(image_path)
                if not os.path.exists(dir_name):
                    os.makedirs(dir_name)
                image.save(image_path)
                depth_image_dict[str(depth_index)] = Image.open(image_path)
                depth_index += 1
        for index in rgb_image_dict:
            image_dict[index] = {
                'rgb': rgb_image_dict[index],
                'depth': depth_image_dict[index]
            }
            
        return instruction, image_dict
    
    def construct_image_dicts(self, batch_distance, batch_angles, image_dict):
        waypoint_distances = {}
        waypoint_radius = {}
        waypoint_images = {}
        angles = batch_angles[-1]
        for angle_idx in range(len(angles)):
            angle = angles[angle_idx]
            angle_deg = np.rad2deg(angle)
            if 0 < angle_deg <= 30:
                waypoint_images['1'] = image_dict['1']
                waypoint_distances['1'] = batch_distance[angle_idx]
                waypoint_radius['1'] = angles[angle_idx]
            elif 30 < angle_deg <= 60:
                waypoint_images['2'] = image_dict['2']
                waypoint_distances['2'] = batch_distance[angle_idx]
                waypoint_radius['2'] = angles[angle_idx]
            elif 60 < angle_deg <= 90:
                waypoint_images['3'] = image_dict['3']
                waypoint_distances['3'] = batch_distance[angle_idx]
                waypoint_radius['3'] = angles[angle_idx]
            elif 90 < angle_deg <= 120:
                waypoint_images['4'] = image_dict['4']
                waypoint_distances['4'] = batch_distance[angle_idx]
                waypoint_radius['4'] = angles[angle_idx]
            elif 120 < angle_deg <= 150:
                waypoint_images['5'] = image_dict['5']
                waypoint_distances['5'] = batch_distance[angle_idx]
                waypoint_radius['5'] = angles[angle_idx]
            elif 150 < angle_deg <= 180:
                waypoint_images['6'] = image_dict['6']
                waypoint_distances['6'] = batch_distance[angle_idx]
                waypoint_radius['6'] = angles[angle_idx]
            elif 180 < angle_deg <= 210:
                waypoint_images['7'] = image_dict['7']
                waypoint_distances['7'] = batch_distance[angle_idx]
                waypoint_radius['7'] = angles[angle_idx]
            elif 210 < angle_deg <= 240:
                waypoint_images['8'] = image_dict['8']
                waypoint_distances['8'] = batch_distance[angle_idx]
                waypoint_radius['8'] = angles[angle_idx]
            elif 240 < angle_deg <= 270:
                waypoint_images['9'] = image_dict['9']
                waypoint_distances['9'] = batch_distance[angle_idx]
                waypoint_radius['9'] = angles[angle_idx]
            elif 270 < angle_deg <= 300:
                waypoint_images['10'] = image_dict['10']
                waypoint_distances['10'] = batch_distance[angle_idx]
                waypoint_radius['10'] = angles[angle_idx]
            elif 300 < angle_deg <= 330:
                waypoint_images['11'] = image_dict['11']
                waypoint_distances['11'] = batch_distance[angle_idx]
                waypoint_radius['11'] = angles[angle_idx]
            else:
                waypoint_images['0'] = image_dict['0']  
                waypoint_distances['0'] = batch_distance[angle_idx]
                waypoint_radius['0'] = angles[angle_idx]
                
        return waypoint_images, waypoint_radius, waypoint_distances
    

    def _eval_llm(
        self,
    ) -> None:
        r"""Evaluation.

        Args:
            writer: tensorboard writer object
            checkpoint_index: index of the current checkpoint

        Returns:
            None
        """
        config = self.config.clone()


        config.defrost()
        config.TASK_CONFIG.ENVIRONMENT.ITERATOR_OPTIONS.SHUFFLE = False
        config.TASK_CONFIG.ENVIRONMENT.ITERATOR_OPTIONS.MAX_SCENE_REPEAT_STEPS = (
            -1
        )
        if len(config.VIDEO_OPTION) > 0:
            config.defrost()
            config.TASK_CONFIG.TASK.MEASUREMENTS.append("TOP_DOWN_MAP_VLNCE")
            config.TASK_CONFIG.TASK.MEASUREMENTS.append("COLLISIONS")
        config.freeze()

        os.makedirs(config.RESULTS_DIR, exist_ok=True)
        if config.EVAL.SAVE_RESULTS:
            fname = os.path.join(
                config.RESULTS_DIR,
                f"stats_ckpt_{config.TASK_CONFIG.DATASET.SPLIT}.json",
            )
            if os.path.exists(fname):
                print(f"skipping -- evaluation exists. File path: {fname}")
                user_input = input("Do you want to overwrite the results? (yes/no): ").strip().lower()
                if user_input != "yes":
                    print("Skipping evaluation.")
                    return
                else:
                    print("Overwriting previous results...")
                

        # Smoke/debug episode whitelist (env-gated, no-op when unset): restrict the
        # eval to a comma/space-separated list of episode ids in OPENNAV_EPISODE_IDS,
        # intersected with the normal val trajectory set. Used to run a handful of
        # path-covering episodes for the clean-baseline smoke. MUST be unset for the
        # full run (leaves episodes_allowed == self.traj -> zero behavior difference).
        _episodes_allowed = self.traj
        _episode_whitelist = os.environ.get("OPENNAV_EPISODE_IDS", "").strip()
        if _episode_whitelist:
            _wl = {tok.strip() for tok in _episode_whitelist.replace(",", " ").split() if tok.strip()}
            _episodes_allowed = [e for e in self.traj if str(e) in _wl]
            print(
                f"[OPENNAV_EPISODE_IDS] restricting eval to {len(_episodes_allowed)} "
                f"episode(s): {_episodes_allowed}"
            )
        envs = construct_envs(
            config, get_env_class(config.ENV_NAME),
            auto_reset_done=False,
            episodes_allowed=_episodes_allowed
        )

        #envs.number_of_episodes = [1] # set the number of episodes
        dataset_length = sum(envs.number_of_episodes) 
        print('local rank:', self.local_rank, '|', 'dataset length:', dataset_length)

        obs_transforms = get_active_obs_transforms(config) 
        observation_space = apply_obs_transforms_obs_space(
            envs.observation_spaces[0], obs_transforms
        )
        self._initialize_policy(
            config,
            load_from_ckpt=False,
            observation_space=observation_space,
            action_space=envs.action_spaces[0],
        )
        self.policy.eval() 
        self.waypoint_predictor.eval()
        observations = envs.reset()
        
        instruction, images_list = self.generate_input(observations[-1])
        observations = extract_instruction_tokens(
            observations, self.config.TASK_CONFIG.TASK.INSTRUCTION_SENSOR_UUID
        ) 
        batch = batch_obs(observations, self.device) 
        batch = apply_obs_transforms_batch(batch, obs_transforms) 

        not_done_masks = torch.zeros(
            envs.num_envs, 1, dtype=torch.uint8, device=self.device
        ) 

        stats_episodes = {}
        rgb_frames = [[] for _ in range(envs.num_envs)]
        if len(config.VIDEO_OPTION) > 0:
            os.makedirs(config.VIDEO_DIR, exist_ok=True)

        if config.EVAL.EPISODE_COUNT == -1:
            episodes_to_eval = sum(envs.number_of_episodes)
        else:
            episodes_to_eval = min(
                config.EVAL.EPISODE_COUNT, sum(envs.number_of_episodes)
            )

        pbar = tqdm.tqdm(total=episodes_to_eval) if config.use_pbar else None
        log_str = (
            " [Episodes evaluated: {evaluated}/{total}]"
            " [Time elapsed (s): {time}]"
        )
        start_time = time.time()

        # set up the navigation record logger
        run_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_date = os.environ.get("OPENNAV_RUN_DATE") or run_stamp[:8]
        episode_group = os.environ.get("OPENNAV_EPISODE_GROUP") or _episode_group_name(
            config.EVAL.EPISODE_COUNT
        )
        nav_record_dir = os.path.join(
            "logs", "navigation_records", episode_group, run_date
        )
        os.makedirs(nav_record_dir, exist_ok=True)
        exp_name = os.path.splitext(os.path.basename(config.LOG_FILE))[0]
        nav_record_prefix = f"{exp_name}_navigation_{run_stamp}"
        log_file = os.path.join(nav_record_dir, f"{nav_record_prefix}.log")
        nav_jsonl_file = os.path.join(nav_record_dir, f"{nav_record_prefix}.jsonl")
        import logging
        logging.basicConfig(
            format='%(asctime)s - %(filename)s/%(funcName)s[line:%(lineno)d] - %(levelname)s: %(message)s',
            datefmt="%Y-%m-%d %H:%M:%S",
            level=os.environ.get("LOGLEVEL", "INFO").upper(),
            stream=sys.stdout,
            filemode="a"
        )
        nav_logger = logging.getLogger("vln_logger")
        nav_logger.setLevel(os.environ.get("LOGLEVEL", "INFO").upper())
        nav_logger.propagate = False
        for handler in list(nav_logger.handlers):
            if getattr(handler, "_opennav_navigation_record", False):
                nav_logger.removeHandler(handler)
                handler.close()
        nav_file_handler = logging.FileHandler(filename=log_file, encoding="utf-8")
        nav_file_handler._opennav_navigation_record = True
        nav_file_handler.setFormatter(
            logging.Formatter(
                fmt='%(asctime)s - %(filename)s/%(funcName)s[line:%(lineno)d] - %(levelname)s: %(message)s',
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        )
        nav_logger.addHandler(nav_file_handler)
        nav_logger.info(f"Navigation text log: {log_file}")
        nav_logger.info(f"Navigation JSONL record: {nav_jsonl_file}")

        validate_a1_harness_config(config)
        harness_enabled = harness_logging_enabled(config)
        harness_logger = None
        geometry_query = None
        grounder_diagnostic = None
        visual_evidence = None
        visual_fallback_ranker = None
        visual_target_verifier = None
        visual_target_verifier_decision_effect = False
        visual_target_verifier_reject_on_uncertain = True
        visual_evidence_memory = None
        multimodal_selector_context = None
        multimodal_selector_context_decision_effect = False
        memory_diagnostic = None
        context_builder = None
        u_series_active = False
        u_decision_unit = "none"
        phase_evidence_tracker = None
        phase_aware_scaffolder = None
        phase_aware_scaffolder_decision_effect = False
        stop_evidence_verifier = None
        stop_evidence_verifier_decision_effect = False
        failure_diagnostic = None
        recovery_policy = None
        recovery_policy_decision_effect = False
        max_recovery_per_episode = 2
        oracle_metrics_enabled = False
        completion_max_tokens = 0
        navigator_max_tokens = 0
        thought_fusion_max_tokens = 0
        decision_max_tokens = 0
        short_action_step_limit = 10
        long_action_step_limit = 12
        short_action_count_threshold = 6
        block_weak_final_target_completion_stop = True
        min_steps_for_weak_final_target_completion_stop = 8
        active_harness_episode_id = None
        active_navigation_episode_id = None
        stop_current_view_candidate_id = STOP_CURRENT_VIEW_CANDIDATE_ID
        arrival_gate = None
        split = config.TASK_CONFIG.DATASET.SPLIT
        if harness_enabled:
            run_id = "{}_seed{}_r{}_w{}".format(
                "{}_{}".format(exp_name, split),
                config.TASK_CONFIG.SEED,
                self.local_rank,
                self.world_size,
            )
            run_id = "{}_{}".format(run_id, run_stamp)
            harness_logger = MetricsLogger(
                trace_dir=get_trace_dir(config),
                run_id=run_id,
                rank=self.local_rank,
                fail_open=fail_open_enabled(config),
                logger=nav_logger,
            )
            if module_enabled(config, "GEOMETRY_QUERY"):
                geometry_query = GeometryQueryLogger()
            if module_enabled(config, "GROUNDER_DIAGNOSTIC"):
                grounder_diagnostic = GrounderDiagnostic()
            if module_enabled(config, "VISUAL_EVIDENCE"):
                visual_evidence_config = config.OPENNAV_HARNESS.VISUAL_EVIDENCE
                visual_evidence = VisualEvidenceLogger(
                    base_url=visual_evidence_config.BASE_URL,
                    model=visual_evidence_config.MODEL,
                    max_candidates=visual_evidence_config.MAX_CANDIDATES,
                    max_image_edge=visual_evidence_config.MAX_IMAGE_EDGE,
                    max_tokens=visual_evidence_config.MAX_TOKENS,
                    timeout_seconds=visual_evidence_config.TIMEOUT_SECONDS,
                    image_jpeg_quality=getattr(
                        visual_evidence_config, "IMAGE_JPEG_QUALITY", 80
                    ),
                    metadata_observation_chars=getattr(
                        visual_evidence_config,
                        "METADATA_OBSERVATION_CHARS",
                        700,
                    ),
                    compact_json=getattr(
                        visual_evidence_config, "COMPACT_JSON", False
                    ),
                )
                visual_fallback_ranker = VisualEvidenceFallbackRanker()
            if module_enabled(config, "VISUAL_TARGET_VERIFIER"):
                verifier_config = config.OPENNAV_HARNESS.VISUAL_TARGET_VERIFIER
                visual_target_verifier = VisualTargetVerifier(
                    confidence_threshold=verifier_config.CONFIDENCE_THRESHOLD,
                    require_arrival_evidence=(
                        verifier_config.REQUIRE_ARRIVAL_EVIDENCE
                    ),
                    reject_on_missing_final_landmarks=(
                        verifier_config.REJECT_ON_MISSING_FINAL_LANDMARKS
                    ),
                    min_steps_before_allow=getattr(
                        verifier_config, "MIN_STEPS_BEFORE_ALLOW", 0
                    ),
                    require_full_coverage_for_allow=getattr(
                        verifier_config,
                        "REQUIRE_FULL_COVERAGE_FOR_ALLOW",
                        False,
                    ),
                    block_generic_final_terms_for_allow=getattr(
                        verifier_config,
                        "BLOCK_GENERIC_FINAL_TERMS_FOR_ALLOW",
                        False,
                    ),
                    require_completion_for_selector_stop=getattr(
                        verifier_config,
                        "REQUIRE_COMPLETION_FOR_SELECTOR_STOP",
                        True,
                    ),
                    require_current_view_text_corroboration_for_allow=getattr(
                        verifier_config,
                        "REQUIRE_CURRENT_VIEW_TEXT_CORROBORATION_FOR_ALLOW",
                        False,
                    ),
                )
                visual_target_verifier_decision_effect = (
                    decision_effect_enabled(config)
                    and not module_log_only(config, "VISUAL_TARGET_VERIFIER")
                )
                visual_target_verifier_reject_on_uncertain = bool(
                    getattr(verifier_config, "REJECT_ON_UNCERTAIN", True)
                )
            if module_enabled(config, "VISUAL_EVIDENCE_MEMORY"):
                visual_memory_config = config.OPENNAV_HARNESS.VISUAL_EVIDENCE_MEMORY
                visual_evidence_memory = VisualEvidenceMemory(
                    max_history=visual_memory_config.MAX_HISTORY,
                    max_notes_chars=visual_memory_config.MAX_NOTES_CHARS,
                )
            if module_enabled(config, "MULTIMODAL_SELECTOR_CONTEXT"):
                vsc_config = config.OPENNAV_HARNESS.MULTIMODAL_SELECTOR_CONTEXT
                multimodal_selector_context = MultimodalSelectorContext(
                    max_summary_chars=vsc_config.MAX_SUMMARY_CHARS,
                    include_memory_suffix=getattr(
                        vsc_config,
                        "INCLUDE_MEMORY_SUFFIX",
                        False,
                    ),
                    decision_mode=getattr(
                        vsc_config,
                        "DECISION_MODE",
                        "phase_gated_u1",
                    ),
                    suppress_target_arrival_for_selector=getattr(
                        vsc_config,
                        "SUPPRESS_TARGET_ARRIVAL_FOR_SELECTOR",
                        True,
                    ),
                    min_confidence_for_target_hint=getattr(
                        vsc_config,
                        "MIN_CONFIDENCE_FOR_TARGET_HINT",
                        0.9,
                    ),
                )
                multimodal_selector_context_decision_effect = (
                    decision_effect_enabled(config)
                    and not module_log_only(config, "MULTIMODAL_SELECTOR_CONTEXT")
                )
            if module_enabled(config, "MEMORY_DIAGNOSTIC"):
                memory_diagnostic = VisualGraphMemoryDiagnostic()
            if module_enabled(config, "CONTEXT_BUILDER"):
                context_builder = ContextBuilder()
            if u_series_enabled(config):
                u_series_active = True
                u_decision_unit = u_decision_effect_unit(config)
                u_config = config.OPENNAV_HARNESS.U_SERIES
                if u_module_enabled(config, "PHASE_EVIDENCE"):
                    phase_config = u_config.PHASE_EVIDENCE
                    phase_evidence_tracker = PhaseEvidenceTracker(
                        late_step_threshold=getattr(
                            phase_config,
                            "LATE_STEP_THRESHOLD",
                            4,
                        ),
                        unknown_confidence_threshold=getattr(
                            phase_config,
                            "UNKNOWN_CONFIDENCE_THRESHOLD",
                            0.4,
                        ),
                    )
                    selector_context_config = getattr(
                        config.OPENNAV_HARNESS,
                        "MULTIMODAL_SELECTOR_CONTEXT",
                        None,
                    )
                    phase_aware_scaffolder = PhaseAwareEvidenceScaffolder(
                        max_context_chars=getattr(
                            selector_context_config,
                            "MAX_SUMMARY_CHARS",
                            220,
                        ),
                        apply_phases=getattr(
                            selector_context_config,
                            "APPLY_PHASES",
                            ["search", "approach"],
                        ),
                    )
                    phase_aware_scaffolder_decision_effect = (
                        decision_effect_enabled(config)
                        and u_decision_unit in {"U1", "combined"}
                        and not u_module_log_only(config, "PHASE_EVIDENCE")
                    )
                if u_module_enabled(config, "STOP_EVIDENCE_VERIFIER"):
                    stop_config = u_config.STOP_EVIDENCE_VERIFIER
                    stop_evidence_verifier = StopEvidenceVerifier(
                        enable_rescue=getattr(
                            stop_config,
                            "ENABLE_RESCUE",
                            False,
                        ),
                        enable_relation_check=getattr(
                            stop_config,
                            "ENABLE_RELATION_CHECK",
                            False,
                        ),
                        enable_weak_target_adjustment=getattr(
                            stop_config,
                            "ENABLE_WEAK_TARGET_ADJUSTMENT",
                            False,
                        ),
                        rescue_confidence_threshold=getattr(
                            stop_config,
                            "RESCUE_CONFIDENCE_THRESHOLD",
                            0.99,
                        ),
                        rescue_min_step=getattr(
                            stop_config,
                            "RESCUE_MIN_STEP",
                            8,
                        ),
                        rescue_max_non_positive_gains=getattr(
                            stop_config,
                            "RESCUE_MAX_NON_POSITIVE_GAINS",
                            1,
                        ),
                        rescue_require_positive_recent_gain=getattr(
                            stop_config,
                            "RESCUE_REQUIRE_POSITIVE_RECENT_GAIN",
                            True,
                        ),
                        rescue_allow_phase_verify=getattr(
                            stop_config,
                            "RESCUE_ALLOW_PHASE_VERIFY",
                            False,
                        ),
                        trajectory_bypass_dist=getattr(
                            stop_config,
                            "TRAJECTORY_BYPASS_DIST",
                            0.0,
                        ),
                        e3_arrival_override_dist=getattr(
                            stop_config,
                            "E3_ARRIVAL_OVERRIDE_DIST",
                            0.0,
                        ),
                        e3_abstain_dist=getattr(
                            stop_config,
                            "E3_ABSTAIN_DIST",
                            0.0,
                        ),
                    )
                    stop_evidence_verifier_decision_effect = (
                        decision_effect_enabled(config)
                        and u_decision_unit in {"U2", "combined"}
                        and not u_module_log_only(config, "STOP_EVIDENCE_VERIFIER")
                    )
                if u_module_enabled(config, "FAILURE_RECOVERY"):
                    recovery_config = u_config.FAILURE_RECOVERY
                    failure_diagnostic = FailureDiagnostic(
                        negative_gain_window=getattr(
                            recovery_config,
                            "NEGATIVE_GAIN_WINDOW",
                            2,
                        )
                    )
                    max_recovery_per_episode = int(
                        getattr(
                            recovery_config,
                            "MAX_RECOVERY_PER_EPISODE",
                            max_recovery_per_episode,
                        )
                    )
                    recovery_policy = RecoveryPolicy(
                        max_recovery_per_episode=max_recovery_per_episode
                    )
                    recovery_policy_decision_effect = (
                        decision_effect_enabled(config)
                        and u_decision_unit in {"U3", "combined"}
                        and not u_module_log_only(config, "FAILURE_RECOVERY")
                        and bool(
                            getattr(
                                recovery_config,
                                "ENABLE_RESELECT",
                                False,
                            )
                        )
                    )
            if arrival_gate_enabled(config):
                _ag_cfg = arrival_gate_config(config)
                arrival_gate = ArrivalGate(
                    dist_threshold=_ag_cfg.get("dist_threshold", 4.0),
                    allowed_phases=_ag_cfg.get("allowed_phases"),
                    min_trigger_step=_ag_cfg.get("min_trigger_step", 1),
                )
            proactive_stop_enabled = proactive_stop_gate_enabled(config)
            proactive_stop_dist = 0.0
            proactive_stop_commit_dist = 0.0
            if proactive_stop_enabled:
                _psg_cfg = proactive_stop_gate_config(config)
                proactive_stop_dist = _psg_cfg.get("dist_threshold", 3.5)
                # E3-for-M3: inner commit zone. V2 re-evaluation fires for any
                # dist < dist_threshold, but STOP is only committed when dist <
                # commit_dist_threshold (inside success radius). When 0, falls back
                # to old behavior (commit anywhere inside dist_threshold).
                proactive_stop_commit_dist = _psg_cfg.get("commit_dist_threshold", 0.0)
                if proactive_stop_commit_dist <= 0:
                    proactive_stop_commit_dist = proactive_stop_dist
            # Distance guard for the U2 visual-STOP rescue/override paths. Without it,
            # a confident navigator STOP + a generic "doorway"-style visual match can
            # commit a stop far from the goal (observed: ep244 stopped at 6.2m / step 3
            # via "rescue override granted despite gain/phase conditions"). When set,
            # rescue/override STOP is blocked while latest_goal_dist exceeds this value.
            # 0 disables the guard (original behavior). See docs/current_task.md E5.
            rescue_max_goal_dist = 0.0
            _u_series_cfg = getattr(config.OPENNAV_HARNESS, "U_SERIES", None)
            if _u_series_cfg is not None:
                _sev_cfg = getattr(_u_series_cfg, "STOP_EVIDENCE_VERIFIER", None)
                if _sev_cfg is not None:
                    rescue_max_goal_dist = getattr(
                        _sev_cfg, "RESCUE_MAX_GOAL_DIST", 0.0
                    )
            # P0 switch: inject WaypointBert sensor distance into navigator observations.
            # When False, observe_environment receives distance_dict=None and injection
            # is skipped (clean baseline for P0 ablation). Default True.
            geometry_injection_enabled = bool(
                getattr(config.OPENNAV_HARNESS, "GEOMETRY_INJECTION", True)
            )
            oracle_metrics_enabled = module_enabled(config, "ORACLE_METRICS")
            llm_runtime_config = getattr(config.OPENNAV_HARNESS, "LLM_RUNTIME", None)
            if llm_runtime_config is not None:
                completion_max_tokens = getattr(
                    llm_runtime_config, "COMPLETION_MAX_TOKENS", 0
                )
                navigator_max_tokens = getattr(
                    llm_runtime_config, "NAVIGATOR_MAX_TOKENS", 0
                )
                thought_fusion_max_tokens = getattr(
                    llm_runtime_config, "THOUGHT_FUSION_MAX_TOKENS", 0
                )
                decision_max_tokens = getattr(
                    llm_runtime_config, "DECISION_MAX_TOKENS", 0
                )
            navigation_runtime_config = getattr(
                config.OPENNAV_HARNESS, "NAVIGATION_RUNTIME", None
            )
            if navigation_runtime_config is not None:
                short_action_step_limit = int(
                    getattr(
                        navigation_runtime_config,
                        "SHORT_ACTION_STEP_LIMIT",
                        short_action_step_limit,
                    )
                )
                long_action_step_limit = int(
                    getattr(
                        navigation_runtime_config,
                        "LONG_ACTION_STEP_LIMIT",
                        long_action_step_limit,
                    )
                )
                short_action_count_threshold = int(
                    getattr(
                        navigation_runtime_config,
                        "SHORT_ACTION_COUNT_THRESHOLD",
                        short_action_count_threshold,
                    )
                )
                block_weak_final_target_completion_stop = bool(
                    getattr(
                        navigation_runtime_config,
                        "BLOCK_WEAK_FINAL_TARGET_COMPLETION_STOP",
                        block_weak_final_target_completion_stop,
                    )
                )
                min_steps_for_weak_final_target_completion_stop = int(
                    getattr(
                        navigation_runtime_config,
                        "MIN_STEPS_FOR_WEAK_FINAL_TARGET_COMPLETION_STOP",
                        min_steps_for_weak_final_target_completion_stop,
                    )
                )

        def run_harness_tool(tool_name, step_id, fallback, func, *args, **kwargs):
            if not harness_enabled or func is None:
                return fallback
            start_time = time.perf_counter()
            try:
                result = func(*args, **kwargs)
                return result
            except Exception as exc:
                if harness_logger is not None:
                    harness_logger.log_tool_failure(tool_name, step_id, exc)
                if isinstance(fallback, dict):
                    failure_payload = dict(fallback)
                    failure_payload.update(
                        {
                            "skipped": True,
                            "reason": "tool_failure",
                            "tool_name": tool_name,
                            "error_type": type(exc).__name__,
                            "error": repr(exc),
                        }
                    )
                    return failure_payload
                return fallback
            finally:
                record_runtime_latency(
                    tool_name,
                    active_navigation_episode_id,
                    step_id,
                    time.perf_counter() - start_time,
                    category="harness_tool",
                )

        def to_jsonable(value):
            if torch.is_tensor(value):
                return value.detach().cpu().tolist()
            if isinstance(value, np.ndarray):
                return value.tolist()
            if isinstance(value, np.generic):
                return value.item()
            if isinstance(value, dict):
                return {str(key): to_jsonable(item) for key, item in value.items()}
            if isinstance(value, (list, tuple)):
                return [to_jsonable(item) for item in value]
            return value

        def write_navigation_record(event, episode_id=None, step=None, **payload):
            record = {
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "event": event,
                "episode_id": str(episode_id) if episode_id is not None else None,
                "step": step,
                "payload": to_jsonable(payload),
            }
            try:
                with open(nav_jsonl_file, "a", encoding="utf-8") as record_file:
                    record_file.write(json.dumps(record, ensure_ascii=False) + "\n")
            except Exception as exc:
                nav_logger.info(f"Navigation JSONL record failed: {exc}")

        def record_runtime_latency(
            operation,
            episode_id,
            step,
            elapsed_seconds,
            **payload,
        ):
            if not harness_enabled:
                return
            latency_payload = {
                "operation": operation,
                "elapsed_seconds": round(float(elapsed_seconds), 4),
            }
            latency_payload.update(payload)
            write_navigation_record(
                "runtime_latency",
                episode_id=episode_id,
                step=step,
                **latency_payload,
            )
            if harness_logger is not None:
                try:
                    harness_logger.log_event(
                        "runtime_latency",
                        step,
                        latency_payload,
                    )
                except Exception as exc:
                    nav_logger.info(f"Runtime latency trace log failed: {exc}")

        def positive_token_cap(value):
            try:
                value = int(value)
            except (TypeError, ValueError):
                return None
            return value if value > 0 else None

        def observation_order_fallback(observe_dict):
            if not isinstance(observe_dict, dict) or not observe_dict:
                return {
                    "fallback_strategy": "observation_order",
                    "fallback_reason": "no_available_candidates",
                    "selected_candidate": None,
                    "available_candidates": [],
                    "ranked_candidates": [],
                    "recovery_rank_trusted": False,
                }
            available_candidates = [str(key) for key in observe_dict.keys()]
            selected_candidate = available_candidates[0]
            return {
                "fallback_strategy": "observation_order",
                "fallback_reason": "ranker_unavailable_or_empty",
                "selected_candidate": selected_candidate,
                "available_candidates": available_candidates,
                "ranked_candidates": [
                    {
                        "candidate_id": candidate,
                        "score": [0, -idx],
                        "source": "observation_order",
                    }
                    for idx, candidate in enumerate(available_candidates)
                ],
                "recovery_rank_trusted": False,
            }

        def rank_movement_fallback(
            tool_name,
            step_id,
            observe_dict,
            visual_evidence_results,
            instruction,
            actions,
            landmarks,
            source_stage,
            reason,
        ):
            fallback_results = run_harness_tool(
                tool_name,
                step_id,
                {},
                visual_fallback_ranker.rank
                if visual_fallback_ranker is not None
                else None,
                observe_dict,
                visual_evidence_results,
                instruction,
                actions,
                landmarks,
                source_stage,
                reason,
            )
            if not isinstance(fallback_results, dict):
                fallback_results = {}
            selected_candidate = fallback_results.get("selected_candidate")
            if selected_candidate not in observe_dict:
                fallback_results = observation_order_fallback(observe_dict)
                selected_candidate = fallback_results.get("selected_candidate")
            if "fallback_strategy" not in fallback_results:
                fallback_results["fallback_strategy"] = "visual_evidence_ranked"
            fallback_results["fallback_reason"] = reason
            fallback_results["source_stage"] = source_stage
            fallback_results.setdefault(
                "available_candidates",
                [str(key) for key in observe_dict.keys()],
            )
            fallback_results["selected_candidate"] = selected_candidate
            fallback_results["recovery_rank_trusted"] = bool(
                fallback_results.get("fallback_strategy") == "visual_evidence_ranked"
                and fallback_results.get("ranked_candidates")
            )
            return fallback_results

        def log_fallback_event(event_name, episode_id, step, payload):
            write_navigation_record(
                event_name,
                episode_id=episode_id,
                step=step,
                **payload,
            )
            if harness_enabled and harness_logger is not None:
                harness_logger.log_event(event_name, step, payload)

        def log_u_event(event_name, episode_id, step, payload):
            if not u_series_active:
                return
            write_navigation_record(
                event_name,
                episode_id=episode_id,
                step=step,
                **(payload or {}),
            )
            if harness_enabled and harness_logger is not None:
                harness_logger.log_event(event_name, step, payload or {})

        unit_override_records = []

        def record_u_override(payload):
            if isinstance(payload, dict):
                unit_override_records.append(dict(payload))
            log_u_event(
                "unit_override",
                current_episode_id,
                current_step,
                payload,
            )

        def build_action_decision_audit(final_action, stop_requested):
            action_overrides = [
                record
                for record in unit_override_records
                if record.get("action_affecting", True)
            ]
            if action_overrides:
                original_action = action_overrides[0].get("original_action")
                proposed_action = action_overrides[-1].get("proposed_action")
                override_reason = " | ".join(
                    str(record.get("override_reason") or "")
                    for record in action_overrides
                    if record.get("override_reason")
                )
                expected_failure_addressed = " | ".join(
                    str(record.get("expected_failure_addressed") or "")
                    for record in action_overrides
                    if record.get("expected_failure_addressed")
                )
            else:
                original_action = final_action
                proposed_action = final_action
                override_reason = ""
                expected_failure_addressed = "none"
            return build_decision_audit(
                unit_enabled=u_decision_unit if u_series_active else "none",
                original_action=original_action,
                proposed_action=proposed_action,
                final_action=final_action,
                override_reason=override_reason,
                expected_failure_addressed=expected_failure_addressed or "none",
                source_stage="action_pre_step",
                extra={
                    "stop_flag": stop_requested,
                    "phase": latest_phase_evidence.get("phase")
                    if isinstance(latest_phase_evidence, dict)
                    else None,
                    "failure_type": latest_failure_signal.get("failure_type")
                    if isinstance(latest_failure_signal, dict)
                    else None,
                    "unit_override_count": len(unit_override_records),
                    "action_override_count": len(action_overrides),
                    "unit_overrides": unit_override_records,
                },
            )

        def apply_failure_recovery(
            trigger_type,
            source_stage,
            observe_dict,
            fallback_results,
            failure_signal,
            current_candidate,
            current_thought,
        ):
            nonlocal recovery_budget_remaining
            recovery_results = run_harness_tool(
                "failure_recovery",
                current_step,
                {},
                recovery_policy.propose if recovery_policy is not None else None,
                trigger_type,
                failure_signal,
                fallback_results,
                observe_dict,
                current_candidate,
                recovery_budget_remaining,
                recent_distance_gains,
            )
            if isinstance(recovery_results, dict):
                recovery_results = dict(recovery_results)
                would_apply = bool(recovery_results.get("applied"))
                recovery_results["would_apply"] = would_apply
                recovery_results[
                    "decision_effect_enabled"
                ] = recovery_policy_decision_effect
                recovery_results["applied_to_action"] = bool(
                    recovery_policy_decision_effect and would_apply
                )
                recovery_results["budget_after_proposed"] = recovery_results.get(
                    "budget_after"
                )
                if not recovery_policy_decision_effect:
                    recovery_results["applied"] = False
                    recovery_results["budget_after"] = recovery_budget_remaining
            if recovery_policy is not None:
                log_u_event(
                    "failure_recovery",
                    current_episode_id,
                    current_step,
                    recovery_results,
                )
            if (
                not recovery_policy_decision_effect
                or not isinstance(recovery_results, dict)
                or not recovery_results.get("applied")
            ):
                return current_candidate, current_thought, recovery_results
            recovered_candidate = recovery_results.get("selected_candidate")
            if recovered_candidate not in observe_dict:
                return current_candidate, current_thought, recovery_results
            recovery_budget_remaining = int(
                recovery_results.get(
                    "budget_after",
                    max(0, recovery_budget_remaining - 1),
                )
            )
            recovered_thought = observe_dict[recovered_candidate]
            recovery_override = build_decision_audit(
                unit_enabled=u_decision_unit,
                original_action=current_candidate,
                proposed_action=recovered_candidate,
                final_action=recovered_candidate,
                override_reason=recovery_results.get("reason", ""),
                expected_failure_addressed=(
                    recovery_results.get("failure_type", "unknown")
                ),
                source_stage=source_stage,
                extra={
                    "trigger_type": trigger_type,
                    "budget_before": recovery_results.get("budget_before"),
                    "budget_after": recovery_results.get("budget_after"),
                },
            )
            recovery_override["effect_type"] = "fallback_reselect"
            recovery_override["action_affecting"] = True
            record_u_override(recovery_override)
            return recovered_candidate, recovered_thought, recovery_results
        
        dataset_name = "R2R"
        if not os.path.exists(f"cache_files/{dataset_name}"):
            os.makedirs(f"cache_files/{dataset_name}")

        actions_cache_path = f"./cache_files/{dataset_name}/actions_cache.json"
        if os.path.exists(actions_cache_path): 
            with open(actions_cache_path, "r", encoding="utf-8") as file:
                actions_cache = json.load(file)
        else:
            actions_cache = {} 
        
        navigator = Open_Nav(self.device,config.LLM, config.API_KEY)
        navigator.block_weak_final_target_completion_stop = (
            block_weak_final_target_completion_stop
        )
        navigator.min_steps_for_weak_final_target_completion_stop = (
            min_steps_for_weak_final_target_completion_stop
        )
        current_step = 0
        nav_history = []
        error_number = 0
        step_error_count = 0
        last_errored_step = -1
        recent_distance_gains = []
        latest_goal_dist = None
        recovery_budget_remaining = max_recovery_per_episode
        latest_phase_evidence = {}
        latest_failure_signal = {}
        recent_ftv_window: List[bool] = []
        while envs.num_envs > 0 and len(stats_episodes) < episodes_to_eval:
            current_episodes = envs.current_episodes()
            positions = []; headings = []
            for ob_i in range(len(current_episodes)): 
                agent_state_i = envs.call_at(ob_i,
                        "get_agent_info", {})
                positions.append(agent_state_i['position'])
                headings.append(agent_state_i['heading'])
            current_episode_id = str(current_episodes[0].episode_id)
            if (
                harness_enabled
                and harness_logger is not None
                and active_harness_episode_id != current_episode_id
            ):
                active_harness_episode_id = current_episode_id
                harness_logger.start_episode(
                    current_episode_id,
                    split,
                    instruction,
                    metadata={
                        "exp_name": exp_name,
                        "llm": config.LLM,
                        "trace_dir": get_trace_dir(config),
                        "positions": positions,
                        "headings": headings,
                    },
                )
                if memory_diagnostic is not None:
                    run_harness_tool(
                        "memory_reset",
                        0,
                        None,
                        memory_diagnostic.reset_episode,
                    )
                if visual_evidence_memory is not None:
                    run_harness_tool(
                        "visual_evidence_memory_reset",
                        0,
                        None,
                        visual_evidence_memory.reset_episode,
                    )
            # ==========Navigator start==========
            nav_logger.info(f"==================== The current episode id is {current_episodes[0].episode_id} ====================")
            nav_logger.info("Instruction: "+instruction)
            actions, landmarks = "", ""
            if instruction not in actions_cache.keys():
                actions = navigator.get_actions(instruction)
                landmarks = navigator.get_landmarks(actions)
                actions_cache[instruction] = {"actions": actions, "landmarks": landmarks}
                with open(actions_cache_path, "w", encoding="utf-8") as f2:
                    json.dump(actions_cache, f2, indent=2)
            else:
                actions = actions_cache[instruction]["actions"]
                landmarks = actions_cache[instruction]["landmarks"]
            nav_logger.info("Actions: "+actions)
            nav_logger.info("Landmarks: " + landmarks)
            if active_navigation_episode_id != current_episode_id:
                active_navigation_episode_id = current_episode_id
                recent_distance_gains = []
                latest_goal_dist = None
                recovery_budget_remaining = max_recovery_per_episode
                latest_phase_evidence = {}
                latest_failure_signal = {}
                recent_ftv_window: List[bool] = []
                write_navigation_record(
                    "episode_start",
                    episode_id=current_episode_id,
                    step=0,
                    instruction=instruction,
                    actions=actions,
                    landmarks=landmarks,
                    positions=positions,
                    headings=headings,
                )
                if harness_enabled and harness_logger is not None:
                    harness_logger.log_event(
                        "episode_metadata",
                        0,
                        {
                            "episode_id": current_episode_id,
                            "actions": actions,
                            "landmarks": landmarks,
                            "positions": positions,
                            "headings": headings,
                        },
                    )
                log_u_event(
                    "u_series_episode_start",
                    current_episode_id,
                    0,
                    {
                        "enabled": u_series_active,
                        "decision_effect_unit": u_decision_unit,
                        "recovery_budget": recovery_budget_remaining,
                    },
                )
            
            action_count = len(navigator._split_actions(actions))
            if action_count <= short_action_count_threshold:
                step_length = short_action_step_limit
            else:
                step_length = long_action_step_limit

            stop_flag = False
            stop_reason = ""
            current_step += 1
            nav_logger.info(f"-------------------- Step {current_step} --------------------")
            write_navigation_record(
                "step_start",
                episode_id=current_episode_id,
                step=current_step,
                positions=positions,
                headings=headings,
            )
            if harness_enabled and harness_logger is not None:
                harness_logger.log_event(
                    "step_start",
                    current_step,
                    {
                        "episode_id": current_episode_id,
                        "instruction": instruction,
                        "positions": positions,
                        "headings": headings,
                    },
                )
            with torch.no_grad():
                # candidate waypoints prediction
                cand_rgb, cand_depth, \
                cand_direction, cand_mask, candidate_lengths, \
                batch_angles, batch_distances = self.policy.net( 
                    mode = "waypoint",
                    waypoint_predictor = self.waypoint_predictor,
                    observations = batch,
                    in_train = False,
                )
            
            images_dict, radius_dict, distance_dict = self.construct_image_dicts(batch_distances[-1], batch_angles, images_list)
            candidates = []
            geometry_results = []
            grounding_results = []
            visual_evidence_results = {}
            stop_current_view_evidence_results = None
            visual_evidence_memory_results = {}
            memory_results = {}
            if harness_enabled and harness_logger is not None:
                candidates = build_candidate_records(
                    radius_dict,
                    distance_dict,
                    images_dict,
                )
                harness_logger.log_event(
                    "waypoint_candidates",
                    current_step,
                    {
                        "candidates": candidates,
                        "candidate_count": len(candidates),
                        "candidate_lengths": candidate_lengths,
                        "candidate_mask": cand_mask,
                        "candidate_direction": cand_direction,
                    },
                )
            nav_logger.info("========== Get Observation ==========")
            observation, observe_dict = navigator.observe_environment(
                nav_logger,
                current_step,
                images_dict,
                distance_dict if geometry_injection_enabled else None,
            )
            selector_observation = observation
            selector_observe_dict = observe_dict
            multimodal_selector_context_results = {}
            phase_aware_context_results = {}
            phase_context_applied = False
            phase_context_should_apply = False
            unit_override_records = []
            write_navigation_record(
                "observation",
                episode_id=current_episode_id,
                step=current_step,
                observation=observation,
                observe_dict=observe_dict,
                radius_dict=radius_dict,
                distance_dict=distance_dict,
            )
            if harness_enabled and harness_logger is not None:
                harness_logger.log_event(
                    "observation",
                    current_step,
                    {
                        "observation": observation,
                        "observe_dict": observe_dict,
                    },
                )
                geometry_results = run_harness_tool(
                    "geometry_query",
                    current_step,
                    [],
                    geometry_query.run if geometry_query is not None else None,
                    candidates,
                    positions[0] if positions else None,
                    headings[0] if headings else None,
                    images_dict,
                )
                if geometry_query is not None:
                    harness_logger.log_event(
                        "geometry_query",
                        current_step,
                        {"results": geometry_results},
                    )
                grounding_results = run_harness_tool(
                    "grounder_diagnostic",
                    current_step,
                    [],
                    grounder_diagnostic.run
                    if grounder_diagnostic is not None
                    else None,
                    instruction,
                    actions,
                    landmarks,
                    observe_dict,
                    candidates,
                )
                if grounder_diagnostic is not None:
                    harness_logger.log_event(
                        "grounder_diagnostic",
                        current_step,
                        {"results": grounding_results},
                    )
                visual_evidence_results = run_harness_tool(
                    "visual_evidence",
                    current_step,
                    {},
                    visual_evidence.run if visual_evidence is not None else None,
                    instruction,
                    actions,
                    landmarks,
                    candidates,
                    images_dict,
                    observe_dict,
                )
                if visual_evidence is not None:
                    write_navigation_record(
                        "visual_evidence",
                        episode_id=current_episode_id,
                        step=current_step,
                        **visual_evidence_results,
                    )
                    harness_logger.log_event(
                        "visual_evidence",
                        current_step,
                        visual_evidence_results,
                    )
                    sampling_payload = {
                        "total_candidate_ids": visual_evidence_results.get(
                            "total_candidate_ids"
                        ),
                        "requested_candidate_ids": visual_evidence_results.get(
                            "requested_candidate_ids"
                        ),
                        "selection_reason_by_candidate": (
                            visual_evidence_results.get(
                                "selection_reason_by_candidate"
                            )
                        ),
                        "sampled_all": visual_evidence_results.get("sampled_all"),
                        "sampled_candidate_count": visual_evidence_results.get(
                            "sampled_candidate_count"
                        ),
                        "total_candidate_count": visual_evidence_results.get(
                            "total_candidate_count"
                        ),
                    }
                    write_navigation_record(
                        "visual_evidence_sampling",
                        episode_id=current_episode_id,
                        step=current_step,
                        **sampling_payload,
                    )
                    harness_logger.log_event(
                        "visual_evidence_sampling",
                        current_step,
                        sampling_payload,
                    )
                visual_evidence_memory_results = run_harness_tool(
                    "visual_evidence_memory",
                    current_step,
                    {},
                    visual_evidence_memory.update
                    if visual_evidence_memory is not None
                    else None,
                    current_step,
                    positions[0] if positions else None,
                    headings[0] if headings else None,
                    visual_evidence_results,
                    landmarks,
                )
                if visual_evidence_memory is not None:
                    write_navigation_record(
                        "visual_evidence_memory",
                        episode_id=current_episode_id,
                        step=current_step,
                        **visual_evidence_memory_results,
                    )
                    harness_logger.log_event(
                        "visual_evidence_memory",
                        current_step,
                        visual_evidence_memory_results,
                    )
                multimodal_selector_context_results = run_harness_tool(
                    "multimodal_selector_context",
                    current_step,
                    {
                        "augmented_observe_dict": observe_dict,
                        "augmented_observation": observation,
                    },
                    multimodal_selector_context.build
                    if multimodal_selector_context is not None
                    else None,
                    observe_dict,
                    visual_evidence_results,
                    visual_evidence_memory_results,
                )
                if multimodal_selector_context is not None:
                    selector_context_applied = (
                        multimodal_selector_context_decision_effect
                        and isinstance(multimodal_selector_context_results, dict)
                        and not multimodal_selector_context_results.get("skipped")
                        and isinstance(
                            multimodal_selector_context_results.get(
                                "augmented_observe_dict"
                            ),
                            dict,
                        )
                    )
                    multimodal_selector_context_results[
                        "decision_effect_enabled"
                    ] = multimodal_selector_context_decision_effect
                    multimodal_selector_context_results[
                        "applied"
                    ] = selector_context_applied
                    if selector_context_applied:
                        selector_observe_dict = multimodal_selector_context_results[
                            "augmented_observe_dict"
                        ]
                        selector_observation = (
                            multimodal_selector_context_results.get(
                                "augmented_observation"
                            )
                            or list(selector_observe_dict.values())
                        )
                    write_navigation_record(
                        "multimodal_selector_context",
                        episode_id=current_episode_id,
                        step=current_step,
                        **multimodal_selector_context_results,
                    )
                    harness_logger.log_event(
                        "multimodal_selector_context",
                        current_step,
                        multimodal_selector_context_results,
                    )
                memory_results = run_harness_tool(
                    "memory_diagnostic",
                    current_step,
                    {},
                    memory_diagnostic.update
                    if memory_diagnostic is not None
                    else None,
                    current_step,
                    positions[0] if positions else None,
                    headings[0] if headings else None,
                    candidates,
                )
                if memory_diagnostic is not None:
                    harness_logger.log_event(
                        "memory_diagnostic",
                        current_step,
                        memory_results,
                    )
                diagnostic_context = run_harness_tool(
                    "context_builder",
                    current_step,
                    {},
                    context_builder.build_diagnostic
                    if context_builder is not None
                    else None,
                    candidates,
                    geometry_results,
                    grounding_results,
                    memory_results,
                    None,
                )
                if context_builder is not None:
                    harness_logger.log_event(
                        "diagnostic_context",
                        current_step,
                        diagnostic_context,
                    )
            
            nav_logger.info("========== Review History ==========")
            history_traj = navigator.review_history(nav_logger, nav_history) if len(nav_history) > 0 else "Step 0 start position. "
            write_navigation_record(
                "history_review",
                episode_id=current_episode_id,
                step=current_step,
                history=history_traj,
            )

            if not stop_flag:
                nav_logger.info("========== Estimate Completion Progress ==========")
                completion_start_time = time.perf_counter()
                estimation = navigator.estimate_completion(
                    nav_logger,
                    actions,
                    landmarks,
                    history_traj,
                    max_tokens=positive_token_cap(completion_max_tokens),
                )
                record_runtime_latency(
                    "completion_estimation",
                    current_episode_id,
                    current_step,
                    time.perf_counter() - completion_start_time,
                    category="text_llm",
                    max_tokens=positive_token_cap(completion_max_tokens),
                )
                write_navigation_record(
                    "completion_estimation",
                    episode_id=current_episode_id,
                    step=current_step,
                    estimation=estimation,
                )
                latest_phase_evidence = run_harness_tool(
                    "phase_evidence",
                    current_step,
                    {},
                    phase_evidence_tracker.build
                    if phase_evidence_tracker is not None
                    else None,
                    instruction,
                    actions,
                    landmarks,
                    estimation,
                    history_traj,
                    current_step,
                    recent_distance_gains,
                    {},
                    latest_failure_signal,
                )
                if phase_evidence_tracker is not None:
                    log_u_event(
                        "phase_evidence",
                        current_episode_id,
                        current_step,
                        latest_phase_evidence,
                    )
                if arrival_gate is not None:
                    _gate_result = arrival_gate.check(
                        latest_goal_dist=latest_goal_dist,
                        current_phase=latest_phase_evidence.get("phase")
                        if isinstance(latest_phase_evidence, dict)
                        else None,
                        current_step=current_step,
                    )
                    write_navigation_record(
                        "arrival_gate",
                        episode_id=current_episode_id,
                        step=current_step,
                        **_gate_result,
                    )
                    if harness_enabled and harness_logger is not None:
                        harness_logger.log_event(
                            "arrival_gate",
                            current_step,
                            _gate_result,
                        )
                phase_aware_context_results = run_harness_tool(
                    "phase_aware_context",
                    current_step,
                    {},
                    phase_aware_scaffolder.build
                    if phase_aware_scaffolder is not None
                    else None,
                    selector_observe_dict,
                    latest_phase_evidence,
                    multimodal_selector_context_results,
                    latest_failure_signal,
                )
                if phase_aware_scaffolder is not None:
                    v4_context_applied = bool(
                        isinstance(multimodal_selector_context_results, dict)
                        and multimodal_selector_context_results.get("applied")
                    )
                    phase_context_should_apply = (
                        phase_aware_scaffolder_decision_effect
                        and isinstance(phase_aware_context_results, dict)
                        and not phase_aware_context_results.get("skipped")
                        and phase_aware_context_results.get(
                            "eligible_for_application",
                            True,
                        )
                        and isinstance(
                            phase_aware_context_results.get(
                                "augmented_observe_dict"
                            ),
                            dict,
                        )
                    )
                    phase_aware_context_results[
                        "decision_effect_enabled"
                    ] = phase_aware_scaffolder_decision_effect
                    phase_aware_context_results[
                        "would_apply"
                    ] = phase_context_should_apply
                    phase_aware_context_results["applied"] = False
                    phase_aware_context_results["input_context_source"] = (
                        "v4_augmented" if v4_context_applied else "raw_observation"
                    )
                    phase_aware_context_results[
                        "u1_shadow_matches_ablation_context"
                    ] = not v4_context_applied

                def phase_context_not_applied_reason():
                    if not phase_aware_scaffolder_decision_effect:
                        return "log_only"
                    if not isinstance(phase_aware_context_results, dict):
                        return "invalid_context"
                    if phase_aware_context_results.get("skipped"):
                        return "skipped"
                    if not phase_aware_context_results.get(
                        "eligible_for_application",
                        True,
                    ):
                        return "phase_not_eligible"
                    return ""

                def apply_phase_aware_context_for_selector():
                    nonlocal selector_observe_dict
                    nonlocal selector_observation
                    nonlocal phase_context_applied
                    if phase_aware_scaffolder is None:
                        return
                    if phase_context_should_apply:
                        original_selector_candidate_ids = list(
                            selector_observe_dict.keys()
                        )
                        selector_observe_dict = phase_aware_context_results[
                            "augmented_observe_dict"
                        ]
                        selector_observation = (
                            phase_aware_context_results.get(
                                "augmented_observation"
                            )
                            or list(selector_observe_dict.values())
                        )
                        phase_context_applied = True
                        phase_aware_context_results["applied"] = True
                        context_override = build_decision_audit(
                            unit_enabled=u_decision_unit,
                            original_action="selector_context",
                            proposed_action="phase_aware_context",
                            final_action="phase_aware_context",
                            override_reason="u1_phase_aware_selector_context",
                            expected_failure_addressed="progress_drift",
                            source_stage="phase_aware_context",
                            extra={
                                "phase": phase_aware_context_results.get("phase"),
                                "context_mode": (
                                    phase_aware_context_results.get("context_mode")
                                ),
                                "candidate_ids": [
                                    str(candidate_id)
                                    for candidate_id in (
                                        original_selector_candidate_ids
                                    )
                                ],
                                "input_context_source": (
                                    phase_aware_context_results.get(
                                        "input_context_source"
                                    )
                                ),
                            },
                        )
                        context_override["effect_type"] = "selector_context"
                        context_override["action_affecting"] = False
                        record_u_override(context_override)
                    else:
                        not_applied_reason = phase_context_not_applied_reason()
                        if not_applied_reason:
                            phase_aware_context_results[
                                "not_applied_reason"
                            ] = not_applied_reason

                def get_stop_current_view_evidence():
                    nonlocal stop_current_view_evidence_results
                    if stop_current_view_evidence_results is not None:
                        return stop_current_view_evidence_results
                    stop_current_view_evidence_results = run_harness_tool(
                        "stop_current_view_evidence",
                        current_step,
                        {},
                        visual_evidence.run if visual_evidence is not None else None,
                        instruction,
                        actions,
                        landmarks,
                        [],
                        images_list,
                        observe_dict,
                        stop_current_view_evidence=True,
                    )
                    if visual_evidence is not None:
                        write_navigation_record(
                            "stop_current_view_evidence",
                            episode_id=current_episode_id,
                            step=current_step,
                            **stop_current_view_evidence_results,
                        )
                        if harness_enabled and harness_logger is not None:
                            harness_logger.log_event(
                                "stop_current_view_evidence",
                                current_step,
                                stop_current_view_evidence_results,
                            )
                    return stop_current_view_evidence_results

                def record_visual_target_verifier(
                    source,
                    stop_proposal,
                    reason,
                    selected_candidate=None,
                    stop_evidence_mode="selected_candidate",
                ):
                    verifier_selected_candidate = (
                        selected_candidate if stop_proposal else None
                    )
                    verifier_visual_evidence = (
                        get_stop_current_view_evidence()
                        if stop_proposal
                        else visual_evidence_results
                    )
                    verifier_results = run_harness_tool(
                        "visual_target_verifier",
                        current_step,
                        {},
                        visual_target_verifier.verify
                        if visual_target_verifier is not None
                        else None,
                        source,
                        instruction,
                        actions,
                        landmarks,
                        estimation,
                        history_traj,
                        observation,
                        verifier_visual_evidence,
                        stop_proposal,
                        reason,
                        verifier_selected_candidate,
                        current_step,
                        stop_evidence_mode,
                    )
                    if visual_target_verifier is not None:
                        write_navigation_record(
                            "visual_target_verifier",
                            episode_id=current_episode_id,
                            step=current_step,
                            **verifier_results,
                        )
                        harness_logger.log_event(
                            "visual_target_verifier",
                            current_step,
                            verifier_results,
                        )
                    return verifier_results

                def record_stop_evidence_verification(
                    source,
                    verifier_results,
                    stop_gate_context,
                ):
                    _carry_forward = any(recent_ftv_window[-3:]) if recent_ftv_window else False
                    # Unified persistence guard (ORACLE REMOVED 20260704): the removed
                    # geodesic-distance conjuncts are replaced everywhere by >=2
                    # consecutive steps of observable final_target_visible. This is the
                    # single choke point for every verify() call (selector stop gate,
                    # completion gate, proactive stop gate), so #2 e3_arrival_override
                    # and #4 trajectory_bypass all consume the same signal that #5 PSG
                    # commit already uses.
                    _persistent_visual = (
                        len(recent_ftv_window) >= 2 and all(recent_ftv_window[-2:])
                    )
                    stop_evidence_results = run_harness_tool(
                        "stop_evidence_verifier",
                        current_step,
                        {},
                        stop_evidence_verifier.verify
                        if stop_evidence_verifier is not None
                        else None,
                        source,
                        verifier_results,
                        stop_gate_context,
                        latest_phase_evidence,
                        instruction,
                        actions,
                        landmarks,
                        estimation,
                        current_step,
                        latest_goal_dist=latest_goal_dist,
                        carry_forward_visible=_carry_forward,
                        persistent_visual_confirm=_persistent_visual,
                    )
                    if stop_evidence_verifier is not None:
                        if isinstance(stop_evidence_results, dict):
                            stop_evidence_results[
                                "decision_effect_enabled"
                            ] = stop_evidence_verifier_decision_effect
                            stop_evidence_results["decision_scope"] = (
                                "selector_visual_rescue_only"
                                if stop_evidence_verifier_decision_effect
                                else "log_only"
                            )
                        log_u_event(
                            "stop_verification",
                            current_episode_id,
                            current_step,
                            stop_evidence_results,
                        )
                        weak_target_adjustment = stop_evidence_results.get(
                            "weak_target_adjustment",
                            "none",
                        )
                        if (
                            stop_evidence_results.get("weak_target")
                            or weak_target_adjustment != "none"
                        ):
                            log_u_event(
                                "weak_target_decision",
                                current_episode_id,
                                current_step,
                                {
                                    "source": source,
                                    "target_type": stop_evidence_results.get(
                                        "target_type"
                                    ),
                                    "weak_target": stop_evidence_results.get(
                                        "weak_target"
                                    ),
                                    "landmark_terms": stop_evidence_results.get(
                                        "landmark_terms"
                                    ),
                                    "weak_target_adjustment": (
                                        stop_evidence_results.get(
                                            "weak_target_adjustment"
                                        )
                                    ),
                                    "allow_stop": stop_evidence_results.get(
                                        "allow_stop"
                                    ),
                                    "allow_rescue": stop_evidence_results.get(
                                        "allow_rescue"
                                    ),
                                    "reject_reasons": stop_evidence_results.get(
                                        "reject_reasons"
                                    ),
                                },
                            )
                    return stop_evidence_results

                def visual_target_verifier_rejects_stop(verifier_results):
                    if not visual_target_verifier_decision_effect:
                        return False
                    if not isinstance(verifier_results, dict):
                        return False
                    if verifier_results.get("skipped"):
                        return False
                    verdict = verifier_results.get("verdict")
                    if verdict == "allow":
                        return False
                    if verdict == "reject":
                        return True
                    if verdict == "uncertain":
                        return visual_target_verifier_reject_on_uncertain
                    return False

                def visual_target_verifier_allows_stop(verifier_results):
                    return (
                        visual_target_verifier_decision_effect
                        and isinstance(verifier_results, dict)
                        and not verifier_results.get("skipped")
                        and verifier_results.get("verdict") == "allow"
                    )

                def visual_stop_can_rescue_selector_stop(
                    verifier_results,
                    old_stop_flag,
                ):
                    def has_hard_allow_warnings(results):
                        return any(
                            not str(warning).startswith(
                                "uncorroborated_final_target:"
                            )
                            for warning in (results.get("allow_warnings") or [])
                        )

                    if old_stop_flag or not visual_target_verifier_decision_effect:
                        return False
                    if not isinstance(verifier_results, dict):
                        return False
                    if verifier_results.get("skipped"):
                        return False
                    if verifier_results.get("source") != "selector_stop_gate":
                        return False
                    if verifier_results.get("stop_evidence_mode") != "current_pano":
                        return False
                    if (
                        verifier_results.get("stop_relevant_candidate_id")
                        != stop_current_view_candidate_id
                    ):
                        return False
                    selected_verdict = (
                        verifier_results.get("selected_candidate_verdict") or {}
                    )
                    if selected_verdict.get("verdict") != "allow":
                        return False
                    if not selected_verdict.get("final_target_visible"):
                        return False
                    if not selected_verdict.get("arrival_evidence"):
                        return False
                    if has_hard_allow_warnings(verifier_results):
                        return False
                    if selected_verdict.get("missing_instruction_terms"):
                        return False
                    contradictions = set(
                        verifier_results.get("contradictions") or []
                    )
                    if "stop_proposed_while_estimation_is_negative" in contradictions:
                        return False
                    hard_blockers = [
                        blocker
                        for blocker in (verifier_results.get("allow_blockers") or [])
                        if blocker != "selector_stop_without_completion_support"
                    ]
                    return not hard_blockers

                def log_visual_stop_allowed(
                    source,
                    original_stop_reason,
                    verifier_results,
                ):
                    allowed_payload = {
                        "source": source,
                        "original_stop_reason": original_stop_reason,
                        "verdict": verifier_results.get("verdict"),
                        "verifier_reason": verifier_results.get("reason"),
                        "supporting_candidate_id": verifier_results.get(
                            "supporting_candidate_id"
                        ),
                        "future_supporting_candidate_id": verifier_results.get(
                            "future_supporting_candidate_id"
                        ),
                        "stop_relevant_candidate_id": verifier_results.get(
                            "stop_relevant_candidate_id"
                        ),
                        "stop_evidence_mode": verifier_results.get(
                            "stop_evidence_mode"
                        ),
                        "candidate_alignment": verifier_results.get(
                            "candidate_alignment"
                        ),
                        "final_target_visible": verifier_results.get(
                            "final_target_visible"
                        ),
                        "arrival_evidence": verifier_results.get(
                            "arrival_evidence"
                        ),
                        "confidence": verifier_results.get("confidence"),
                        "allow_blockers": verifier_results.get("allow_blockers"),
                        "allow_warnings": verifier_results.get("allow_warnings"),
                    }
                    write_navigation_record(
                        "visual_stop_allowed",
                        episode_id=current_episode_id,
                        step=current_step,
                        **allowed_payload,
                    )
                    if harness_enabled and harness_logger is not None:
                        harness_logger.log_event(
                            "visual_stop_allowed",
                            current_step,
                            allowed_payload,
                        )

                def apply_visual_stop_gate(
                    source,
                    stop_proposal,
                    reason,
                    verifier_results,
                ):
                    if not stop_proposal or not visual_target_verifier_rejects_stop(
                        verifier_results
                    ):
                        return stop_proposal, reason, False
                    rejection_payload = {
                        "source": source,
                        "original_stop_reason": reason,
                        "verdict": verifier_results.get("verdict"),
                        "verifier_reason": verifier_results.get("reason"),
                        "reject_on_uncertain": (
                            visual_target_verifier_reject_on_uncertain
                        ),
                        "final_target_visible": verifier_results.get(
                            "final_target_visible"
                        ),
                        "arrival_evidence": verifier_results.get(
                            "arrival_evidence"
                        ),
                        "missing_final_landmarks": verifier_results.get(
                            "missing_final_landmarks"
                        ),
                        "contradictions": verifier_results.get("contradictions"),
                        "allow_blockers": verifier_results.get("allow_blockers"),
                        "allow_warnings": verifier_results.get("allow_warnings"),
                        "visual_evidence_parse_error": verifier_results.get(
                            "visual_evidence_parse_error"
                        ),
                        "stop_relevant_candidate_id": verifier_results.get(
                            "stop_relevant_candidate_id"
                        ),
                        "stop_evidence_mode": verifier_results.get(
                            "stop_evidence_mode"
                        ),
                    }
                    write_navigation_record(
                        "visual_stop_rejected",
                        episode_id=current_episode_id,
                        step=current_step,
                        **rejection_payload,
                    )
                    if harness_enabled and harness_logger is not None:
                        harness_logger.log_event(
                            "visual_stop_rejected",
                            current_step,
                            rejection_payload,
                        )
                    nav_logger.info(
                        "Visual target verifier rejected STOP from {}: {}".format(
                            source,
                            verifier_results.get("reason"),
                        )
                    )
                    return False, "", True

                def log_visual_stop_rescued(
                    source,
                    original_stop_reason,
                    verifier_results,
                    old_stop_flag,
                    old_stop_reason,
                ):
                    rescued_payload = {
                        "source": source,
                        "original_stop_reason": original_stop_reason,
                        "old_stop_flag": old_stop_flag,
                        "old_stop_reason": old_stop_reason,
                        "verdict": verifier_results.get("verdict"),
                        "verifier_reason": verifier_results.get("reason"),
                        "supporting_candidate_id": verifier_results.get(
                            "supporting_candidate_id"
                        ),
                        "stop_relevant_candidate_id": verifier_results.get(
                            "stop_relevant_candidate_id"
                        ),
                        "stop_evidence_mode": verifier_results.get(
                            "stop_evidence_mode"
                        ),
                        "candidate_alignment": verifier_results.get(
                            "candidate_alignment"
                        ),
                        "selected_candidate_verdict": verifier_results.get(
                            "selected_candidate_verdict"
                        ),
                        "final_target_visible": verifier_results.get(
                            "final_target_visible"
                        ),
                        "arrival_evidence": verifier_results.get(
                            "arrival_evidence"
                        ),
                        "confidence": verifier_results.get("confidence"),
                        "allow_blockers": verifier_results.get("allow_blockers"),
                        "allow_warnings": verifier_results.get("allow_warnings"),
                        "contradictions": verifier_results.get("contradictions"),
                    }
                    write_navigation_record(
                        "visual_stop_rescued",
                        episode_id=current_episode_id,
                        step=current_step,
                        **rescued_payload,
                    )
                    if harness_enabled and harness_logger is not None:
                        harness_logger.log_event(
                            "visual_stop_rescued",
                            current_step,
                            rescued_payload,
                        )

                stop_flag, stop_reason = navigator.should_stop(
                    nav_logger,
                    actions,
                    landmarks,
                    estimation,
                    history_traj,
                    observation,
                    current_step=current_step,
                )
                stop_gate_metadata = dict(
                    getattr(navigator, "last_stop_gate_metadata", {}) or {}
                )
                if (
                    stop_gate_metadata.get("rejection_reason")
                    == "weak_final_target_completion_auto_stop"
                ):
                    weak_stop_payload = {
                        "source": "completion_gate",
                        "rejection_reason": stop_gate_metadata.get(
                            "rejection_reason"
                        ),
                        "weak_final_target": stop_gate_metadata.get(
                            "weak_final_target"
                        ),
                        "current_step": stop_gate_metadata.get(
                            "current_step"
                        ),
                        "weak_final_target_min_steps": (
                            stop_gate_metadata.get(
                                "weak_final_target_min_steps"
                            )
                        ),
                        "landmark_gate": stop_gate_metadata.get("landmark_gate"),
                        "completed_actions": stop_gate_metadata.get(
                            "completed_actions"
                        ),
                        "all_actions_completed": stop_gate_metadata.get(
                            "all_actions_completed"
                        ),
                        "final_action_completed": stop_gate_metadata.get(
                            "final_action_completed"
                        ),
                    }
                    write_navigation_record(
                        "completion_gate_weak_final_target",
                        episode_id=current_episode_id,
                        step=current_step,
                        **weak_stop_payload,
                    )
                    if harness_enabled and harness_logger is not None:
                        harness_logger.log_event(
                            "completion_gate_weak_final_target",
                            current_step,
                            weak_stop_payload,
                        )
                completion_verifier_results = record_visual_target_verifier(
                    "completion_gate",
                    stop_flag,
                    stop_reason,
                    selected_candidate=stop_current_view_candidate_id,
                    stop_evidence_mode="current_pano",
                )
                # E3 carry-forward: track per-step final_target_visible signal.
                # Window size 3 — used by stop_evidence_verifier for E3-C abstain.
                if isinstance(completion_verifier_results, dict):
                    _ftv = bool(completion_verifier_results.get("final_target_visible"))
                    recent_ftv_window.append(_ftv)
                    if len(recent_ftv_window) > 10:
                        recent_ftv_window.pop(0)
                record_stop_evidence_verification(
                    "completion_gate",
                    completion_verifier_results,
                    stop_gate_metadata,
                )
                stop_flag, stop_reason, _ = apply_visual_stop_gate(
                    "completion_gate",
                    stop_flag,
                    stop_reason,
                    completion_verifier_results,
                )
                if stop_flag and visual_target_verifier_allows_stop(
                    completion_verifier_results
                ):
                    log_visual_stop_allowed(
                        "completion_gate",
                        stop_reason,
                        completion_verifier_results,
                    )
                # M3: proactive stop gate — when near goal with visual evidence,
                # re-evaluate with stop_flag=True so V2 gives a real verdict.
                # E3-for-M3: V2 re-evaluation fires for dist < dist_threshold, but
                # STOP is only committed when dist < commit_dist_threshold (inner
                # zone, inside success radius). This prevents premature commitment
                # in the [commit_dist, dist_threshold] uncertain outer zone.
                # E3-C carry-forward PSG: when target was visible in recent steps but
                # current frame shows not_visible (target fills frame at <2m), fire PSG
                # anyway if dist < commit_dist. U2 E3-B then commits via carry-forward.
                _carry_forward_psg = any(recent_ftv_window[-3:]) if recent_ftv_window else False
                # PSG persistence guard: >=2 consecutive steps of final_target_visible
                # substitutes for the removed near-distance commit zone (spatial
                # resolution -> temporal consistency). See ORACLE REMOVED note below.
                _persistent_visual_confirm = (
                    len(recent_ftv_window) >= 2 and all(recent_ftv_window[-2:])
                )
                # ORACLE REMOVED (20260704): PSG previously gated entry on
                # latest_goal_dist < proactive_stop_dist and the carry-forward branch on
                # latest_goal_dist < proactive_stop_commit_dist (simulator geodesic goal
                # distance = GT leakage). Distance conjuncts deleted; entry keys purely
                # on observable visual arrival evidence. proactive_stop_* kept as toggles.
                if (
                    not stop_flag
                    and proactive_stop_enabled
                    and (
                        (
                            bool(completion_verifier_results.get("final_target_visible"))
                            and bool(completion_verifier_results.get("arrival_evidence"))
                        )
                        or _carry_forward_psg
                    )
                ):
                    # ORACLE REMOVED (20260704): reason strings no longer claim a
                    # distance threshold (the geodesic gate is gone; PSG now keys on
                    # visual evidence + the >=2-step persistence guard). Stating a
                    # distance here would misdescribe stops that commit far from goal.
                    proactive_reason = (
                        "Proactive stop: target visible with arrival evidence "
                        "(persistent visual confirmation)."
                        if (bool(completion_verifier_results.get("final_target_visible"))
                            and bool(completion_verifier_results.get("arrival_evidence")))
                        else
                        "Proactive stop: carry-forward visual evidence "
                        "(persistent visual confirmation)."
                    )
                    proactive_verifier_results = record_visual_target_verifier(
                        "proactive_stop_gate",
                        True,
                        proactive_reason,
                        selected_candidate=stop_current_view_candidate_id,
                        stop_evidence_mode="current_pano",
                    )
                    proactive_stop_evidence = record_stop_evidence_verification(
                        "proactive_stop_gate",
                        proactive_verifier_results,
                        stop_gate_metadata,
                    )
                    # ORACLE REMOVED (20260704): commit previously required
                    # _within_commit_zone = latest_goal_dist < proactive_stop_commit_dist
                    # (simulator geodesic goal distance = GT leakage). Replaced by the
                    # >=2-step visual persistence guard (temporal consistency in place of
                    # spatial resolution).
                    # E3-B commit: carry_forward arrival_evidence (observable) substitutes
                    # for V2 allow; e3.arrival_override is now itself gated on pure visual
                    # evidence inside stop_evidence_verifier.
                    _e3_b_allow = bool(
                        isinstance(proactive_stop_evidence, dict)
                        and proactive_stop_evidence.get("allow_stop")
                        and proactive_stop_evidence.get("e3", {}).get("arrival_override")
                    )
                    if (
                        (_persistent_visual_confirm and visual_target_verifier_allows_stop(proactive_verifier_results))
                        or _e3_b_allow
                    ):
                        stop_flag = True
                        stop_reason = proactive_reason
                        log_visual_stop_allowed(
                            "proactive_stop_gate",
                            stop_reason,
                            proactive_verifier_results,
                        )
                if stop_flag:
                    if phase_aware_scaffolder is not None:
                        phase_aware_context_results[
                            "not_applied_reason"
                        ] = "completion_stop_before_selector"
                        log_u_event(
                            "phase_aware_context",
                            current_episode_id,
                            current_step,
                            phase_aware_context_results,
                        )
                    next_vp = STOP_CANDIDATE
                    thought = stop_reason
                    nav_logger.info(f"========== Stop Decision: {stop_reason} ==========")
                    write_navigation_record(
                        "stop_decision",
                        episode_id=current_episode_id,
                        step=current_step,
                        reason=stop_reason,
                        estimation=estimation,
                        stop_gate_metadata=stop_gate_metadata,
                    )
                    write_navigation_record(
                        "selector_final",
                        episode_id=current_episode_id,
                        step=current_step,
                        selected_candidate=next_vp,
                        thought=thought,
                        error_number=error_number,
                        selector_context_applied=(
                            multimodal_selector_context_results.get(
                                "applied", False
                            )
                            if isinstance(
                                multimodal_selector_context_results, dict
                            )
                            else False
                        ),
                        phase_context_applied=phase_context_applied,
                    )
                    if harness_enabled and harness_logger is not None:
                        harness_logger.log_event(
                            "stop_decision",
                            current_step,
                            {
                                "reason": stop_reason,
                                "estimation": estimation,
                            },
                        )
                        harness_logger.log_event(
                            "selector_final",
                            current_step,
                            {
                                "selected_candidate": next_vp,
                                "thought": thought,
                                "error_number": error_number,
                                "selector_context_applied": (
                                    multimodal_selector_context_results.get(
                                        "applied", False
                                    )
                                    if isinstance(
                                        multimodal_selector_context_results, dict
                                    )
                                    else False
                                ),
                                "phase_context_applied": phase_context_applied,
                            },
                        )
                else:
                    apply_phase_aware_context_for_selector()
                    if phase_aware_scaffolder is not None:
                        log_u_event(
                            "phase_aware_context",
                            current_episode_id,
                            current_step,
                            phase_aware_context_results,
                        )
                    nav_logger.info("========== Next Action Prediction ==========")
                    selector_start_time = time.perf_counter()
                    predictions, thoughts, break_flag, navigator_prompt = navigator.move_to_next_vp(
                        nav_logger,
                        current_step,
                        instruction,
                        actions,
                        landmarks,
                        history_traj,
                        estimation,
                        selector_observation,
                        selector_observe_dict,
                        max_tokens=positive_token_cap(navigator_max_tokens),
                        return_prompt=True,
                    )
                    # P1 (Path B): log the exact assembled navigator prompt so offline
                    # dispersion calibration can replay it verbatim (zero reconstruction).
                    # Use log_fallback_event so it lands in the harness trace (event_type/
                    # step_id schema) that p1_dispersion_calib.py reads, not just nav_jsonl.
                    log_fallback_event(
                        "navigator_prompt",
                        current_episode_id,
                        current_step,
                        {
                            "prompt": navigator_prompt,
                            "candidate_ids": [str(key) for key in selector_observe_dict.keys()],
                        },
                    )
                    record_runtime_latency(
                        "navigator_move_to_next_vp",
                        current_episode_id,
                        current_step,
                        time.perf_counter() - selector_start_time,
                        category="text_llm",
                        max_tokens=positive_token_cap(navigator_max_tokens),
                    )
                    write_navigation_record(
                        "selector_raw",
                        episode_id=current_episode_id,
                        step=current_step,
                        predictions=predictions,
                        thoughts=thoughts,
                        break_flag=break_flag,
                        selector_context_applied=(
                            multimodal_selector_context_results.get(
                                "applied", False
                            )
                            if isinstance(
                                multimodal_selector_context_results, dict
                            )
                            else False
                        ),
                        phase_context_applied=phase_context_applied,
                    )
                    if harness_enabled and harness_logger is not None:
                        harness_logger.log_event(
                            "selector_raw",
                            current_step,
                            {
                                "predictions": predictions,
                                "thoughts": thoughts,
                                "break_flag": break_flag,
                                "estimation": estimation,
                                "selector_context_applied": (
                                    multimodal_selector_context_results.get(
                                        "applied", False
                                    )
                                    if isinstance(
                                        multimodal_selector_context_results, dict
                                    )
                                    else False
                                ),
                                "phase_context_applied": phase_context_applied,
                            },
                        )
                    if not predictions:
                        fallback_results = rank_movement_fallback(
                            "selector_empty_prediction_fallback",
                            current_step,
                            selector_observe_dict,
                            visual_evidence_results,
                            instruction,
                            actions,
                            landmarks,
                            source_stage="selector_raw",
                            reason="empty_selector_prediction",
                        )
                        selected_fallback = (
                            fallback_results.get("selected_candidate")
                            if isinstance(fallback_results, dict)
                            else None
                        )
                        if selected_fallback in selector_observe_dict:
                            predictions = [selected_fallback]
                            thoughts = [selector_observe_dict[selected_fallback]]
                        fallback_results["previous_predictions"] = []
                        latest_failure_signal = run_harness_tool(
                            "failure_type_diagnostic",
                            current_step,
                            {},
                            failure_diagnostic.diagnose
                            if failure_diagnostic is not None
                            else None,
                            "selector_empty",
                            current_step,
                            latest_phase_evidence,
                            recent_distance_gains,
                            fallback_results,
                            {},
                            {
                                "predictions": [],
                                "thoughts": thoughts,
                                "source_stage": "selector_raw",
                            },
                        )
                        if failure_diagnostic is not None:
                            log_u_event(
                                "failure_type_diagnostic",
                                current_episode_id,
                                current_step,
                                latest_failure_signal,
                            )
                        if selected_fallback in selector_observe_dict:
                            recovered_candidate, recovered_thought, recovery_results = (
                                apply_failure_recovery(
                                    "selector_empty",
                                    "selector_empty_prediction_fallback",
                                    selector_observe_dict,
                                    fallback_results,
                                    latest_failure_signal,
                                    selected_fallback,
                                    selector_observe_dict[selected_fallback],
                                )
                            )
                            fallback_results["recovery_results"] = recovery_results
                            if recovered_candidate in selector_observe_dict:
                                selected_fallback = recovered_candidate
                                predictions = [recovered_candidate]
                                thoughts = [recovered_thought]
                                if (
                                    isinstance(recovery_results, dict)
                                    and recovery_results.get("applied_to_action")
                                ):
                                    fallback_results[
                                        "selected_candidate_after_recovery"
                                    ] = recovered_candidate
                        log_fallback_event(
                            "selector_empty_prediction_fallback",
                            current_episode_id,
                            current_step,
                            fallback_results,
                        )

                    nav_logger.info("========== Thought ==========")
                    thought_fusion_start_time = time.perf_counter()
                    fused_pred_thought = navigator.thought_fusion(
                        nav_logger,
                        predictions,
                        thoughts,
                        max_tokens=positive_token_cap(thought_fusion_max_tokens),
                    )
                    record_runtime_latency(
                        "thought_fusion",
                        current_episode_id,
                        current_step,
                        time.perf_counter() - thought_fusion_start_time,
                        category="text_llm",
                        max_tokens=positive_token_cap(thought_fusion_max_tokens),
                        prediction_count=len(predictions),
                    )
                    write_navigation_record(
                        "selector_fused",
                        episode_id=current_episode_id,
                        step=current_step,
                        fused_predictions=fused_pred_thought,
                    )
                    if harness_enabled and harness_logger is not None:
                        harness_logger.log_event(
                            "selector_fused",
                            current_step,
                            {"fused_predictions": fused_pred_thought},
                        )

                    nav_logger.info("========== Test Decision ==========")
                    test_decision_start_time = time.perf_counter()
                    next_vp, thought, error_number = navigator.test_decisions(
                        nav_logger,
                        fused_pred_thought,
                        selector_observation,
                        instruction,
                        error_number,
                        selector_observe_dict,
                        max_tokens=positive_token_cap(decision_max_tokens),
                    )
                    record_runtime_latency(
                        "test_decision",
                        current_episode_id,
                        current_step,
                        time.perf_counter() - test_decision_start_time,
                        category="text_llm",
                        max_tokens=positive_token_cap(decision_max_tokens),
                        fused_candidate_count=len(fused_pred_thought),
                    )
                    if next_vp == STOP_CANDIDATE:
                        old_stop_flag, old_stop_reason = navigator.should_stop(
                            nav_logger,
                            actions,
                            landmarks,
                            estimation,
                            history_traj,
                            observation,
                            current_step=current_step,
                        )
                        old_stop_gate_metadata = dict(
                            getattr(navigator, "last_stop_gate_metadata", {})
                            or {}
                        )
                        if old_stop_gate_metadata:
                            selector_stop_gate_payload = dict(
                                old_stop_gate_metadata
                            )
                            selector_stop_gate_payload[
                                "source"
                            ] = "selector_stop_gate"
                            write_navigation_record(
                                "selector_stop_gate_metadata",
                                episode_id=current_episode_id,
                                step=current_step,
                                **selector_stop_gate_payload,
                            )
                            if harness_enabled and harness_logger is not None:
                                harness_logger.log_event(
                                    "selector_stop_gate_metadata",
                                    current_step,
                                    selector_stop_gate_payload,
                                )
                        stop_flag = old_stop_flag
                        stop_reason = old_stop_reason
                        selector_stop_rejection_reason = (
                            "navigator_stop_failed_stop_gate"
                        )
                        selector_verifier_results = record_visual_target_verifier(
                            "selector_stop_gate",
                            True,
                            old_stop_reason or "Navigator selected STOP.",
                            selected_candidate=stop_current_view_candidate_id,
                            stop_evidence_mode="current_pano",
                        )
                        selector_stop_evidence_results = (
                            record_stop_evidence_verification(
                                "selector_stop_gate",
                                selector_verifier_results,
                                old_stop_gate_metadata,
                            )
                        )
                        selector_visual_rescue_candidate = (
                            visual_stop_can_rescue_selector_stop(
                                selector_verifier_results,
                                old_stop_flag,
                            )
                        )
                        u2_rescue_allowed = True
                        if (
                            stop_evidence_verifier_decision_effect
                            and selector_visual_rescue_candidate
                        ):
                            u2_rescue_allowed = bool(
                                selector_stop_evidence_results.get("allow_rescue")
                            )
                        # E5: block visual-STOP rescue/override when the target has
                        # never been visually confirmed this episode (guards against
                        # far false stops such as ep244).
                        # ORACLE REMOVED (20260704): previously gated on latest_goal_dist
                        # > rescue_max_goal_dist (simulator geodesic goal distance = GT
                        # leakage). Replaced by the observable "target never visible"
                        # signal from recent_ftv_window. rescue_max_goal_dist > 0 kept
                        # only as the enable toggle; variable name retained downstream.
                        _never_visually_confirmed = not any(recent_ftv_window)
                        _rescue_dist_blocked = (
                            rescue_max_goal_dist > 0
                            and _never_visually_confirmed
                        )
                        if _rescue_dist_blocked:
                            nav_logger.info(
                                "E5: rescue/override STOP blocked, target never "
                                "visually confirmed this episode."
                            )
                        if old_stop_flag and visual_target_verifier_allows_stop(
                            selector_verifier_results
                        ):
                            stop_flag = True
                            stop_reason = (
                                "Navigator selected STOP and visual target "
                                "verifier allowed STOP."
                            )
                            log_visual_stop_allowed(
                                "selector_stop_gate",
                                old_stop_reason
                                or "Navigator selected STOP.",
                                selector_verifier_results,
                            )
                        elif (
                            selector_visual_rescue_candidate
                            and u2_rescue_allowed
                            and not _rescue_dist_blocked
                        ):
                            stop_flag = True
                            stop_reason = (
                                "Navigator selected STOP and current-view visual "
                                "evidence rescued STOP."
                            )
                            log_visual_stop_rescued(
                                "selector_stop_gate",
                                old_stop_reason or "Navigator selected STOP.",
                                selector_verifier_results,
                                old_stop_flag,
                                old_stop_reason,
                            )
                        elif (
                            selector_visual_rescue_candidate
                            and not u2_rescue_allowed
                        ):
                            if (
                                visual_target_verifier_allows_stop(
                                    selector_verifier_results
                                )
                                and selector_stop_evidence_results.get("allow_stop")
                                and not _rescue_dist_blocked
                            ):
                                stop_flag = True
                                stop_reason = (
                                    "Navigator selected STOP; visual verifier "
                                    "allowed and stop evidence confirmed; "
                                    "rescue override granted despite gain/phase "
                                    "conditions."
                                )
                                log_visual_stop_allowed(
                                    "selector_stop_gate",
                                    old_stop_reason or "Navigator selected STOP.",
                                    selector_verifier_results,
                                )
                                nav_logger.info(
                                    "V2+U2 both allow STOP; granting rescue despite "
                                    "gain/phase blocker."
                                )
                            else:
                                selector_stop_rejection_reason = (
                                    "u2_stop_evidence_blocked_visual_rescue"
                                )
                                override_payload = build_decision_audit(
                                    unit_enabled=u_decision_unit,
                                    original_action=STOP_CANDIDATE,
                                    proposed_action=STOP_CANDIDATE,
                                    final_action="movement_fallback",
                                    override_reason=selector_stop_rejection_reason,
                                    expected_failure_addressed="stop_false_positive",
                                    source_stage="selector_stop_gate",
                                    extra={
                                        "source": "selector_stop_gate",
                                        "stop_evidence": selector_stop_evidence_results,
                                    },
                                )
                                override_payload["effect_type"] = "stop_rescue_block"
                                override_payload["action_affecting"] = True
                                record_u_override(override_payload)
                                nav_logger.info(
                                    "U2 stop evidence blocked visual STOP rescue; fallback to movement."
                                )
                        elif visual_target_verifier_allows_stop(
                            selector_verifier_results
                        ):
                            nav_logger.info(
                                "Navigator selected STOP and visual target verifier allowed it, "
                                "but rescue conditions did not pass; fallback to movement."
                            )
                            selector_stop_rejection_reason = (
                                "visual_stop_rescue_conditions_failed"
                            )
                        elif visual_target_verifier_rejects_stop(
                            selector_verifier_results
                        ):
                            stop_flag, stop_reason, _ = apply_visual_stop_gate(
                                "selector_stop_gate",
                                True,
                                old_stop_reason or "Navigator selected STOP.",
                                selector_verifier_results,
                            )
                            selector_stop_rejection_reason = (
                                "visual_target_verifier_rejected_stop"
                            )
                        # E3-B standalone override: V2 rejected but U2 confirms
                        # carry-forward arrival evidence at near-goal distance.
                        # Only fires when stop_evidence_verifier has decision effect.
                        if (
                            not stop_flag
                            and stop_evidence_verifier_decision_effect
                            and isinstance(selector_stop_evidence_results, dict)
                            and selector_stop_evidence_results.get("allow_stop")
                            and isinstance(
                                selector_stop_evidence_results.get("e3"), dict
                            )
                            and selector_stop_evidence_results["e3"].get(
                                "arrival_override"
                            )
                        ):
                            stop_flag = True
                            stop_reason = (
                                "U2 E3-B carry-forward arrival override: target visible "
                                "in recent steps; STOP allowed at near-goal distance."
                            )
                            selector_stop_rejection_reason = ""
                            e3b_payload = {
                                "source": "selector_stop_gate",
                                "original_stop_reason": old_stop_reason or "Navigator selected STOP.",
                                "e3": selector_stop_evidence_results.get("e3"),
                                "allow_stop": True,
                                "v2_verdict": selector_verifier_results.get("verdict"),
                                "latest_goal_dist": (
                                    selector_stop_evidence_results.get("e3") or {}
                                ).get("latest_goal_dist"),
                            }
                            write_navigation_record(
                                "visual_stop_e3b_override",
                                episode_id=current_episode_id,
                                step=current_step,
                                **e3b_payload,
                            )
                            if harness_enabled and harness_logger is not None:
                                harness_logger.log_event(
                                    "visual_stop_e3b_override",
                                    current_step,
                                    e3b_payload,
                                )
                            nav_logger.info(
                                "U2 E3-B carry-forward override: committing STOP "
                                "despite V2 not_visible rejection."
                            )
                        if stop_flag:
                            if not stop_reason:
                                stop_reason = (
                                    "Navigator selected STOP and stop gate passed."
                                )
                        else:
                            nav_logger.info("Navigator selected STOP but stop gate failed; fallback to a movement candidate")
                            filtered_fused_pred_thought = {
                                key: value for key, value in fused_pred_thought.items()
                                if key != STOP_CANDIDATE
                            }
                            stop_fallback_start_time = time.perf_counter()
                            next_vp, thought, error_number = navigator.test_decisions(
                                nav_logger,
                                filtered_fused_pred_thought,
                                selector_observation,
                                instruction,
                                error_number,
                                selector_observe_dict,
                                max_tokens=positive_token_cap(decision_max_tokens),
                            )
                            record_runtime_latency(
                                "test_decision_after_stop_rejection",
                                current_episode_id,
                                current_step,
                                time.perf_counter() - stop_fallback_start_time,
                                category="text_llm",
                                max_tokens=positive_token_cap(decision_max_tokens),
                                fused_candidate_count=len(
                                    filtered_fused_pred_thought
                                ),
                            )
                            test_decision_metadata = dict(
                                getattr(
                                    navigator,
                                    "last_test_decision_metadata",
                                    {},
                                )
                                or {}
                            )
                            if (
                                next_vp == STOP_CANDIDATE
                                or next_vp not in selector_observe_dict
                                or test_decision_metadata.get("fallback_used")
                            ):
                                nav_logger.info(
                                    "Stop rejection fallback needs ranked movement candidate; invalid result {}".format(
                                        next_vp
                                    )
                                )
                                fallback_results = rank_movement_fallback(
                                    "stop_rejected_fallback",
                                    current_step,
                                    selector_observe_dict,
                                    visual_evidence_results,
                                    instruction,
                                    actions,
                                    landmarks,
                                    source_stage="stop_rejected",
                                    reason="stop_rejected_no_valid_movement",
                                )
                                fallback_vp = fallback_results.get(
                                    "selected_candidate"
                                )
                                fallback_results[
                                    "invalid_candidate"
                                ] = next_vp
                                fallback_results[
                                    "decision_test_metadata"
                                ] = test_decision_metadata
                                if fallback_vp in selector_observe_dict:
                                    next_vp = fallback_vp
                                    thought = selector_observe_dict[fallback_vp]
                            else:
                                fallback_results = {
                                    "fallback_strategy": "decision_test_non_stop",
                                    "fallback_reason": "stop_rejected_non_stop_candidate",
                                    "source_stage": "stop_rejected",
                                    "selected_candidate": next_vp,
                                    "available_candidates": [
                                        str(key)
                                        for key in selector_observe_dict.keys()
                                    ],
                                    "decision_test_metadata": (
                                        test_decision_metadata
                                    ),
                                }
                            latest_failure_signal = run_harness_tool(
                                "failure_type_diagnostic",
                                current_step,
                                {},
                                failure_diagnostic.diagnose
                                if failure_diagnostic is not None
                                else None,
                                "stop_rejected",
                                current_step,
                                latest_phase_evidence,
                                recent_distance_gains,
                                fallback_results,
                                selector_verifier_results,
                                {
                                    "predictions": predictions,
                                    "next_vp": next_vp,
                                    "source_stage": "selector_stop_gate",
                                },
                            )
                            if failure_diagnostic is not None:
                                log_u_event(
                                    "failure_type_diagnostic",
                                    current_episode_id,
                                    current_step,
                                    latest_failure_signal,
                                )
                            recovered_candidate, recovered_thought, recovery_results = (
                                apply_failure_recovery(
                                    "stop_rejected",
                                    "stop_rejected_fallback",
                                    selector_observe_dict,
                                    fallback_results,
                                    latest_failure_signal,
                                    next_vp,
                                    thought,
                                )
                            )
                            fallback_results["recovery_results"] = recovery_results
                            if recovered_candidate in selector_observe_dict:
                                next_vp = recovered_candidate
                                thought = recovered_thought
                                if (
                                    isinstance(recovery_results, dict)
                                    and recovery_results.get("applied_to_action")
                                ):
                                    fallback_results[
                                        "selected_candidate_after_recovery"
                                    ] = recovered_candidate
                            log_fallback_event(
                                "stop_rejected_fallback",
                                current_episode_id,
                                current_step,
                                fallback_results,
                            )
                            write_navigation_record(
                                "stop_rejected",
                                episode_id=current_episode_id,
                                step=current_step,
                                reason=selector_stop_rejection_reason,
                                fallback_candidate=next_vp,
                            )
                    write_navigation_record(
                        "selector_final",
                        episode_id=current_episode_id,
                        step=current_step,
                        selected_candidate=next_vp,
                        thought=thought,
                        error_number=error_number,
                        selector_context_applied=(
                            multimodal_selector_context_results.get(
                                "applied", False
                            )
                            if isinstance(
                                multimodal_selector_context_results, dict
                            )
                            else False
                        ),
                    )
                    if harness_enabled and harness_logger is not None:
                        final_context = run_harness_tool(
                            "context_builder_final",
                            current_step,
                            {},
                            context_builder.build_diagnostic
                            if context_builder is not None
                            else None,
                            candidates,
                            geometry_results,
                            grounding_results,
                            memory_results,
                            next_vp,
                        )
                        if context_builder is not None:
                            harness_logger.log_event(
                                "diagnostic_context_final",
                                current_step,
                                final_context,
                            )
                        harness_logger.log_event(
                            "selector_final",
                            current_step,
                            {
                                "selected_candidate": next_vp,
                                "thought": thought,
                                "error_number": error_number,
                                "selector_context_applied": (
                                    multimodal_selector_context_results.get(
                                        "applied", False
                                    )
                                    if isinstance(
                                        multimodal_selector_context_results, dict
                                    )
                                    else False
                                ),
                            },
                        )
           
            try:
                termination_reasons = [None for _ in range(envs.num_envs)]
                env_actions = []
                if stop_flag:
                    env_actions.append({"action": {"action": 0, "action_args": None}})
                else:
                    if next_vp not in radius_dict or next_vp not in distance_dict:
                        fallback_results = rank_movement_fallback(
                            "selector_fallback",
                            current_step,
                            observe_dict,
                            visual_evidence_results,
                            instruction,
                            actions,
                            landmarks,
                            source_stage="env_action_validation",
                            reason="selected_candidate_not_in_action_space",
                        )
                        fallback_vp = fallback_results.get("selected_candidate")
                        fallback_results["invalid_candidate"] = next_vp
                        nav_logger.info(
                            "Invalid selected candidate {}; ranked fallback to {}".format(
                                next_vp,
                                fallback_vp,
                            )
                        )
                        log_fallback_event(
                            "selector_fallback",
                            current_episode_id,
                            current_step,
                            fallback_results,
                        )
                        if fallback_vp in radius_dict and fallback_vp in distance_dict:
                            next_vp = fallback_vp
                        elif radius_dict:
                            fallback_vp = list(radius_dict.keys())[0]
                            last_resort_payload = {
                                "fallback_strategy": "action_space_order_last_resort",
                                "fallback_reason": "ranked_fallback_not_in_action_space",
                                "source_stage": "env_action_validation",
                                "invalid_candidate": next_vp,
                                "selected_candidate": fallback_vp,
                                "available_candidates": [
                                    str(key) for key in radius_dict.keys()
                                ],
                            }
                            log_fallback_event(
                                "selector_fallback_last_resort",
                                current_episode_id,
                                current_step,
                                last_resort_payload,
                            )
                            next_vp = fallback_vp
                        else:
                            no_action_payload = {
                                "fallback_strategy": "stop_last_resort",
                                "fallback_reason": "empty_action_space",
                                "source_stage": "env_action_validation",
                                "invalid_candidate": next_vp,
                                "selected_candidate": STOP_CANDIDATE,
                                "available_candidates": [],
                            }
                            log_fallback_event(
                                "selector_fallback_last_resort",
                                current_episode_id,
                                current_step,
                                no_action_payload,
                            )
                            stop_flag = True
                            stop_reason = (
                                "No movement candidates remained after fallback."
                            )
                            next_vp = STOP_CANDIDATE
                    if stop_flag:
                        env_actions.append(
                            {"action": {"action": 0, "action_args": None}}
                        )
                    else:
                        env_actions.append({'action':
                            {'action': 4,
                            'action_args':{
                                'angle': radius_dict[next_vp],
                                'distance': distance_dict[next_vp],
                            }}})

                nav_logger.info(f"The final env action: {env_actions}")
                decision_audit_payload = build_action_decision_audit(
                    next_vp,
                    stop_flag,
                )
                log_u_event(
                    "decision_audit",
                    current_episode_id,
                    current_step,
                    decision_audit_payload,
                )
                write_navigation_record(
                    "action_pre_step",
                    episode_id=current_episode_id,
                    step=current_step,
                    selected_candidate=next_vp,
                    env_action=env_actions[0],
                    stop_reason=stop_reason if stop_flag else None,
                )
                if harness_enabled and harness_logger is not None:
                    harness_logger.log_event(
                        "action_pre_step",
                        current_step,
                        {
                            "selected_candidate": next_vp,
                            "env_action": env_actions[0],
                            "stop_reason": stop_reason if stop_flag else None,
                        },
                    )
                outputs = envs.step(env_actions)
                step_output_summary = summarize_step_outputs(outputs)
                write_navigation_record(
                    "action_post_step",
                    episode_id=current_episode_id,
                    step=current_step,
                    selected_candidate=next_vp,
                    step_outputs=step_output_summary,
                    stop_reason=stop_reason if stop_flag else None,
                )
                if harness_enabled and harness_logger is not None:
                    harness_logger.log_event(
                        "action_post_step",
                        current_step,
                        {
                            "selected_candidate": next_vp,
                            "step_outputs": step_output_summary,
                            "stop_reason": stop_reason if stop_flag else None,
                        },
                    )
                selected_gain = None
                if step_output_summary:
                    selected_gain = step_output_summary[0].get(
                        "distance_gain_selected"
                    )
                    try:
                        _d = step_output_summary[0].get("distance_to_goal")
                        if _d is not None:
                            latest_goal_dist = float(_d)
                    except (TypeError, ValueError):
                        pass
                try:
                    if selected_gain is not None:
                        recent_distance_gains.append(float(selected_gain))
                        recent_distance_gains = recent_distance_gains[-3:]
                except (TypeError, ValueError):
                    pass
                log_u_event(
                    "post_action_progress",
                    current_episode_id,
                    current_step,
                    {
                        "selected_candidate": next_vp,
                        "distance_gain_selected": selected_gain,
                        "recent_distance_gains": recent_distance_gains,
                        "phase": latest_phase_evidence.get("phase")
                        if isinstance(latest_phase_evidence, dict)
                        else None,
                        "failure_type": latest_failure_signal.get("failure_type")
                        if isinstance(latest_failure_signal, dict)
                        else None,
                    },
                )

                if not stop_flag:
                    curr_observe = observe_dict[next_vp]
                    nav_logger.info("========== save history ==========")
                    nav_history = navigator.save_history(nav_logger, current_step, next_vp, thought, curr_observe, nav_history)
                    write_navigation_record(
                        "history_saved",
                        episode_id=current_episode_id,
                        step=current_step,
                        selected_candidate=next_vp,
                        thought=thought,
                        current_observation=curr_observe,
                        nav_history=nav_history,
                    )

                observations, _, dones, infos = [list(x) for x in zip(*outputs)]
                termination_reasons = [
                    "environment_done" if done else None for done in dones
                ]
                if stop_flag:
                    termination_reasons = [
                        "stop_requested" if done else None for done in dones
                    ]
                    if not dones[0]:
                        termination_reasons[0] = "stop_requested"
                        dones[0] = True
                else:
                    instruction, images_list = self.generate_input(observations[-1])
                    error_number = 0 
                    # finish navigation
                    if current_step >= step_length:
                        if not dones[0]:
                            termination_reasons[0] = "step_length_limit"
                        dones[0] = True
                    else:
                        for j, ob in enumerate(observations):
                            envs.call_at(j, 
                                'change_current_path',
                                {'new_path': ob.pop('positions'),
                                'collisions': ob.pop('collisions')}
                            )
                
                not_done_masks = torch.tensor(
                    [[0] if done else [1] for done in dones],
                    dtype=torch.uint8, device=self.device)
                
                for i in range(envs.num_envs):
                    
                    if not dones[i]:
                        continue
                    
                    episode_end_step = current_step
                    current_step = 0
                    nav_history = []
                    info = infos[i]
                    metric = {}
                    metric['steps_taken'] = info['steps_taken']
                    ep_id = str(envs.current_episodes()[i].episode_id)
                    gt_path = np.array(self.gt_data[ep_id]['locations']).astype(float)
                    if 'current_path' in envs.current_episodes()[i].info.keys():
                        positions_ = np.array(envs.current_episodes()[i].info['current_path']).astype(float)
                        collisions_ = np.array(envs.current_episodes()[i].info['collisions'])
                        assert collisions_.shape[0] == positions_.shape[0] - 1
                    else:
                        positions_ = np.array(dis_to_con(np.array(info['position']['position']))).astype(float)
                        collisions_ = np.zeros(max(positions_.shape[0] - 1, 0), dtype=float)
                    distance = np.array(info['position']['distance']).astype(float)
                    metric['distance_to_goal'] = distance[-1]
                    metric['success'] = 1. if distance[-1] <= 3. else 0.
                    metric['oracle_success'] = 1. if (distance <= 3.).any() else 0.
                    metric['path_length'] = np.linalg.norm(positions_[1:] - positions_[:-1],axis=1).sum()
                    metric['collisions'] = collisions_.mean() if collisions_.size > 0 else 0.0
                    gt_length = distance[0]
                    metric['spl'] = metric['success']*gt_length/max(gt_length,metric['path_length'])

                    act_con_path = positions_
                    gt_con_path = np.array(gt_path).astype(float)
                    dtw_distance = fastdtw(act_con_path, gt_con_path, dist=NDTW.euclidean_distance)[0]
                    nDTW = np.exp(-dtw_distance / (len(gt_con_path) * config.TASK_CONFIG.TASK.SUCCESS_DISTANCE))

                    metric['ndtw'] = nDTW
                    stats_episodes[current_episodes[i].episode_id] = metric
                    step_error_count = 0
                    last_errored_step = -1
                    try:
                        os.makedirs(config.RESULTS_DIR, exist_ok=True)
                        ckpt_path = os.path.join(config.RESULTS_DIR, "checkpoint_stats_episodes.json")
                        ckpt_tmp = ckpt_path + ".tmp"
                        with open(ckpt_tmp, "w") as _ckpt_f:
                            json.dump(stats_episodes, _ckpt_f, indent=4)
                        os.replace(ckpt_tmp, ckpt_path)
                    except Exception as _ckpt_exc:
                        nav_logger.info(f"Checkpoint write failed: {_ckpt_exc}")
                    write_navigation_record(
                        "episode_termination",
                        episode_id=ep_id,
                        step=episode_end_step,
                        reason=termination_reasons[i],
                        step_length=step_length,
                    )
                    if harness_enabled and harness_logger is not None:
                        harness_logger.log_event(
                            "episode_termination",
                            episode_end_step,
                            {
                                "episode_id": ep_id,
                                "reason": termination_reasons[i],
                                "step_length": step_length,
                            },
                        )
                    write_navigation_record(
                        "episode_end",
                        episode_id=ep_id,
                        step=episode_end_step,
                        metrics=metric,
                        termination_reason=termination_reasons[i],
                        evaluated_episodes=len(stats_episodes),
                        total_episodes=episodes_to_eval,
                    )
                    log_u_event(
                        "u_series_episode_end",
                        ep_id,
                        episode_end_step,
                        {
                            "decision_effect_unit": u_decision_unit,
                            "recent_distance_gains": recent_distance_gains,
                            "latest_phase": latest_phase_evidence.get("phase")
                            if isinstance(latest_phase_evidence, dict)
                            else None,
                            "latest_failure_type": latest_failure_signal.get(
                                "failure_type"
                            )
                            if isinstance(latest_failure_signal, dict)
                            else None,
                            "recovery_budget_remaining": recovery_budget_remaining,
                        },
                    )
                    active_navigation_episode_id = None
                    if harness_enabled and harness_logger is not None:
                        if oracle_metrics_enabled:
                            harness_logger.log_event(
                                "oracle_metrics",
                                episode_end_step,
                                {
                                    "distance_gain_selected": selected_distance_gain(info),
                                },
                            )
                        harness_logger.end_episode(
                            ep_id,
                            metric,
                            step_id=episode_end_step,
                        )
                        active_harness_episode_id = None

                    observations[i] = envs.reset_at(i)[0]
                    instruction, images_list = self.generate_input(observations[i])
                    
                    if config.use_pbar:
                        pbar.update()
                    else:
                        logger.info(
                            log_str.format(
                                evaluated=len(stats_episodes),
                                total=episodes_to_eval,
                                time=round(time.time() - start_time),
                            )
                        )
                observations = extract_instruction_tokens(
                    observations,
                    self.config.TASK_CONFIG.TASK.INSTRUCTION_SENSOR_UUID,
                )
                batch = batch_obs(observations, self.device)
                batch = apply_obs_transforms_batch(batch, obs_transforms)   
                
                envs_to_pause = []
                next_episodes = envs.current_episodes()

                for i in range(envs.num_envs):
                    if next_episodes[i].episode_id in stats_episodes:
                        envs_to_pause.append(i)

                headings = torch.tensor(headings)
                (
                    envs,
                    not_done_masks,
                    headings,  
                    batch,
                    rgb_frames,
                ) = self._pause_envs(
                    envs_to_pause,
                    envs,
                    not_done_masks,
                    headings,
                    batch,
                    rgb_frames,
                )
                headings = headings.tolist()
            except Exception as e:
                nav_logger.info(f"Error in next action prediction: {e}")
                if harness_enabled and harness_logger is not None:
                    harness_logger.log_tool_failure(
                        "navigation_loop",
                        locals().get("current_step", 0),
                        e,
                    )
                write_navigation_record(
                    "navigation_error",
                    episode_id=locals().get("current_episode_id"),
                    step=locals().get("current_step"),
                    error=repr(e),
                )
                failed_step = locals().get("current_step", 0)
                if failed_step == last_errored_step:
                    step_error_count += 1
                else:
                    last_errored_step = failed_step
                    step_error_count = 1
                if step_error_count >= 3:
                    nav_logger.info(
                        f"Step {failed_step} failed {step_error_count} consecutive times; "
                        "skipping retry to prevent infinite loop."
                    )
                    step_error_count = 0
                    last_errored_step = -1
                else:
                    current_step -= 1
        envs.close()
        if config.use_pbar:
            pbar.close()
        if self.world_size > 1:
            distr.barrier()
        aggregated_stats = {}
        num_episodes = len(stats_episodes)
        for stat_key in next(iter(stats_episodes.values())).keys():
            aggregated_stats[stat_key] = (
                sum(v[stat_key] for v in stats_episodes.values())
                / num_episodes
            )
        total = torch.tensor(num_episodes).cuda()
        if self.world_size > 1:
            dist.reduce(total,dst=0)
        total = total.item()

        if self.world_size > 1:
            logger.info(
                f"rank {self.local_rank}'s {num_episodes}-episode results: {aggregated_stats}")
            for k,v in aggregated_stats.items():
                v = torch.tensor(v*num_episodes).cuda()
                cat_v = gather_list_and_concat(v,self.world_size)
                v = (sum(cat_v)/total).item()
                aggregated_stats[k] = v

        split = config.TASK_CONFIG.DATASET.SPLIT
        fname = os.path.join(
            config.RESULTS_DIR,
            f"stats_ep_ckpt_{split}_r{self.local_rank}_w{self.world_size}.json",
        )
        with open(fname, "w") as f:
            json.dump(stats_episodes, f, indent=4)

        if self.local_rank < 1:
            if config.EVAL.SAVE_RESULTS:
                fname = os.path.join(
                    config.RESULTS_DIR,
                    f"stats_ckpt_{split}.json",
                )
                with open(fname, "w") as f:
                    json.dump(aggregated_stats, f, indent=4)

            logger.info(f"Episodes evaluated: {total}")
            for k, v in aggregated_stats.items():
                logger.info(f"Average episode {k}: {v:.6f}")
        
    def collect_val_traj(self):
        trajectories = defaultdict(list)
        split = self.config.TASK_CONFIG.DATASET.SPLIT
        with gzip.open(
            self.config.TASK_CONFIG.TASK.NDTW.GT_PATH.format(
                split=split)
        ) as f:
            gt_data = json.load(f)
        self.gt_data = gt_data
        trajectories = gt_data
        self.trajectories = gt_data
        trajectories = list(trajectories.keys())[self.config.local_rank::self.config.GPU_NUMBERS]
        return trajectories
        
    def eval(self) -> None:
        r"""Main method of trainer evaluation. 

        Returns:
            None
        """
        self.device = (
            torch.device("cuda", self.config.TORCH_GPU_ID)
            if torch.cuda.is_available()
            else torch.device("cpu")
        )

        if "tensorboard" in self.config.VIDEO_OPTION:
            assert (
                len(self.config.TENSORBOARD_DIR) > 0
            ), "Must specify a tensorboard directory for video display"
            os.makedirs(self.config.TENSORBOARD_DIR, exist_ok=True)
        if "disk" in self.config.VIDEO_OPTION:
            assert (
                len(self.config.VIDEO_DIR) > 0
            ), "Must specify a directory for storing videos on disk"

        world_size = self.config.GPU_NUMBERS
        self.world_size = world_size
        self.local_rank = self.config.local_rank

        self.config.defrost()
        self.config.TASK_CONFIG.DATASET.ROLES = ["guide"]
        self.config.TASK_CONFIG.TASK.MEASUREMENTS = ['POSITION',
                                                     'STEPS_TAKEN',
                                                     ]
        if 'HIGHTOLOW' in self.config.TASK_CONFIG.TASK.POSSIBLE_ACTIONS:
            idx = self.config.TASK_CONFIG.TASK.POSSIBLE_ACTIONS.index('HIGHTOLOW')
            self.config.TASK_CONFIG.TASK.POSSIBLE_ACTIONS[idx] = 'HIGHTOLOWEVAL'
        self.config.TASK_CONFIG.DATASET.LANGUAGES = self.config.EVAL.LANGUAGES
        self.config.TASK_CONFIG.DATASET.SPLIT = self.config.EVAL.SPLIT
        self.config.TASK_CONFIG.TASK.NDTW.SPLIT = self.config.EVAL.SPLIT
        self.config.TASK_CONFIG.TASK.SDTW.SPLIT = self.config.EVAL.SPLIT
        self.config.use_pbar = not is_slurm_batch_job()
        if 'rxr' in self.config.BASE_TASK_CONFIG_PATH:
            self.config.EVAL.trajectories_file = \
                self.config.EVAL.trajectories_file[:-8] + '_w' + \
                str(self.world_size) + '_r' + str(self.local_rank) + '.json.gz'
        
        # if choosing image
        resize_config = self.config.RL.POLICY.OBS_TRANSFORMS.RESIZER_PER_SENSOR.SIZES
        config = self.config.TASK_CONFIG
        camera_orientations = get_camera_orientations(12)

        # sensor_uuids = []
        for sensor_type in ["RGB", "DEPTH"]:
            resizer_size = dict(resize_config)[sensor_type.lower()]
            sensor = getattr(config.SIMULATOR, f"{sensor_type}_SENSOR")
            for action, orient in camera_orientations.items():
                camera_template = f"{sensor_type}_{action}"
                camera_config = deepcopy(sensor)
                camera_config.ORIENTATION = camera_orientations[action]
                camera_config.UUID = camera_template.lower()
                # sensor_uuids.append(camera_config.UUID)
                setattr(config.SIMULATOR, camera_template, camera_config)
                config.SIMULATOR.AGENT_0.SENSORS.append(camera_template)
                resize_config.append((camera_template.lower(), resizer_size))
        self.config.RL.POLICY.OBS_TRANSFORMS.RESIZER_PER_SENSOR.SIZES = resize_config
        self.config.TASK_CONFIG = config
        self.config.SENSORS = config.SIMULATOR.AGENT_0.SENSORS
        
        self.config.freeze()
        torch.cuda.set_device(self.device)
        if world_size > 1:
            distr.init_process_group(backend='nccl', init_method='env://')
            self.device = self.config.TORCH_GPU_IDS[self.local_rank]
            torch.cuda.set_device(self.device)
            self.config.defrost()
            self.config.TORCH_GPU_ID = self.config.TORCH_GPU_IDS[self.local_rank]
            self.config.freeze()
            
        self.traj = self.collect_val_traj()
        self._eval_llm()
