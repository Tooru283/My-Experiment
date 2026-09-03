import json
import re
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
    build_anchor_chain,
    ConstraintQueueLocator,
    LandmarkPool,
    TerminalGate,
    gate_stop_request,
    build_route_state,
    ActionCommand,
    DecisionRecord,
    StopProposal,
    StopEvidenceItem,
    StopCoordinator,
    ActionCompiler,
    ResolvedAction,
    build_m3_2_evidence_items,
    build_action_receipt,
    build_observation_frame,
    RouteStateReducer,
    ACNL1ProgressProvider,
    direction_aligned_terminal_result,
    weak_generic_close_is_confirmed,
    TerminalEvidenceMemory,
    TerminalInstanceTracker,
    is_weak_generic_terminal,
    terminal_target_is_confirmed,
)
from vlnce_baselines.common.opennav_ext.landmark_matching import (
    final_landmark_terms,
)
from vlnce_baselines.common.opennav_ext.harness_config import (
    arrival_gate_config,
    arrival_gate_enabled,
    proactive_stop_gate_config,
    proactive_stop_gate_enabled,
    decision_effect_enabled,
    fail_open_enabled,
    get_trace_dir,
    harness_logging_enabled,
    module_enabled,
    module_log_only,
    u_decision_effect_unit,
    u_module_enabled,
    u_module_log_only,
    u_series_enabled,
    validate_a1_harness_config,
)
from vlnce_baselines.common.opennav_ext.agent_state import build_candidate_records
from vlnce_baselines.common.opennav_ext.oracle_metrics import (
    selected_distance_gain,
    summarize_step_outputs,
)
from vlnce_baselines.common.opennav_ext.decision_audit import build_decision_audit
from vlnce_baselines.common.opennav_ext import candidate_prior
from vlnce_baselines.common.opennav_ext.backtrack_policy import (
    BacktrackPolicy,
    MOVE_BACK_CANDIDATE,
)
import numpy as np
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


# ----------------------------------------------------------------------------
# v2 DEPTH STOP VETO (default OFF) helpers.
#
# Motivation (20260705): the >=2-step persistence guard cannot stop a VLM that
# is *confidently and consistently* wrong about arrival (ep377: conf-1.00 "final
# target visible" @ 5.42 m for >=2 steps). A metric depth reading of the claimed
# target direction is an observable veto that geometry can enforce where the VLM
# cannot self-correct. All VLN-CE methods legally use the depth sensor -> zero
# oracle exposure (unlike info["position"]["distance"] which is GT geodesic).
#
# HONEST SCOPE (do NOT overclaim in paper/docs): this reads the *center region
# of the current forward view*, NOT "the distance to the target". Two blind
# spots, accepted and to be quantified from the trace (raw depth is logged so the
# A/B run can measure hit/false-veto rate):
#   (1) target off-center: a target at the frame edge is not measured; the center
#       may read a nearer wall (false veto) or a farther opening (missed veto).
#   (2) line-of-sight euclidean vs success's geodesic: a 2.8 m euclidean reading
#       through a wall that is 5 m geodesic would be (wrongly) let through.
# So this vetoes the "staring at a distant open area and calling it arrival" lie
# (ep377), it does NOT catch every confident lie. Wording must stay specific.
# ----------------------------------------------------------------------------
def _current_view_depth_array(observations):
    """Raw current-view (front) depth map from the sim observation, or None.

    Prefers the exact front sensor uuid ('depth'); falls back to any '*depth*'
    key. Returns the raw numpy array BEFORE generate_input's per-frame min-max
    renormalization (which destroys metric meaning) -- so metric meters are
    recoverable. Fail-open: any structural surprise returns None (=> no veto).
    """
    try:
        obs0 = observations[0] if isinstance(observations, (list, tuple)) else observations
        if not isinstance(obs0, dict):
            return None
        if "depth" in obs0:
            return obs0["depth"]
        for key in obs0.keys():
            if "depth" in str(key).lower():
                return obs0[key]
    except Exception:
        return None
    return None


def _direction_depth_array(observations, direction_id):
    """Return the raw metric-depth frame for an observed panorama direction."""
    try:
        obs0 = observations[0] if isinstance(observations, (list, tuple)) else observations
        if not isinstance(obs0, dict):
            return None
        index = int(direction_id)
        depth_keys = [
            key for key in obs0.keys()
            if "depth" in str(key).lower() and not str(key).startswith("tilt_")
        ]
        if index < 0 or index >= len(depth_keys):
            return None
        return obs0[depth_keys[index]]
    except (TypeError, ValueError, IndexError, KeyError):
        return None


def _metric_depth_center_median(depth_array, min_depth, max_depth, normalize, center_frac):
    """Median metric depth (m) of the center patch, or None if unusable (fail-open).

    depth_array: HxW or HxWx1. When ``normalize`` (habitat NORMALIZE_DEPTH), values
    are in [0,1] and metric = v*(max-min)+min; otherwise values are already meters.
    Zero/near-zero pixels are dropped (invalid depth returns). Median is robust to
    the holes typical of depth sensors.
    """
    try:
        arr = np.asarray(depth_array, dtype=np.float32)
        if arr.ndim == 3 and arr.shape[-1] == 1:
            arr = arr[..., 0]
        if arr.ndim != 2 or arr.size == 0:
            return None
        h, w = arr.shape
        cf = float(center_frac)
        half_h = max(1, int(h * cf / 2.0))
        half_w = max(1, int(w * cf / 2.0))
        cy, cx = h // 2, w // 2
        patch = arr[cy - half_h:cy + half_h, cx - half_w:cx + half_w]
        vals = patch[patch > 1e-6]
        if vals.size == 0:
            return None
        median_val = float(np.median(vals))
        if normalize:
            return median_val * (float(max_depth) - float(min_depth)) + float(min_depth)
        return median_val
    except Exception:
        return None


def _metric_depth_near_surface(
    depth_array, min_depth, max_depth, normalize, center_frac=0.70,
    percentile=20.0,
):
    """Estimate distance to an opening's frame, not through its empty center."""
    try:
        arr = np.asarray(depth_array, dtype=np.float32)
        if arr.ndim == 3 and arr.shape[-1] == 1:
            arr = arr[..., 0]
        if arr.ndim != 2 or arr.size == 0:
            return None
        h, w = arr.shape
        half_h = max(1, int(h * float(center_frac) / 2.0))
        half_w = max(1, int(w * float(center_frac) / 2.0))
        cy, cx = h // 2, w // 2
        patch = arr[cy - half_h:cy + half_h, cx - half_w:cx + half_w]
        vals = patch[patch > 1e-6]
        if vals.size == 0:
            return None
        value = float(np.percentile(vals, float(percentile)))
        if normalize:
            return value * (float(max_depth) - float(min_depth)) + float(min_depth)
        return value
    except Exception:
        return None


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
            # The endgame tilt sensors must not enter this sweep: rgb_index /
            # depth_index are positional and construct_image_dicts maps 1..12 to
            # headings, so counting a 13th view here would rotate every direction.
            # The tilt view is read directly from observations at the stop gate.
            if str(key).startswith('tilt_'):
                continue
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
        memory_alert_decision_effect = False
        # ACN (20260805): L0/L1/M2/L4. All LOG_ONLY -- they write trace and change nothing.
        anchor_chain_enabled = False
        progress_locator = None
        progress_locator_decision_effect = False
        landmark_pool = None
        terminal_gate = None
        terminal_gate_decision_effect = False
        context_builder = None
        route_state_reducer = None
        progress_provider = None
        stop_coordinator = StopCoordinator({
            "required_evaluators": ("v2", "progress"),
            "opposing_evaluators": ("v2", "progress"),
            "policy_version": "coordinated_v2.independent_v2_progress",
        })
        action_compiler = ActionCompiler()
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
            harness_logger.log_event(
                "run_metadata",
                0,
                {
                    "config_snapshot": os.environ.get("OPENNAV_CONFIG_SNAPSHOT"),
                    "resolved_config_sha256": os.environ.get("OPENNAV_CONFIG_SHA256"),
                    "exp_config_sha256": os.environ.get("OPENNAV_EXP_CONFIG_SHA256"),
                    "llm": config.LLM,
                    "visual_evidence_model": os.environ.get("OPENNAV_LLM_MODEL"),
                    "visual_evidence_base_url": os.environ.get("OPENNAV_LLM_BASE_URL"),
                    "trace_dir": get_trace_dir(config),
                },
            )

            if module_enabled(config, "GEOMETRY_QUERY"):
                geometry_query = GeometryQueryLogger()
            if module_enabled(config, "GROUNDER_DIAGNOSTIC"):
                grounder_diagnostic = GrounderDiagnostic()
            if module_enabled(config, "VISUAL_EVIDENCE"):
                visual_evidence_config = config.OPENNAV_HARNESS.VISUAL_EVIDENCE
                visual_evidence = VisualEvidenceLogger(
                    base_url=os.environ.get("OPENNAV_LLM_BASE_URL", visual_evidence_config.BASE_URL),
                    model=os.environ.get("OPENNAV_LLM_MODEL", visual_evidence_config.MODEL),
                    api_key=(
                        os.environ.get("OPENNAV_LLM_API_KEY")
                        or os.environ.get("DASHSCOPE_API_KEY")
                        or getattr(config, "API_KEY", "not-needed")
                    ),
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
                _mem_cfg = config.OPENNAV_HARNESS.MEMORY_DIAGNOSTIC
                memory_diagnostic = VisualGraphMemoryDiagnostic(
                    revisit_radius=float(
                        getattr(_mem_cfg, "REVISIT_RADIUS_M", 1.0)
                    ),
                    merge_radius=float(getattr(_mem_cfg, "MERGE_RADIUS_M", 0.8)),
                    loop_alert_threshold=int(
                        getattr(_mem_cfg, "LOOP_ALERT_THRESHOLD", 3)
                    ),
                    vertical_alert_m=float(
                        getattr(_mem_cfg, "VERTICAL_ALERT_M", 0.3)
                    ),
                    enable_vertical_alert=bool(
                        getattr(_mem_cfg, "ENABLE_VERTICAL_ALERT", False)
                    ),
                    floor_height_m=float(getattr(_mem_cfg, "FLOOR_HEIGHT_M", 1.5)),
                )
                # C5: decision effect = append G_topo's alert to the navigator prompt.
                memory_alert_decision_effect = (
                    decision_effect_enabled(config)
                    and not module_log_only(config, "MEMORY_DIAGNOSTIC")
                )
            # ---- ACN L0 / L1 / M2 / L4 (20260805) ----
            anchor_chain_enabled = module_enabled(config, "ANCHOR_CHAIN")
            if module_enabled(config, "PROGRESS_LOCATOR"):
                _pl_cfg = config.OPENNAV_HARNESS.PROGRESS_LOCATOR
                progress_locator = ConstraintQueueLocator(
                    object_radius_m=float(getattr(_pl_cfg, "OBJECT_RADIUS_M", 3.0)),
                    direction_window=int(getattr(_pl_cfg, "DIRECTION_WINDOW", 2)),
                    turn_deg=float(getattr(_pl_cfg, "TURN_DEG", 35.0)),
                    around_deg=float(getattr(_pl_cfg, "AROUND_DEG", 120.0)),
                    forward_m=float(getattr(_pl_cfg, "FORWARD_M", 1.0)),
                    heading_sign=float(getattr(_pl_cfg, "HEADING_SIGN", -1.0)),
                    enable_location=bool(getattr(_pl_cfg, "ENABLE_LOCATION", True)),
                    location_dominance=float(
                        getattr(_pl_cfg, "LOCATION_DOMINANCE", 0.5)
                    ),
                    vacuous_unknown=bool(
                        getattr(_pl_cfg, "VACUOUS_UNKNOWN", True)
                    ),
                )
                # decision effect = the queue's text replaces the LLM `estimation` string
                progress_locator_decision_effect = (
                    decision_effect_enabled(config)
                    and not module_log_only(config, "PROGRESS_LOCATOR")
                )
            if module_enabled(config, "LANDMARK_POOL"):
                _lp_cfg = config.OPENNAV_HARNESS.LANDMARK_POOL
                landmark_pool = LandmarkPool(
                    arrival_radius_m=float(getattr(_lp_cfg, "ARRIVAL_RADIUS_M", 3.0)),
                    merge_radius_m=float(getattr(_lp_cfg, "MERGE_RADIUS_M", 2.0)),
                    drop_stopwords=bool(getattr(_lp_cfg, "DROP_STOPWORDS", True)),
                    restrict_to_vocabulary=bool(
                        getattr(_lp_cfg, "RESTRICT_TO_VOCABULARY", True)
                    ),
                )
            if module_enabled(config, "TERMINAL_GATE"):
                _tg_cfg = config.OPENNAV_HARNESS.TERMINAL_GATE
                terminal_gate = TerminalGate(
                    require_chain_complete=bool(
                        getattr(_tg_cfg, "REQUIRE_CHAIN_COMPLETE", True)
                    ),
                    require_all_verified=bool(
                        getattr(_tg_cfg, "REQUIRE_ALL_VERIFIED", True)
                    ),
                    require_landmark_evidence=bool(
                        getattr(_tg_cfg, "REQUIRE_LANDMARK_EVIDENCE", True)
                    ),
                )
                terminal_gate_decision_effect = (
                    decision_effect_enabled(config)
                    and not module_log_only(config, "TERMINAL_GATE")
                )
            if module_enabled(config, "CONTEXT_BUILDER"):
                context_builder = ContextBuilder()
            route_state_reducer = RouteStateReducer(
                anchor_chain_enabled=anchor_chain_enabled,
                progress_locator=progress_locator,
                spatial_graph=memory_diagnostic,
                landmark_pool=landmark_pool,
                terminal_gate=terminal_gate,
            )
            progress_provider = ACNL1ProgressProvider(route_state_reducer)
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
            # v2 DEPTH STOP VETO (default OFF -> byte-identical baseline). When on,
            # a STOP-enabling gate additionally requires the metric depth of the
            # claimed target direction to be within MAX_TARGET_DIST_M. Read metric
            # scale from the sim depth sensor config (habitat defaults 0..10 m,
            # NORMALIZE_DEPTH True). See _metric_depth_center_median.
            depth_veto_enabled = False
            depth_veto_dist = 3.0
            depth_veto_center_frac = 0.25
            _sim_cfg = getattr(config.TASK_CONFIG, "SIMULATOR", None)
            _depth_sensor_cfg = getattr(_sim_cfg, "DEPTH_SENSOR", None) if _sim_cfg is not None else None
            depth_veto_min = float(getattr(_depth_sensor_cfg, "MIN_DEPTH", 0.0)) if _depth_sensor_cfg is not None else 0.0
            depth_veto_max = float(getattr(_depth_sensor_cfg, "MAX_DEPTH", 10.0)) if _depth_sensor_cfg is not None else 10.0
            depth_veto_normalize = bool(getattr(_depth_sensor_cfg, "NORMALIZE_DEPTH", True)) if _depth_sensor_cfg is not None else True
            _vtv_cfg = getattr(config.OPENNAV_HARNESS, "VISUAL_TARGET_VERIFIER", None)
            if _vtv_cfg is not None:
                _dveto_cfg = getattr(_vtv_cfg, "DEPTH_STOP_VETO", None)
                if _dveto_cfg is not None:
                    depth_veto_enabled = bool(getattr(_dveto_cfg, "ENABLED", False))
                    depth_veto_dist = float(getattr(_dveto_cfg, "MAX_TARGET_DIST_M", 3.0))
                    depth_veto_center_frac = float(getattr(_dveto_cfg, "CENTER_FRAC", 0.25))

            # ---- Endgame tilt view: config, per-episode state, trigger ----------
            tilt_enabled = False
            tilt_log_only = True
            tilt_pitch_deg = -30.0
            tilt_arm_final_clause = False
            tilt_arm_final_landmark = True
            tilt_arm_step_frac = 0.7
            tilt_max_per_episode = 4
            tilt_min_step_gap = 3
            tilt_always_on_stop = True
            tilt_dedupe_dist = 0.5
            tilt_dedupe_heading = 15.0
            tilt_center_frac = 0.25
            _tv_cfg = getattr(config.OPENNAV_HARNESS, "ENDGAME_TILT_VIEW", None)
            if _tv_cfg is not None:
                tilt_enabled = bool(getattr(_tv_cfg, "ENABLED", False))
                tilt_log_only = bool(getattr(_tv_cfg, "LOG_ONLY", True))
                tilt_pitch_deg = float(getattr(_tv_cfg, "PITCH_DEG", -30.0))
                tilt_arm_final_clause = bool(getattr(_tv_cfg, "ARM_ON_FINAL_CLAUSE", False))
                tilt_arm_final_landmark = bool(getattr(_tv_cfg, "ARM_ON_FINAL_LANDMARK", True))
                tilt_arm_step_frac = float(getattr(_tv_cfg, "ARM_STEP_FRAC", 0.7))
                tilt_max_per_episode = int(getattr(_tv_cfg, "MAX_PER_EPISODE", 4))
                tilt_min_step_gap = int(getattr(_tv_cfg, "MIN_STEP_GAP", 3))
                tilt_always_on_stop = bool(getattr(_tv_cfg, "ALWAYS_ON_STOP", True))
                tilt_dedupe_dist = float(getattr(_tv_cfg, "DEDUPE_DIST_M", 0.5))
                tilt_dedupe_heading = float(getattr(_tv_cfg, "DEDUPE_HEADING_DEG", 15.0))
                tilt_center_frac = float(getattr(_tv_cfg, "CENTER_FRAC", 0.25))
            # ---- Arm C: candidate prior config ---------------------------------
            candidate_prior_enabled = False
            candidate_prior_log_only = True
            candidate_prior_min_gap = 0.10
            _cp_cfg = getattr(config.OPENNAV_HARNESS, "CANDIDATE_PRIOR", None)
            if _cp_cfg is not None:
                candidate_prior_enabled = bool(getattr(_cp_cfg, "ENABLED", False))
                candidate_prior_log_only = bool(getattr(_cp_cfg, "LOG_ONLY", True))
                candidate_prior_min_gap = float(getattr(_cp_cfg, "MIN_PROB_GAP", 0.10))
            # (scores, ranks) from this step's prior; consumed by the override hook
            # after the selector has spoken. Reset each step alongside the candidates.
            latest_candidate_prior = None
            # ACN L1 is the only completion progress provider.
            progress_provider_name = "acn_l1"

            def _candidate_prior_override(selector_vp):
                """Override the selector when the prior is decisively ahead.

                Returns (final_vp, audit_payload). The prior CANNOT be delivered to
                the selector any other way: the prompt's candidate order carries no
                position bias (first-slot pick rate 27.1% vs 25.8% uniform, flat
                across slots), and prompt-side geometry injection was already
                measured net-negative (GEOMETRY_INJECTION=False). So overriding the
                emitted choice is the only channel left.

                Gate = softmax probability gap, not rank. Measured on the fitting run:

                    full override (rank>=1)   60% intervention -> 41.8%
                    gap > 0.10                29% intervention -> 41.5%
                    gap > 0.20                10% intervention -> 39.5%
                    LLM alone                  0% intervention -> 36.7%

                gap>0.10 buys 96% of the benefit for half the intervention, so it is
                the default. Never touches STOP or MOVE_BACK -- those are termination
                decisions the prior has no label for.
                """
                if latest_candidate_prior is None or candidate_prior_log_only:
                    return selector_vp, None
                scores, ranks = latest_candidate_prior
                if not scores or str(selector_vp) not in scores:
                    return selector_vp, None
                try:
                    keys = list(scores)
                    mx = max(scores.values())
                    exp = {k: math.exp(scores[k] - mx) for k in keys}
                    tot = sum(exp.values()) or 1.0
                    prob = {k: exp[k] / tot for k in keys}
                    top = max(prob, key=prob.get)
                    gap = prob[top] - prob[str(selector_vp)]
                    if top == str(selector_vp) or gap <= candidate_prior_min_gap:
                        return selector_vp, None
                    return top, {
                        "selector_choice": str(selector_vp),
                        "prior_choice": top,
                        "prob_gap": gap,
                        "min_prob_gap": candidate_prior_min_gap,
                    }
                except (KeyError, TypeError, ValueError, OverflowError):
                    return selector_vp, None

            def _parse_visual_evidence_candidates(ve_results):
                """{candidate_id: entry} from the visual_evidence raw response.

                The parsed candidate list is not exposed as a field, so this reads
                `raw_response` the same way the offline fit did -- keeping the
                runtime nmatch identical to the one the weights were trained on.
                Returns {} on any parse surprise; the prior then scores nmatch=0,
                which is exactly what the fit saw for unsampled candidates.
                """
                if not isinstance(ve_results, dict):
                    return {}
                try:
                    raw = json.loads(ve_results.get("raw_response") or "{}")
                    return {
                        str(c["candidate_id"]): c
                        for c in raw.get("candidates", [])
                        if isinstance(c, dict) and "candidate_id" in c
                    }
                except (ValueError, TypeError, KeyError):
                    return {}

            # Geometric floor for the tilt reading: a -30deg ray from a 1.25m-high
            # sensor hits flat empty floor at 2.50m, so a "close" tilt depth is only
            # evidence of an object if it is meaningfully below this. Computed from
            # the live sensor height so it stays correct if either value is retuned.
            try:
                _cam_h = float(
                    getattr(config.TASK_CONFIG.SIMULATOR.DEPTH_SENSOR, "POSITION", [0, 1.25, 0])[1]
                )
                tilt_floor_intersect_m = _cam_h / math.sin(math.radians(abs(tilt_pitch_deg)))
            except (AttributeError, IndexError, TypeError, ValueError, ZeroDivisionError):
                tilt_floor_intersect_m = None
            # Per-episode state; reset in the episode-boundary block below.
            tilt_fire_count = 0
            tilt_last_pose = None
            tilt_last_fire_step = None
            tilt_last_reading = None

            def _tilt_endgame_armed():
                """Is the agent in the instruction's final stage? (oracle-free)

                Any one of the enabled signals arms it. Returns (armed, reason).
                final_clause is off by default: measured on ep100 20260719 it is true
                on step 1 in 81/100 episodes, because the LLM's completed_action_count
                is already at action_count-1 before the agent has moved. It is clean of
                oracle but carries no endgame information, and leaving it on spent the
                whole budget in steps 1-3.
                NOT keyed on phase_evidence.phase -- that field branches "recover"
                on recent_distance_gains (simulator geodesic goal-distance deltas =
                GT), so anything conditioned on it inherits an inference-time oracle.
                completed/action_count come from _completed_action_count(estimation,
                actions), which reads only instruction text and the LLM's own recap.
                """
                pe = latest_phase_evidence or {}
                if tilt_arm_final_clause:
                    try:
                        n_actions = int(pe.get("action_count") or 0)
                        n_done = int(pe.get("completed_action_count") or 0)
                        if n_actions > 0 and n_done >= n_actions - 1:
                            return True, "final_clause"
                    except (TypeError, ValueError):
                        pass
                if tilt_arm_final_landmark and recent_ftv_window:
                    # Final target reported visible recently == endgame. Last 3 steps,
                    # matching the E3 carry-forward window already used for stop
                    # evidence; the full 10-slot window would keep the tilt armed for
                    # the rest of the episode after a single sighting.
                    if any(bool(v) for v in recent_ftv_window[-3:]):
                        return True, "final_landmark_seen"
                if tilt_arm_step_frac > 0:
                    try:
                        if current_step >= tilt_arm_step_frac * float(step_length):
                            return True, "step_budget"
                    except (TypeError, ValueError, ZeroDivisionError):
                        pass
                return False, None

            def _tilt_pose_is_new():
                """Rate limit by pose: skip if we already tilted from ~here."""
                if tilt_last_pose is None:
                    return True
                try:
                    px, py, pz, ph = tilt_last_pose
                    cx, cy, cz = positions[0]
                    ch = float(headings[0])
                    moved = float(np.linalg.norm(np.array([cx, cy, cz]) - np.array([px, py, pz])))
                    turned = abs(np.degrees(ch - ph))
                    turned = min(turned, 360.0 - turned)
                    return moved >= tilt_dedupe_dist or turned >= tilt_dedupe_heading
                except Exception:
                    return True

            def _tilt_observe(source_label, force=False):
                """Observe the down-pitched view at an endgame stop gate.

                Fires only when armed AND rate limits allow. Emits an
                `endgame_tilt_view` event carrying the tilt depth center median and
                the SpatialBot/RAM reading of the tilted frame -- the offline channel
                for measuring how many near-field targets the eye-level rig misses.
                Fail-open: any structural surprise logs nothing and changes nothing.
                Returns the tilt reading dict, or None.

                force=True is the committed-stop sample: it bypasses the budget, the
                step gap, the arm test and the pose dedupe, because that one pose is
                the whole point of the instrument and every rate limit above exists
                only to ration the in-episode samples around it.
                """
                nonlocal tilt_fire_count, tilt_last_pose, tilt_last_fire_step
                nonlocal tilt_last_reading
                if not tilt_enabled:
                    return None
                if force and tilt_last_fire_step == current_step and tilt_last_reading:
                    # A budget fire already sampled this exact pose this step. Re-label
                    # it instead of paying for a second identical generation -- the
                    # commit is what makes the sample interesting, not a new reading.
                    reading = dict(tilt_last_reading)
                    reading["source"] = source_label
                    reading["arm_reason"] = "committed_stop"
                    reading["committed_stop"] = True
                    reading["deduped_from_budget_fire"] = True
                    log_fallback_event(
                        "endgame_tilt_view",
                        current_episode_id,
                        current_step,
                        reading,
                    )
                    return reading
                if not force:
                    if tilt_fire_count >= tilt_max_per_episode:
                        return None
                    if (
                        tilt_last_fire_step is not None
                        and current_step - tilt_last_fire_step < tilt_min_step_gap
                    ):
                        return None
                try:
                    # Arm/dedupe live inside the guard too: they read episode-scoped
                    # state (latest_phase_evidence, recent_ftv_window, positions) and
                    # must never be able to break the main loop.
                    armed, arm_reason = _tilt_endgame_armed()
                    if force:
                        arm_reason = "committed_stop"
                    elif not armed:
                        return None
                    if not force and not _tilt_pose_is_new():
                        return None
                    obs0 = observations[0] if isinstance(observations, (list, tuple)) else observations
                    if not isinstance(obs0, dict) or "tilt_rgb" not in obs0:
                        return None
                    tilt_depth_raw = obs0.get("tilt_depth")
                    tilt_depth_m = _metric_depth_center_median(
                        tilt_depth_raw, depth_veto_min, depth_veto_max,
                        depth_veto_normalize, tilt_center_frac,
                    )
                    tilt_rgb_img = Image.fromarray(obs0["tilt_rgb"], mode="RGB")
                    tilt_depth_img = None
                    if tilt_depth_raw is not None:
                        d = np.asarray(tilt_depth_raw, dtype=np.float32)
                        if d.ndim == 3 and d.shape[-1] == 1:
                            d = d[..., 0]
                        span = float(np.max(d) - np.min(d))
                        if span > 1e-6:
                            tilt_depth_img = Image.fromarray(
                                (255 * (d - np.min(d)) / span).astype(np.uint8)
                            )
                    description = None
                    if tilt_depth_img is not None:
                        description = navigator.spatial.observe_view(
                            nav_logger,
                            current_step,
                            "tilt",
                            {"rgb": tilt_rgb_img, "depth": tilt_depth_img},
                        )
                    tilt_fire_count += 1
                    tilt_last_fire_step = current_step
                    tilt_last_pose = (
                        positions[0][0], positions[0][1], positions[0][2],
                        float(headings[0]),
                    )
                    reading = {
                        "source": source_label,
                        "arm_reason": arm_reason,
                        "committed_stop": bool(force),
                        "pitch_deg": tilt_pitch_deg,
                        # Range at which the pitched ray meets a flat empty floor
                        # (camera_height / sin|pitch|). At -30deg with the 1.25m
                        # sensor this is 2.50m, and the ep100 20260719 tilt median
                        # was 2.20m -- i.e. most readings are floor, not object.
                        # Any near-field claim must clear this by a real margin.
                        "floor_intersect_m": tilt_floor_intersect_m,
                        "tilt_depth_center_m": tilt_depth_m,
                        "eye_level_depth_center_m": _metric_depth_center_median(
                            _current_view_depth_array(observations),
                            depth_veto_min, depth_veto_max,
                            depth_veto_normalize, tilt_center_frac,
                        ),
                        "observation": description,
                        "fire_index": tilt_fire_count,
                        "log_only": tilt_log_only,
                    }
                    tilt_last_reading = reading
                    log_fallback_event(
                        "endgame_tilt_view",
                        current_episode_id,
                        current_step,
                        reading,
                    )
                    return reading
                except Exception as exc:
                    log_fallback_event(
                        "endgame_tilt_view",
                        current_episode_id,
                        current_step,
                        {"source": source_label, "error": repr(exc)},
                    )
                    return None

            def _depth_stop_ok(source_label):
                """Observable depth veto for a STOP-enabling gate. Returns (ok, reading).

                ok=True means "not vetoed" (depth within threshold, OR unavailable ->
                fail-open, OR feature disabled). Logs raw depth + view + verdict every
                call: this trace is the ONLY channel to measure hit / false-veto rate
                offline, so it is emitted even when disabled=False is NOT the case.
                """
                # _depth_stop_ok is the shared choke point for every STOP-enabling
                # gate, so the endgame tilt observation hangs here -- but ABOVE the
                # disabled early-return, so the two switches stay independent
                # (attribution: tilt must be runnable with the veto off).
                _tilt_observe(source_label)
                if not depth_veto_enabled:
                    return True, None
                reading = _metric_depth_center_median(
                    _current_view_depth_array(observations),
                    depth_veto_min, depth_veto_max, depth_veto_normalize,
                    depth_veto_center_frac,
                )
                vetoed = reading is not None and reading > depth_veto_dist
                # Use the UNCONDITIONAL logger (not log_u_event, which early-returns
                # unless u_series_active): the depth-veto trace is the only offline
                # channel to measure hit / false-veto rate, and it must survive an
                # A/B run that enables DEPTH_STOP_VETO alone with U_SERIES off.
                log_fallback_event(
                    "depth_stop_veto",
                    current_episode_id,
                    current_step,
                    {
                        "enabled": True,
                        "source": source_label,
                        "depth_reading_m": reading,
                        "view_id": stop_current_view_candidate_id,
                        "threshold_m": depth_veto_dist,
                        "center_frac": depth_veto_center_frac,
                        "vetoed": bool(vetoed),
                        "fail_open_unavailable": reading is None,
                    },
                )
                return (not vetoed), reading

            # Backtracking activation (hunk 4, default OFF -> baseline byte-identical).
            # The policy module is oracle-free (ego dead-reckoning / dead-end only).
            backtrack_enabled = False
            backtrack_max_per_episode = 2
            _bt_cfg = getattr(config.OPENNAV_HARNESS, "BACKTRACK", None)
            if _bt_cfg is not None:
                backtrack_enabled = bool(getattr(_bt_cfg, "ENABLED", False))
                backtrack_max_per_episode = int(getattr(_bt_cfg, "MAX_BACKTRACKS_PER_EPISODE", 2))
            backtrack_policy = (
                BacktrackPolicy(
                    max_backtracks_per_episode=backtrack_max_per_episode,
                    stall_window=int(getattr(_bt_cfg, "STALL_WINDOW", 3)) if _bt_cfg is not None else 3,
                    disp_eps=float(getattr(_bt_cfg, "DISP_EPS", 1.0)) if _bt_cfg is not None else 1.0,
                    disp_ratio=float(getattr(_bt_cfg, "DISP_RATIO", 0.35)) if _bt_cfg is not None else 0.35,
                )
                if backtrack_enabled
                else None
            )
            # Resolved-switch banner. Printed BEFORE the episode loop so a mis-launch
            # (e.g. YACS overrides not forwarded) is visible in seconds instead of
            # after a ~9h ep100 run -- see the 20260718 run that silently reproduced
            # the clean baseline byte-for-byte because DEPTH_STOP_VETO stayed False.
            print(
                "[HARNESS SWITCHES] depth_stop_veto={} (max_dist={}m) backtrack={} "
                "(max/ep={}) endgame_tilt={} (pitch={}deg log_only={} cap={} "
                "gap={} on_stop={} arm_fc={}) cand_prior={} (log_only={} min_gap={}) "
                "progress_provider={} u_series={} unit={} geometry_injection={}".format(
                    depth_veto_enabled, depth_veto_dist,
                    backtrack_enabled, backtrack_max_per_episode,
                    tilt_enabled, tilt_pitch_deg, tilt_log_only,
                    tilt_max_per_episode, tilt_min_step_gap,
                    tilt_always_on_stop, tilt_arm_final_clause,
                    candidate_prior_enabled, candidate_prior_log_only,
                    candidate_prior_min_gap,
                    progress_provider_name,
                    u_module_enabled(config, "PHASE_EVIDENCE"),
                    u_decision_unit,
                    geometry_injection_enabled,
                ),
                flush=True,
            )
            # C0 / C2 / C5 / ACN switches. Added 20260805: the banner above predates all
            # of them, so a mis-launch of exactly the levers now queued would NOT have
            # been visible -- which is the failure mode the banner exists to prevent.
            _vtv = config.OPENNAV_HARNESS.VISUAL_TARGET_VERIFIER
            _nav_rt = getattr(config.OPENNAV_HARNESS, "NAVIGATION_RUNTIME", None)
            _mem = config.OPENNAV_HARNESS.MEMORY_DIAGNOSTIC
            print(
                "[HARNESS SWITCHES 2] C0.min_steps_before_allow={} | "
                "C2.step_limit short/long={}/{} | "
                "C5.memory log_only={} tau={} merge_r={}m vertical={} | "
                "ACN anchor_chain={} locator(log_only={} location={} dom={}) "
                "pool(log_only={} restrict={}) terminal_gate(log_only={})".format(
                    getattr(_vtv, "MIN_STEPS_BEFORE_ALLOW", None),
                    getattr(_nav_rt, "SHORT_ACTION_STEP_LIMIT", None),
                    getattr(_nav_rt, "LONG_ACTION_STEP_LIMIT", None),
                    module_log_only(config, "MEMORY_DIAGNOSTIC"),
                    getattr(_mem, "LOOP_ALERT_THRESHOLD", None),
                    getattr(_mem, "MERGE_RADIUS_M", None),
                    getattr(_mem, "ENABLE_VERTICAL_ALERT", None),
                    module_enabled(config, "ANCHOR_CHAIN"),
                    module_log_only(config, "PROGRESS_LOCATOR"),
                    getattr(
                        config.OPENNAV_HARNESS.PROGRESS_LOCATOR,
                        "ENABLE_LOCATION", None,
                    ),
                    getattr(
                        config.OPENNAV_HARNESS.PROGRESS_LOCATOR,
                        "LOCATION_DOMINANCE", None,
                    ),
                    module_log_only(config, "LANDMARK_POOL"),
                    getattr(
                        config.OPENNAV_HARNESS.LANDMARK_POOL,
                        "RESTRICT_TO_VOCABULARY", None,
                    ),
                    module_log_only(config, "TERMINAL_GATE"),
                ),
                flush=True,
            )

            oracle_metrics_enabled = module_enabled(config, "ORACLE_METRICS")
            llm_runtime_config = getattr(config.OPENNAV_HARNESS, "LLM_RUNTIME", None)
            if llm_runtime_config is not None:
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

        def run_harness_tool(tool_name, trace_step_id, fallback, func, *args, **kwargs):
            if not harness_enabled or func is None:
                return fallback
            start_time = time.perf_counter()
            try:
                result = func(*args, **kwargs)
                return result
            except Exception as exc:
                if harness_logger is not None:
                    harness_logger.log_tool_failure(tool_name, trace_step_id, exc)
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
                    trace_step_id,
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

        def log_route_state(stage, selected_candidate=None, stop_requested=False,
                            stop_reason="", action_result=None):
            if not harness_enabled or harness_logger is None:
                return
            state = build_route_state(
                episode_id=current_episode_id,
                step_id=current_step,
                stage=stage,
                instruction=instruction,
                actions=actions,
                landmarks=landmarks,
                anchors=_anchors,
                anchor_chain_result=anchor_chain_result,
                progress_locator_result=progress_locator_results,
                progress_update_result=(
                    progress_update_contract.to_dict()
                    if progress_update_contract is not None
                    else None
                ),
                progress_locator_decision_effect=progress_locator_decision_effect,
                phase_evidence=latest_phase_evidence,
                spatial_memory_result=memory_results,
                landmark_pool_result=landmark_pool_results,
                failure_status=latest_failure_signal,
                terminal_gate_result=terminal_gate_results,
                stop_decision_result=(
                    m3_stop_decision.to_dict()
                    if m3_stop_decision is not None
                    else None
                ),
                stop_evidence_items=[
                    item.to_dict() for item in m3_stop_evidence_items
                ],
                candidate_ids=[getattr(c, "candidate_id", None) for c in candidates],
                selected_candidate=selected_candidate,
                stop_requested=stop_requested,
                stop_reason=stop_reason,
                step_limit=step_length,
                action_result=action_result,
            )
            write_navigation_record(
                "route_state",
                episode_id=current_episode_id,
                step=current_step,
                state=state,
            )
            harness_logger.log_event("route_state", current_step, {"state": state})

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
        terminal_evidence_memory = TerminalEvidenceMemory(
            max_age_steps=2, max_displacement_m=2.25
        )
        terminal_instance_tracker = TerminalInstanceTracker(
            max_position_delta_m=2.25
        )
        episode_forward_translation_m = 0.0
        episode_forward_move_steps = 0
        # Backtracking (hunk 4) episode-scoped state. Initialized pre-loop so the
        # per-step position append cannot NameError on an episode's first step
        # (the per-episode reset block runs later in the loop body). Reset again
        # per episode below.
        recent_positions_history: List[Any] = []
        last_executed_move: Optional[Dict[str, float]] = None
        backtrack_budget_remaining = backtrack_max_per_episode
        backtrack_blocked_headings: set = set()
        just_backtracked = False
        active_anchor_chain_result = {}
        active_anchors = []
        while envs.num_envs > 0 and len(stats_episodes) < episodes_to_eval:
            current_episodes = envs.current_episodes()
            positions = []; headings = []
            for ob_i in range(len(current_episodes)): 
                agent_state_i = envs.call_at(ob_i,
                        "get_agent_info", {})
                positions.append(agent_state_i['position'])
                headings.append(agent_state_i['heading'])
            # Backtracking (hunk 4): roll the ego pose history (window+1 poses) for the
            # observable displacement-stall trigger. Uses only the agent's own pose
            # differences (odometry-equivalent) -- no goal distance, no oracle.
            if backtrack_enabled and positions:
                recent_positions_history.append(positions[0])
                if len(recent_positions_history) > backtrack_policy.stall_window + 1:
                    recent_positions_history.pop(0)
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
            # ACN state is episode-scoped. Keep the last chain for all later steps;
            # resetting L1/M2 here would erase the very cross-step state they provide.
            anchor_chain_result = active_anchor_chain_result
            _anchors = active_anchors
            if active_navigation_episode_id != current_episode_id:
                active_navigation_episode_id = current_episode_id
                active_anchor_chain_result = run_harness_tool(
                    "route_state_reducer_reset",
                    0,
                    {},
                    route_state_reducer.reset_episode
                    if route_state_reducer is not None
                    else None,
                    current_episode_id,
                    actions,
                    landmarks,
                ) or {}
                if progress_provider is None:
                    raise RuntimeError(
                        "ACN L1 progress provider is required; LLM completion fallback is disabled"
                    )
                progress_provider.reset_episode(current_episode_id)
                if harness_enabled and harness_logger is not None:
                    harness_logger.log_event(
                        "anchor_chain", 0, active_anchor_chain_result
                    )
                    harness_logger.log_event(
                        "route_state_reducer_reset",
                        0,
                        route_state_reducer.snapshot(),
                    )
                active_anchors = (
                    active_anchor_chain_result.get("anchors", [])
                    if isinstance(active_anchor_chain_result, dict)
                    else []
                )
                anchor_chain_result = active_anchor_chain_result
                _anchors = active_anchors
                recent_distance_gains = []
                latest_goal_dist = None
                recovery_budget_remaining = max_recovery_per_episode
                latest_phase_evidence = {}
                latest_failure_signal = {}
                recent_ftv_window: List[bool] = []
                terminal_evidence_memory.reset()
                terminal_instance_tracker.reset(current_episode_id)
                episode_forward_translation_m = 0.0
                episode_forward_move_steps = 0
                route_last_pre_action_position = None
                # Backtracking per-episode reset (guarantees no cross-episode leakage
                # into a firing decision; a firing needs a full fresh pose window).
                recent_positions_history = []
                last_executed_move = None
                backtrack_budget_remaining = backtrack_max_per_episode
                backtrack_blocked_headings = set()
                just_backtracked = False
                # Endgame tilt per-episode reset (budget and pose dedupe must not
                # carry across episodes).
                tilt_fire_count = 0
                tilt_last_pose = None
                tilt_last_fire_step = None
                tilt_last_reading = None
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
            m3_stop_proposals = []
            m3_stop_evidence_items = []
            m3_stop_decision = None
            m3_preplanner_stop_decided = False
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
            latest_candidate_prior = None
            geometry_results = []
            grounding_results = []
            visual_evidence_results = {}
            stop_current_view_evidence_results = None
            visual_evidence_memory_results = {}
            memory_results = {}
            memory_spatial_alert = ""
            progress_locator_results = {}
            landmark_pool_results = {}
            terminal_gate_results = {}
            progress_update_contract = None
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
            observation_frame_contract = run_harness_tool(
                "pipeline_observation_frame",
                current_step,
                None,
                build_observation_frame,
                episode_id=current_episode_id,
                step_id=current_step,
                instruction=instruction,
                candidates=candidates,
                observe_dict=observe_dict,
                position=positions[0] if positions else None,
                heading=headings[0] if headings else None,
                plan_ref="{}:plan".format(current_episode_id),
            )
            if observation_frame_contract is not None and harness_logger is not None:
                harness_logger.log_event(
                    "pipeline_observation_frame",
                    current_step,
                    observation_frame_contract.to_dict(),
                )
            # Backtracking (hunk 4): decide whether MOVE_BACK is on the menu this step,
            # from observable signals only (ego displacement stall OR waypoint dead-end).
            # dead_end := len(radius_dict) == 0 (NO forward candidate). This is a
            # deliberate tightening from an earlier <= 1 draft: a single remaining
            # candidate is no longer treated as a dead-end (avoids offering backtrack
            # when a real forward move still exists). offer_move_back=False keeps the
            # navigator/test_decisions path byte-identical.
            offer_move_back = False
            backtrack_offer_info = None
            if backtrack_enabled and backtrack_policy is not None:
                backtrack_offer_info = backtrack_policy.available(
                    last_move=last_executed_move,
                    recent_positions=recent_positions_history,
                    budget_remaining=backtrack_budget_remaining,
                    just_backtracked=just_backtracked,
                    dead_end=(len(radius_dict) == 0),
                )
                offer_move_back = bool(backtrack_offer_info.get("offered"))
                log_fallback_event(
                    "backtrack_offer",
                    current_episode_id,
                    current_step,
                    backtrack_offer_info,
                )
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
                # ---- Arm C: observable-feature candidate prior -------------------
                # Placed here because it needs visual_evidence (nmatch) and must land
                # before the selector context is built. Writes into the previously
                # inert scaffold fields on the candidate records so any downstream
                # consumer sees a populated geometry_score / final_prompt_rank.
                # LOG_ONLY by default: scores are computed and logged but the
                # candidate ORDER handed to the selector is untouched, so a
                # LOG_ONLY run stays comparable to one with the prior off.
                if candidate_prior_enabled and candidates:
                    try:
                        _ve_by_id = _parse_visual_evidence_candidates(
                            visual_evidence_results
                        )
                        _prior_scores, _prior_ranks = candidate_prior.score_candidates(
                            candidates, _ve_by_id
                        )
                        if _prior_scores:
                            # candidates are CandidateState dataclasses at runtime
                            # but plain dicts when this path is replayed from a
                            # trace; write back through whichever shape applies.
                            for _c in candidates:
                                if isinstance(_c, dict):
                                    _cid = str(_c.get("candidate_id"))
                                else:
                                    _cid = str(getattr(_c, "candidate_id", None))
                                if _cid not in _prior_scores:
                                    continue
                                if isinstance(_c, dict):
                                    _c["geometry_score"] = _prior_scores[_cid]
                                    _c["final_prompt_rank"] = _prior_ranks[_cid]
                                else:
                                    _c.geometry_score = _prior_scores[_cid]
                                    _c.final_prompt_rank = _prior_ranks[_cid]
                            latest_candidate_prior = (_prior_scores, _prior_ranks)
                            harness_logger.log_event(
                                "candidate_prior",
                                current_step,
                                {
                                    "log_only": candidate_prior_log_only,
                                    "scores": _prior_scores,
                                    "ranks": _prior_ranks,
                                    "prior_top": min(
                                        _prior_ranks, key=_prior_ranks.get
                                    ),
                                    "candidate_count": len(candidates),
                                },
                            )
                    except Exception as _cp_exc:
                        harness_logger.log_event(
                            "candidate_prior",
                            current_step,
                            {"error": repr(_cp_exc)},
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
                    route_state_reducer.update_spatial
                    if memory_diagnostic is not None
                    else None,
                    current_step,
                    positions[0] if positions else None,
                    headings[0] if headings else None,
                    candidates,
                )
                if memory_diagnostic is not None:
                    # C5: the ONLY decision effect -- hand G_topo's alert text to the
                    # navigator prompt. Candidate set, STOP path and scoring untouched.
                    _alert = (
                        memory_results.get("alert_text", "")
                        if isinstance(memory_results, dict)
                        else ""
                    )
                    memory_alert_applied = bool(
                        memory_alert_decision_effect and _alert
                    )
                    memory_spatial_alert = _alert if memory_alert_applied else ""
                    if isinstance(memory_results, dict):
                        memory_results["decision_effect_enabled"] = (
                            memory_alert_decision_effect
                        )
                        memory_results["applied"] = memory_alert_applied
                    harness_logger.log_event(
                        "memory_diagnostic",
                        current_step,
                        memory_results,
                    )

                # ---- ACN L1 / M2 per step (LOG_ONLY by default) ----
                # Both read only the agent's own perception + odometry: RAM tags out of
                # observe_dict and the waypoint geometry already in `candidates`. No GT,
                # no new model, no LLM call.
                _view_tags, _view_geometry = {}, {}
                for _cand in candidates or []:
                    _cid = str(getattr(_cand, "candidate_id", ""))
                    if not _cid:
                        continue
                    _view_geometry[_cid] = {
                        "angle_rad": getattr(_cand, "angle_rad", None),
                        "distance": getattr(_cand, "distance", None),
                    }
                for _cid, _text in (observe_dict or {}).items():
                    _m = re.search(r"scene objects:(.*)", str(_text), re.I | re.S)
                    if _m:
                        _view_tags[str(_cid)] = [
                            t.strip().lower()
                            for t in _m.group(1).split("|")
                            if t.strip()
                        ]
                _cur_pos = positions[0] if positions else None
                _cur_head = headings[0] if headings else None
                if progress_locator is not None:
                    progress_locator_results = run_harness_tool(
                        "progress_locator", current_step, {},
                        route_state_reducer.update_progress,
                        current_step, _cur_pos, _cur_head, _view_tags, _view_geometry,
                    ) or {}
                    harness_logger.log_event(
                        "progress_locator", current_step, progress_locator_results
                    )
                if landmark_pool is not None:
                    landmark_pool_results = run_harness_tool(
                        "landmark_pool", current_step, {},
                        route_state_reducer.update_landmarks,
                        current_step, _cur_pos, _cur_head, _view_tags, _view_geometry,
                    ) or {}
                    harness_logger.log_event(
                        "landmark_pool", current_step, landmark_pool_results
                    )
                if terminal_gate is not None:
                    terminal_gate_results = run_harness_tool(
                        "terminal_gate", current_step, {},
                        route_state_reducer.evaluate_stop,
                        progress_locator_results,
                        landmark_pool,
                        (final_landmark_terms(landmarks) or [None])[-1],
                    ) or {}
                    if isinstance(terminal_gate_results, dict):
                        terminal_gate_results["decision_effect_enabled"] = (
                            terminal_gate_decision_effect
                        )
                    harness_logger.log_event(
                        "terminal_gate", current_step, terminal_gate_results
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
                nav_logger.info("========== ACN L1 Progress ==========")
                completion_start_time = time.perf_counter()
                if progress_provider is None:
                    raise RuntimeError(
                        "ACN L1 progress provider is required; LLM completion fallback is disabled"
                    )
                progress_update_contract = progress_provider.update(
                    progress_locator_results,
                    evidence_refs=(
                        "progress_locator:{}".format(current_step),
                    ),
                )
                estimation = progress_update_contract.display_text
                record_runtime_latency(
                    "acn_l1_progress",
                    current_episode_id,
                    current_step,
                    time.perf_counter() - completion_start_time,
                    category="state_reducer",
                    provider="acn_l1",
                )
                write_navigation_record(
                    "completion_estimation",
                    episode_id=current_episode_id,
                    step=current_step,
                    estimation=estimation,
                    provider="acn_l1",
                    llm_called=False,
                    progress_update=progress_update_contract.to_dict(),
                )
                if harness_logger is not None:
                    harness_logger.log_event(
                        "pipeline_progress_update",
                        current_step,
                        progress_update_contract.to_dict(),
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
                    # v2 depth veto (default OFF): additional observable conjunct on the
                    # STOP-enabling gates (#2 arrival_override / #4 trajectory_bypass) inside
                    # verify(). _depth_ok=True when disabled or unavailable (fail-open), so
                    # byte-identical when off. See _depth_stop_ok / _metric_depth_center_median.
                    _depth_ok, _ = _depth_stop_ok(source)
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
                        depth_confirm=_depth_ok,
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

                def apply_terminal_gate(source, stop_proposal, reason):
                    """Apply L4 at the last STOP decision point."""
                    gate = (
                        terminal_gate_results
                        if isinstance(terminal_gate_results, dict)
                        else {}
                    )
                    resolved_stop, resolved_reason, blocked = gate_stop_request(
                        stop_proposal,
                        reason,
                        gate,
                        terminal_gate_decision_effect,
                    )
                    if not blocked:
                        return resolved_stop, resolved_reason, False
                    payload = {
                        "source": source,
                        "original_stop_reason": reason,
                        "verdict": gate.get("verdict"),
                        "gate_reason": gate.get("reason"),
                        "conjuncts": gate.get("conjuncts"),
                        "j": gate.get("j"),
                        "n_anchors": gate.get("n_anchors"),
                        "action_affecting": True,
                    }
                    write_navigation_record(
                        "terminal_gate_rejected",
                        episode_id=current_episode_id,
                        step=current_step,
                        **payload,
                    )
                    if harness_enabled and harness_logger is not None:
                        harness_logger.log_event(
                            "terminal_gate_rejected",
                            current_step,
                            payload,
                        )
                    nav_logger.info(
                        "Terminal gate rejected STOP from {}: {}".format(
                            source, gate.get("reason")
                        )
                    )
                    return False, "", True

                def resolve_m3_goal_stop(
                    source, requested, allowed, reason, verifier_result=None,
                    proactive_policy=None
                ):
                    """Make StopCoordinator authoritative without changing gate policy."""
                    nonlocal m3_stop_decision, m3_preplanner_stop_decided
                    if not requested:
                        return None
                    state_id = "{}:{}:decision_ready".format(
                        current_episode_id, current_step
                    )
                    proposal_id = "{}:{}:{}".format(
                        current_episode_id, current_step, source
                    )
                    proposal = StopProposal(
                        proposal_id=proposal_id,
                        source=source,
                        kind="goal_stop",
                        reason=str(reason or "STOP requested."),
                        route_state_id=state_id,
                        step_id=current_step,
                    )
                    evidence_items = build_m3_2_evidence_items(
                        proposal,
                        allowed=bool(allowed),
                        verifier_result=verifier_result,
                        terminal_gate_result=terminal_gate_results,
                        progress_update=(
                            progress_update_contract.to_dict()
                            if progress_update_contract is not None
                            else {}
                        ),
                        proactive_policy=proactive_policy,
                    )
                    decision = stop_coordinator.resolve(
                        route_state_id=state_id,
                        step_id=current_step,
                        proposals=(proposal,),
                        evidence_items=evidence_items,
                        movement_candidate_ids=tuple(
                            str(key) for key in (observe_dict or {}).keys()
                        ),
                    )
                    m3_stop_proposals.append(proposal)
                    m3_stop_evidence_items.extend(evidence_items)
                    m3_stop_decision = decision
                    if source in {"progress_completion", "proactive_visual"}:
                        m3_preplanner_stop_decided = True
                    event_payloads = [
                        ("pipeline_stop_proposal", proposal.to_dict())
                    ]
                    event_payloads.extend(
                        ("pipeline_stop_evidence", item.to_dict())
                        for item in evidence_items
                    )
                    event_payloads.append(
                        ("pipeline_stop_decision", decision.to_dict())
                    )
                    for event_type, payload in event_payloads:
                        write_navigation_record(
                            event_type,
                            episode_id=current_episode_id,
                            step=current_step,
                            **payload
                        )
                        if harness_enabled and harness_logger is not None:
                            harness_logger.log_event(
                                event_type, current_step, payload
                            )
                    return decision

                def resolve_m3_forced_termination(source, reason):
                    """Resolve operational STOP only when no movement remains."""
                    nonlocal m3_stop_decision
                    if m3_stop_decision is not None:
                        if m3_stop_decision.outcome in {
                            "commit_goal_stop", "commit_forced_termination"
                        }:
                            return m3_stop_decision
                    state_id = "{}:{}:decision_ready".format(
                        current_episode_id, current_step
                    )
                    proposal = StopProposal(
                        proposal_id="{}:{}:{}".format(
                            current_episode_id, current_step, source
                        ),
                        source=source,
                        kind="forced_termination",
                        reason=str(reason),
                        route_state_id=state_id,
                        step_id=current_step,
                    )
                    m3_stop_proposals.append(proposal)
                    decision = stop_coordinator.resolve(
                        route_state_id=state_id,
                        step_id=current_step,
                        proposals=tuple(m3_stop_proposals),
                        evidence_items=tuple(m3_stop_evidence_items),
                        movement_candidate_ids=(),
                    )
                    m3_stop_decision = decision
                    for event_type, payload in (
                        ("pipeline_stop_proposal", proposal.to_dict()),
                        ("pipeline_stop_decision", decision.to_dict()),
                    ):
                        write_navigation_record(
                            event_type, episode_id=current_episode_id,
                            step=current_step, **payload
                        )
                        if harness_enabled and harness_logger is not None:
                            harness_logger.log_event(
                                event_type, current_step, payload
                            )
                    return decision

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

                log_route_state("pre_action")
                stop_flag, stop_reason = navigator.should_stop(
                    nav_logger,
                    actions,
                    landmarks,
                    estimation,
                    history_traj,
                    observation,
                    current_step=current_step,
                )
                legacy_completion_stop_requested = bool(stop_flag)
                stop_gate_metadata = dict(
                    getattr(navigator, "last_stop_gate_metadata", {}) or {}
                )
                stop_gate_metadata["legacy_completion_requested"] = (
                    legacy_completion_stop_requested
                )
                # ACN route completion alone cannot create a goal-stop proposal.
                # Terminal evidence is joined below after the current visual frame
                # has been evaluated.
                completion_stop_requested = False
                completion_stop_reason = ""
                stop_flag, stop_reason = False, ""
                proactive_stop_requested = False
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
                    True,
                    "Evaluate current-frame terminal evidence; this is not yet a STOP proposal.",
                    selected_candidate=stop_current_view_candidate_id,
                    stop_evidence_mode="current_pano",
                )
                completion_verifier_results = direction_aligned_terminal_result(
                    completion_verifier_results
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
                _terminal_candidate = (
                    completion_verifier_results.get("selected_candidate_verdict") or {}
                )
                _terminal_direction_id = _terminal_candidate.get("target_direction_id")
                _terminal_depth_array = _direction_depth_array(
                    observations, _terminal_direction_id
                )
                _terminal_center_depth_m = _metric_depth_center_median(
                    _terminal_depth_array,
                    depth_veto_min, depth_veto_max, depth_veto_normalize,
                    depth_veto_center_frac,
                )
                _terminal_near_surface_depth_m = _metric_depth_near_surface(
                    _terminal_depth_array,
                    depth_veto_min, depth_veto_max, depth_veto_normalize,
                )
                _terminal_required_terms = [
                    str(v).strip().lower()
                    for v in (
                        completion_verifier_results.get("required_landmark_terms")
                        or []
                    )
                ]
                _terminal_is_opening = any(
                    any(token in term for token in (
                        "archway", "doorway", "door", "entryway", "entry way"
                    ))
                    for term in _terminal_required_terms
                )
                _terminal_depth_m = (
                    _terminal_near_surface_depth_m
                    if _terminal_is_opening
                    else _terminal_center_depth_m
                )
                _terminal_spatial_confirmed = bool(
                    _terminal_depth_m is not None
                    and _terminal_depth_m <= depth_veto_dist
                )
                terminal_spatial_payload = {
                    "target_direction_id": _terminal_direction_id,
                    "local_surface_depth_m": _terminal_depth_m,
                    "target_depth_m": _terminal_depth_m,
                    "target_depth_m_deprecated": True,
                    "center_depth_m": _terminal_center_depth_m,
                    "near_surface_depth_m": _terminal_near_surface_depth_m,
                    "distance_estimator": (
                        "opening_frame_p20" if _terminal_is_opening
                        else "target_direction_center_median"
                    ),
                    "max_target_depth_m": depth_veto_dist,
                    "distance_satisfied": _terminal_spatial_confirmed,
                    "source": "rgbd_target_direction",
                    "distance_semantics": "direction_local_surface_not_instance_mask",
                    "oracle_free": True,
                }
                _weak_generic_terminal = is_weak_generic_terminal(
                    completion_verifier_results
                )
                _weak_generic_route_matured = bool(
                    episode_forward_move_steps >= 4
                    and episode_forward_translation_m >= 6.0
                )
                terminal_spatial_payload.update({
                    "weak_generic_terminal": _weak_generic_terminal,
                    "route_maturity_required": _weak_generic_terminal,
                    "route_maturity_satisfied": (
                        _weak_generic_route_matured
                        if _weak_generic_terminal else True
                    ),
                    "non_collision_forward_steps": episode_forward_move_steps,
                    "actual_displacement_m": episode_forward_translation_m,
                    "position_source": "simulator_agent_pose",
                    "min_forward_move_steps": 4,
                    "min_forward_translation_m": 6.0,
                })
                terminal_instance_result = terminal_instance_tracker.update(
                    step_id=current_step,
                    position=(positions[0] if positions else ()),
                    heading=(headings[0] if headings else None),
                    direction_id=_terminal_direction_id,
                    depth_m=_terminal_depth_m,
                    target_terms=_terminal_required_terms,
                    supporting_landmarks=(
                        _terminal_candidate.get("visible_landmarks") or []
                    ),
                    visually_supported=bool(
                        completion_verifier_results.get("final_target_visible") is True
                        and completion_verifier_results.get("arrival_evidence") is True
                    ),
                )
                log_fallback_event(
                    "terminal_instance_track", current_episode_id, current_step,
                    terminal_instance_result,
                )
                _persistent_visual_confirm = bool(
                    terminal_instance_result.get("consecutive_observations", 0) >= 2
                )
                terminal_spatial_payload["instance_key"] = (
                    terminal_instance_result.get("instance_key")
                )
                terminal_spatial_payload["instance_consecutive_observations"] = (
                    terminal_instance_result.get("consecutive_observations")
                )
                terminal_spatial_payload["same_instance"] = (
                    terminal_instance_result.get("same_instance")
                )
                terminal_spatial_payload["context_overlap"] = (
                    terminal_instance_result.get("context_overlap")
                )
                terminal_spatial_payload["confirmed_instance_switch"] = (
                    terminal_instance_result.get("confirmed_instance_switch")
                )
                _terminal_identity_supported = terminal_target_is_confirmed(
                    completion_verifier_results,
                    persistent_visual_confirm=True,
                    spatial_distance_confirmed=True,
                    weak_generic_route_matured=_weak_generic_route_matured,
                )
                _terminal_direct_confirmed = terminal_target_is_confirmed(
                    completion_verifier_results,
                    persistent_visual_confirm=_persistent_visual_confirm,
                    spatial_distance_confirmed=_terminal_spatial_confirmed,
                    weak_generic_route_matured=_weak_generic_route_matured,
                )
                _route_complete_now = bool(
                    progress_update_contract is not None
                    and progress_update_contract.route_progress_complete is True
                )
                _weak_generic_close_confirm = weak_generic_close_is_confirmed(
                    completion_verifier_results,
                    route_progress_complete=_route_complete_now,
                    route_maturity_satisfied=_weak_generic_route_matured,
                    local_surface_depth_m=_terminal_depth_m,
                    max_depth_m=1.5,
                )
                _terminal_direct_confirmed = bool(
                    _terminal_direct_confirmed or _weak_generic_close_confirm
                )
                terminal_spatial_payload["weak_generic_close_confirm"] = (
                    _weak_generic_close_confirm
                )
                terminal_spatial_payload["weak_generic_close_depth_m"] = 1.5
                log_fallback_event(
                    "terminal_spatial_evidence", current_episode_id, current_step,
                    terminal_spatial_payload,
                )
                terminal_memory_result = terminal_evidence_memory.update(
                    step_id=current_step,
                    position=(positions[0] if positions else ()),
                    direct_confirmed=_terminal_direct_confirmed,
                    current_visual_support=_terminal_identity_supported,
                    instance_key=terminal_instance_result.get("instance_key"),
                    confirmed_instance_switch=bool(
                        terminal_instance_result.get("confirmed_instance_switch")
                    ),
                    evidence=terminal_spatial_payload,
                )
                terminal_target_confirmed = bool(
                    terminal_memory_result.get("confirmed")
                )
                if (
                    terminal_target_confirmed
                    and completion_verifier_results.get("verdict") != "allow"
                    and _terminal_identity_supported
                ):
                    # A generic-only V2 uncertainty may be resolved only by the
                    # explicit current-text + RGB-D + persistence/memory policy.
                    # Preserve the raw verdict for audit; downstream consumes the
                    # single resolved TerminalEvidence rather than re-running V2.
                    completion_verifier_results = dict(
                        completion_verifier_results
                    )
                    completion_verifier_results["raw_v2_verdict"] = (
                        completion_verifier_results.get("verdict")
                    )
                    completion_verifier_results["verdict"] = "allow"
                    completion_verifier_results["verdict_resolution"] = (
                        "generic_target_text_rgbd_terminal_policy"
                    )
                    log_fallback_event(
                        "terminal_evidence_resolution",
                        current_episode_id,
                        current_step,
                        {
                            "raw_v2_verdict": completion_verifier_results.get(
                                "raw_v2_verdict"
                            ),
                            "resolved_verdict": "allow",
                            "policy": completion_verifier_results.get(
                                "verdict_resolution"
                            ),
                        },
                    )
                log_fallback_event(
                    "terminal_evidence_memory",
                    current_episode_id,
                    current_step,
                    terminal_memory_result,
                )
                progress_update_contract = progress_provider.apply_terminal_evidence(
                    progress_update_contract,
                    terminal_target_confirmed=terminal_target_confirmed,
                    evidence_refs=(
                        "visual_target_verifier:{}".format(current_step),
                        "visual_persistence:{}".format(current_step),
                        "terminal_spatial_evidence:{}".format(current_step),
                        "terminal_evidence_memory:{}".format(current_step),
                    ),
                )
                goal_progress_payload = progress_update_contract.to_dict()
                write_navigation_record(
                    "goal_progress_update",
                    episode_id=current_episode_id,
                    step=current_step,
                    **goal_progress_payload
                )
                if harness_logger is not None:
                    harness_logger.log_event(
                        "pipeline_goal_progress_update",
                        current_step,
                        goal_progress_payload,
                    )
                if progress_update_contract.goal_complete is True:
                    completion_stop_requested = True
                    completion_stop_reason = (
                        "Route progress is complete and the terminal target has "
                        "persistent visual arrival confirmation."
                    )
                    # Reuse the same formal V2 evidence that formed
                    # terminal_target_confirmed; do not create a second reading.
                    stop_flag, stop_reason, _ = apply_visual_stop_gate(
                        "completion_gate",
                        True,
                        completion_stop_reason,
                        completion_verifier_results,
                    )
                # Proactive is now a diagnostic endgame hint only. It never calls V2
                # again and cannot independently create or commit a STOP proposal.
                _route_index = progress_update_contract.current_index
                _route_total = progress_update_contract.total
                _route_endgame = bool(
                    progress_update_contract.route_progress_complete is True
                    or (
                        _route_index is not None
                        and _route_total is not None
                        and _route_total > 0
                        and _route_index >= _route_total - 1
                    )
                )
                proactive_terminal_hint = bool(
                    proactive_stop_enabled
                    and _route_endgame
                    and completion_verifier_results.get("final_target_visible") is True
                )
                if proactive_terminal_hint:
                    log_fallback_event(
                        "proactive_terminal_hint", current_episode_id, current_step,
                        {
                            "route_endgame": _route_endgame,
                            "shared_v2_verdict": completion_verifier_results.get("verdict"),
                            "terminal_target_confirmed": terminal_target_confirmed,
                            "goal_complete": progress_update_contract.goal_complete,
                            "decision_effect": False,
                        },
                    )
                proactive_stop_requested = False
                proactive_reason = ""
                proactive_verifier_results = completion_verifier_results
                _e3_b_allow = False
                _psg_depth_ok = _terminal_spatial_confirmed
                stop_flag, stop_reason, _ = apply_terminal_gate(
                    "completion_gate", stop_flag, stop_reason
                )
                preplanner_source = None
                if stop_flag and proactive_stop_requested:
                    preplanner_source = "proactive_visual"
                elif completion_stop_requested:
                    preplanner_source = "progress_completion"
                elif proactive_stop_requested:
                    preplanner_source = "proactive_visual"
                if preplanner_source is not None:
                    m3_stop_decision = resolve_m3_goal_stop(
                        preplanner_source,
                        True,
                        bool(stop_flag),
                        (
                            proactive_reason
                            if preplanner_source == "proactive_visual"
                            else completion_stop_reason
                        ),
                        (
                            proactive_verifier_results
                            if preplanner_source == "proactive_visual"
                            else completion_verifier_results
                        ),
                        (
                            {
                                "persistent_visual_confirm": _persistent_visual_confirm,
                                "v2_allow": visual_target_verifier_allows_stop(
                                    proactive_verifier_results
                                ),
                                "e3_b_allow": _e3_b_allow,
                                "depth_ok": _psg_depth_ok,
                            }
                            if preplanner_source == "proactive_visual"
                            else None
                        ),
                    )
                    stop_flag = (
                        m3_stop_decision.outcome in {
                            "commit_goal_stop",
                            "commit_forced_termination",
                        }
                    )
                    if stop_flag:
                        stop_reason = m3_stop_decision.reason
                    else:
                        stop_reason = ""
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
                        offer_move_back=offer_move_back,
                        spatial_alert=memory_spatial_alert,
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
                        offer_move_back=offer_move_back,
                    )
                    if next_vp not in (STOP_CANDIDATE, MOVE_BACK_CANDIDATE):
                        next_vp, _cp_audit = _candidate_prior_override(next_vp)
                        if _cp_audit is not None:
                            record_u_override(
                                build_decision_audit(
                                    unit_enabled="candidate_prior",
                                    original_action=_cp_audit["selector_choice"],
                                    proposed_action=_cp_audit["prior_choice"],
                                    final_action=next_vp,
                                    override_reason="candidate_prior_prob_gap",
                                    expected_failure_addressed="selection_regret",
                                    source_stage="candidate_prior",
                                    extra=_cp_audit,
                                )
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
                    if (
                        next_vp == STOP_CANDIDATE
                        and not m3_preplanner_stop_decided
                    ):
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
                        # Reuse the immutable per-step TerminalEvidence; selector
                        # STOP must not trigger a second V2 reading.
                        selector_verifier_results = dict(completion_verifier_results)
                        selector_verifier_results["source"] = "selector_stop_gate"
                        selector_verifier_results["reused_from"] = "completion_gate"
                        selector_stop_evidence_results = (
                            record_stop_evidence_verification(
                                "selector_stop_gate",
                                selector_verifier_results,
                                old_stop_gate_metadata,
                            )
                        )
                        selector_visual_rescue_candidate = bool(
                            progress_update_contract.goal_complete is True
                            and _terminal_spatial_confirmed
                            and visual_stop_can_rescue_selector_stop(
                                selector_verifier_results, old_stop_flag
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
                        if (
                            old_stop_flag
                            and progress_update_contract.goal_complete is True
                            and _terminal_spatial_confirmed
                            and visual_target_verifier_allows_stop(
                                selector_verifier_results
                            )
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
                            and progress_update_contract.goal_complete is True
                            and _terminal_spatial_confirmed
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
                        stop_flag, stop_reason, _ = apply_terminal_gate(
                            "selector_stop_gate", stop_flag, stop_reason
                        )
                        if m3_stop_decision is None:
                            m3_stop_decision = resolve_m3_goal_stop(
                                "selector",
                                True,
                                bool(stop_flag),
                                stop_reason or "Navigator selected STOP.",
                                selector_verifier_results,
                            )
                            stop_flag = (
                                m3_stop_decision.outcome in {
                                    "commit_goal_stop",
                                    "commit_forced_termination",
                                }
                            )
                        else:
                            # A rejected pre-planner STOP already consumed this
                            # step's single arbitration. Selector cannot reopen STOP.
                            stop_flag = False
                        if stop_flag:
                            stop_reason = m3_stop_decision.reason
                        else:
                            stop_reason = ""
                        if stop_flag:
                            if not stop_reason:
                                stop_reason = (
                                    "Navigator selected STOP and stop gate passed."
                                )
                        else:
                            nav_logger.info("Navigator selected STOP but stop gate failed; fallback to a movement candidate")
                            fallback_results = rank_movement_fallback(
                                "stop_rejected_fallback",
                                current_step,
                                selector_observe_dict,
                                visual_evidence_results,
                                instruction,
                                actions,
                                landmarks,
                                source_stage="stop_rejected",
                                reason="stop_rejected_no_second_selector",
                            )
                            next_vp = fallback_results.get("selected_candidate")
                            if next_vp in selector_observe_dict:
                                thought = selector_observe_dict[next_vp]
                            if next_vp not in (
                                None, STOP_CANDIDATE, MOVE_BACK_CANDIDATE
                            ):
                                next_vp, _cp_audit = _candidate_prior_override(
                                    next_vp
                                )
                                if _cp_audit is not None:
                                    _cp_audit["path"] = "after_stop_rejection"
                                    record_u_override(
                                        build_decision_audit(
                                            unit_enabled="candidate_prior",
                                            original_action=_cp_audit["selector_choice"],
                                            proposed_action=_cp_audit["prior_choice"],
                                            final_action=next_vp,
                                            override_reason="candidate_prior_prob_gap",
                                            expected_failure_addressed="selection_regret",
                                            source_stage="candidate_prior",
                                            extra=_cp_audit,
                                        )
                                    )
                            fallback_results["second_selector_called"] = False
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
                _contract_state_id = "{}:{}:decision_ready".format(
                    current_episode_id, current_step
                )
                _contract_decision_id = "{}:{}:decision".format(
                    current_episode_id, current_step
                )
                _contract_command_id = "{}:{}:command".format(
                    current_episode_id, current_step
                )
                forced_termination_source = None
                backtrack_action_args = None
                # Backtracking (hunk 4) apply hook: when the navigator selected MOVE_BACK,
                # emit the reverse move directly (it bypasses radius_dict/distance_dict, which
                # do not contain the synthetic MOVE_BACK id). Runs before the normal action
                # builder and sets backtrack_reverse_emitted so the builder is skipped.
                backtrack_reverse_emitted = False
                if (not stop_flag) and backtrack_enabled and next_vp == MOVE_BACK_CANDIDATE:
                    bt_apply = backtrack_policy.apply(
                        MOVE_BACK_CANDIDATE,
                        last_executed_move,
                        backtrack_budget_remaining,
                        last_move_heading=(headings[0] if headings else None),
                        blocked_headings=backtrack_blocked_headings,
                    )
                    log_fallback_event(
                        "backtrack_apply",
                        current_episode_id,
                        current_step,
                        bt_apply,
                    )
                    if bt_apply.get("applied"):
                        backtrack_action_args = dict(bt_apply["action_args"])
                        backtrack_budget_remaining = bt_apply["budget_after"]
                        backtrack_blocked_headings = set(bt_apply["blocked_headings"])
                        backtrack_reverse_emitted = True
                    else:
                        # rev2 fix #1/#3: a non-appliable backtrack (no reversible move / budget
                        # exhausted) must resolve to the best REAL candidate, never leave the
                        # MOVE_BACK placeholder to cascade into stop_last_resort. None-guard the
                        # ranked fallback (it can return None when nothing ranks).
                        bt_fb = rank_movement_fallback(
                            "backtrack_apply_failed",
                            current_step,
                            observe_dict,
                            visual_evidence_results,
                            instruction,
                            actions,
                            landmarks,
                            source_stage="backtrack_apply",
                            reason=bt_apply.get("reason", "backtrack_not_applied"),
                        )
                        bt_fb_vp = bt_fb.get("selected_candidate") if isinstance(bt_fb, dict) else None
                        if bt_fb_vp is not None and bt_fb_vp in radius_dict and bt_fb_vp in distance_dict:
                            next_vp = bt_fb_vp
                        elif radius_dict:
                            next_vp = list(radius_dict.keys())[0]
                        else:
                            stop_flag = True
                            stop_reason = "Backtrack unavailable and no movement candidates remained."
                            forced_termination_source = "no_movement_candidate"
                            next_vp = STOP_CANDIDATE
                if stop_flag:
                    pass
                elif backtrack_reverse_emitted:
                    # Reverse move already emitted by the apply hook; skip normal builder.
                    pass
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
                            forced_termination_source = "empty_action_space"
                            next_vp = STOP_CANDIDATE
                    if stop_flag:
                        pass

                if forced_termination_source is not None:
                    m3_stop_decision = resolve_m3_forced_termination(
                        forced_termination_source, stop_reason
                    )
                if stop_flag:
                    if m3_stop_decision is None or m3_stop_decision.outcome not in {
                        "commit_goal_stop", "commit_forced_termination"
                    }:
                        raise RuntimeError(
                            "STOP reached ActionCompiler without committed StopDecision"
                        )
                    resolved_action = ResolvedAction(
                        action_type="stop",
                        route_state_id=_contract_state_id,
                        decision_record_id=_contract_decision_id,
                        candidate_id=STOP_CANDIDATE,
                        stop_decision_id=m3_stop_decision.decision_id,
                    )
                elif backtrack_reverse_emitted:
                    resolved_action = ResolvedAction(
                        action_type="backtrack",
                        route_state_id=_contract_state_id,
                        decision_record_id=_contract_decision_id,
                        candidate_id=MOVE_BACK_CANDIDATE,
                        action_args=backtrack_action_args,
                    )
                else:
                    resolved_action = ResolvedAction(
                        action_type="move",
                        route_state_id=_contract_state_id,
                        decision_record_id=_contract_decision_id,
                        candidate_id=str(next_vp),
                        action_args={
                            "angle": radius_dict[next_vp],
                            "distance": distance_dict[next_vp],
                        },
                    )
                _contract_command = action_compiler.compile(
                    command_id=_contract_command_id,
                    resolved_action=resolved_action,
                    stop_decision=m3_stop_decision,
                )
                env_actions = [action_compiler.to_env_action(_contract_command)]
                stop_flag = bool(_contract_command.stop_requested)
                write_navigation_record(
                    "pipeline_action_command",
                    episode_id=current_episode_id,
                    step=current_step,
                    **_contract_command.to_dict()
                )
                if harness_enabled and harness_logger is not None:
                    harness_logger.log_event(
                        "pipeline_action_command",
                        current_step,
                        _contract_command.to_dict(),
                    )

                # Backtracking (hunk 4) capture (rev2 fix #2): record what was actually
                # sent, keyed off the single grep-verified action-4 emitter (env_actions[0]),
                # so it is robust to whichever upstream path (selector / E5 fallback / U3
                # recovery) finalized next_vp. There is exactly ONE action-4 emission site in
                # this loop, so this single capture covers every movement; any future emitter
                # that bypasses this site MUST also update last_executed_move (or set it None).
                if backtrack_enabled and env_actions:
                    _sent = (
                        env_actions[0].get("action", {})
                        if isinstance(env_actions[0], dict)
                        else {}
                    )
                    if backtrack_reverse_emitted:
                        # After a backtrack, the reverse-of-the-reverse would just undo it ->
                        # clear last_executed_move so no MOVE_BACK is offered next step.
                        last_executed_move = None
                        just_backtracked = True
                    elif _sent.get("action") == 4 and isinstance(_sent.get("action_args"), dict):
                        last_executed_move = dict(_sent["action_args"])
                        just_backtracked = False
                    # action 0 (STOP): leave state as-is; the episode terminates on STOP.
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
                # Committed-stop tilt sample. This is the single choke point where the
                # stop is already decided and the current-step observations are still
                # live (envs.step() has not run), so it is the only place that can
                # recognise the terminal pose. Budget-free by design -- see
                # ENDGAME_TILT_VIEW.ALWAYS_ON_STOP.
                if stop_flag and tilt_enabled and tilt_always_on_stop:
                    _tilt_observe("committed_stop", force=True)
                _contract_decision = DecisionRecord(
                    decision_record_id=_contract_decision_id,
                    route_state_id=_contract_state_id,
                    step_id=current_step,
                    progress_provider=(
                        progress_update_contract.provider
                        if progress_update_contract is not None
                        else "unavailable"
                    ),
                    planner_output={
                        "selected_candidate": next_vp,
                        "thought": thought,
                        "stop_reason": stop_reason if stop_flag else None,
                    },
                    stop_proposals=tuple(m3_stop_proposals),
                    override_ledger=tuple(unit_override_records),
                    final_command=_contract_command,
                    stop_evidence_items=tuple(m3_stop_evidence_items),
                    stop_decision=m3_stop_decision,
                    planner_skipped_reason=(
                        "committed_preplanner_stop"
                        if m3_preplanner_stop_decided and stop_flag
                        else None
                    ),
                )
                _contract_decision_payload = run_harness_tool(
                    "pipeline_decision_record_contract",
                    current_step,
                    None,
                    _contract_decision.to_dict,
                )
                if (
                    harness_enabled
                    and harness_logger is not None
                    and _contract_decision_payload is not None
                ):
                    harness_logger.log_event(
                        "pipeline_decision_record",
                        current_step,
                        _contract_decision_payload,
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
                _observable_step_result = {}
                if step_output_summary:
                    for _observable_key in ("done", "collision", "steps_taken"):
                        if _observable_key in step_output_summary[0]:
                            _observable_step_result[_observable_key] = (
                                step_output_summary[0][_observable_key]
                            )
                _contract_receipt = run_harness_tool(
                    "pipeline_action_receipt",
                    current_step,
                    None,
                    build_action_receipt,
                    receipt_id="{}:{}:receipt".format(
                        current_episode_id, current_step
                    ),
                    command_id=_contract_command_id,
                    episode_id=current_episode_id,
                    step_id=current_step,
                    executed_action=env_actions[0],
                    candidate_id=next_vp,
                    step_result=_observable_step_result,
                )
                if _contract_receipt is not None and harness_logger is not None:
                    harness_logger.log_event(
                        "pipeline_action_receipt",
                        current_step,
                        _contract_receipt.to_dict(),
                    )
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
                    if (
                        next_vp != MOVE_BACK_CANDIDATE
                        and _contract_command.action_code == 4
                    ):
                        try:
                            _post_agent_state = envs.call_at(
                                0, "get_agent_info", {}
                            )
                            _post_position = _post_agent_state.get("position")
                            _pre_position = positions[0] if positions else None
                            if _pre_position is not None and _post_position is not None:
                                _actual_displacement = sum(
                                    (float(a) - float(b)) ** 2
                                    for a, b in zip(_post_position, _pre_position)
                                ) ** 0.5
                                _collision = bool(
                                    step_output_summary
                                    and step_output_summary[0].get("collision")
                                )
                                if not _collision and _actual_displacement > 0.05:
                                    episode_forward_translation_m += _actual_displacement
                                    episode_forward_move_steps += 1
                                log_fallback_event(
                                    "route_maturity_odometry",
                                    current_episode_id, current_step,
                                    {
                                        "actual_displacement_m": _actual_displacement,
                                        "collision": _collision,
                                        "counted_forward_step": bool(
                                            not _collision and _actual_displacement > 0.05
                                        ),
                                        "cumulative_actual_displacement_m": (
                                            episode_forward_translation_m
                                        ),
                                        "non_collision_forward_steps": (
                                            episode_forward_move_steps
                                        ),
                                        "position_source": "simulator_agent_pose",
                                    },
                                )
                        except (TypeError, ValueError, KeyError, IndexError):
                            pass
                if not stop_flag:
                    nav_logger.info("========== save history ==========")
                    if next_vp == MOVE_BACK_CANDIDATE:
                        # MOVE_BACK is a synthetic control command, not a camera
                        # candidate. Record it structurally instead of feeding a
                        # fake observation through save_history's direction parser.
                        curr_observe = (
                            "Backtrack executed using the previous reversible movement."
                        )
                        nav_history.append({
                            "step": current_step,
                            "viewpoint": MOVE_BACK_CANDIDATE,
                            "observation": curr_observe,
                            "thought": str(thought or "Backtrack recovery."),
                            "synthetic_action": True,
                        })
                        nav_logger.info("The history at current step is {}".format(nav_history))
                    elif next_vp in observe_dict:
                        curr_observe = observe_dict[next_vp]
                        nav_history = navigator.save_history(
                            nav_logger, current_step, next_vp, thought,
                            curr_observe, nav_history
                        )
                    else:
                        raise KeyError(
                            "selected movement candidate missing observation: {}".format(
                                next_vp
                            )
                        )
                    write_navigation_record(
                        "history_saved",
                        episode_id=current_episode_id,
                        step=current_step,
                        selected_candidate=next_vp,
                        thought=thought,
                        current_observation=curr_observe,
                        synthetic_action=(next_vp == MOVE_BACK_CANDIDATE),
                        nav_history=nav_history,
                    )

                log_route_state(
                    "post_action",
                    selected_candidate=next_vp,
                    stop_requested=stop_flag,
                    stop_reason=stop_reason,
                    action_result={
                        "done": bool(step_output_summary and step_output_summary[0].get("done")),
                        "collision": (step_output_summary[0].get("collision") if step_output_summary else None),
                        "steps_taken": (step_output_summary[0].get("steps_taken") if step_output_summary else None),
                        "stop_committed": bool(stop_flag),
                    },
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
                            # os.replace is atomic against a *process* crash but not
                            # against a host power loss: without this fsync the rename
                            # metadata can reach disk while the data blocks are still in
                            # page cache, leaving a 0-byte checkpoint. That is exactly
                            # how run 20260720_004318 lost 8.8h of stats when the VM
                            # dropped at 09:42 (the per-episode navigation_records were
                            # the only reason those 82 episodes were recoverable).
                            _ckpt_f.flush()
                            os.fsync(_ckpt_f.fileno())
                        os.replace(ckpt_tmp, ckpt_path)
                        # Persist the rename itself; if this is lost the previous
                        # checkpoint survives intact, which is the safe failure mode.
                        _ckpt_dir_fd = os.open(config.RESULTS_DIR, os.O_RDONLY)
                        try:
                            os.fsync(_ckpt_dir_fd)
                        finally:
                            os.close(_ckpt_dir_fd)
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
        # Endgame tilt view: one extra down-pitched RGB+depth pair at forward yaw.
        # Added ONLY when enabled, so a disabled run keeps the exact sensor set of
        # clean_baseline_v1. The 'tilt_' uuid prefix is load-bearing: generate_input
        # indexes rgb/depth keys POSITIONALLY into 1..12 and construct_image_dicts
        # maps those indices to headings, so an extra sensor caught by that sweep
        # would silently rotate the direction mapping. generate_input skips 'tilt_'.
        _tilt_cfg = getattr(self.config.OPENNAV_HARNESS, "ENDGAME_TILT_VIEW", None)
        if _tilt_cfg is not None and bool(getattr(_tilt_cfg, "ENABLED", False)):
            tilt_pitch_rad = float(
                np.radians(float(getattr(_tilt_cfg, "PITCH_DEG", -30.0)))
            )
            for sensor_type in ["RGB", "DEPTH"]:
                resizer_size = dict(resize_config)[sensor_type.lower()]
                camera_config = deepcopy(
                    getattr(config.SIMULATOR, f"{sensor_type}_SENSOR")
                )
                camera_config.ORIENTATION = [tilt_pitch_rad, 0.0, 0.0]
                camera_config.UUID = f"tilt_{sensor_type.lower()}"
                camera_template = f"TILT_{sensor_type}"
                setattr(config.SIMULATOR, camera_template, camera_config)
                config.SIMULATOR.AGENT_0.SENSORS.append(camera_template)
                resize_config.append((camera_config.UUID, resizer_size))

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
