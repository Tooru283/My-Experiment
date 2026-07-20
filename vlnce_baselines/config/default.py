from typing import List, Optional, Union

import habitat_baselines.config.default
from habitat.config.default import CONFIG_FILE_SEPARATOR
from habitat.config.default import Config as CN

from habitat_extensions.config.default import (
    get_extended_config as get_task_config,
)

# -----------------------------------------------------------------------------
# EXPERIMENT CONFIG
# -----------------------------------------------------------------------------
_C = CN()
_C.BASE_TASK_CONFIG_PATH = "habitat_extensions/config/vlnce_task.yaml"
_C.TASK_CONFIG = CN()  # task_config will be stored as a config node
_C.TRAINER_NAME = "dagger"
_C.ENV_NAME = "VLNCEDaggerEnv"
_C.SIMULATOR_GPU_IDS = [0]
_C.VIDEO_OPTION = []  # options: "disk", "tensorboard"
_C.VIDEO_DIR = "videos/debug"
_C.TENSORBOARD_DIR = "data/tensorboard_dirs/debug"
_C.RESULTS_DIR = "data/checkpoints/pretrained/evals"

# -----------------------------------------------------------------------------
# EVAL CONFIG
# -----------------------------------------------------------------------------
_C.EVAL = CN()
# The split to evaluate on
_C.EVAL.SPLIT = "val_seen"
_C.EVAL.EPISODE_COUNT = -1
_C.EVAL.LANGUAGES = ["en-US", "en-IN"]
_C.EVAL.SAMPLE = False
_C.EVAL.SAVE_RESULTS = True
_C.EVAL.EVAL_NONLEARNING = False
_C.EVAL.NONLEARNING = CN()
_C.EVAL.NONLEARNING.AGENT = "RandomAgent"

# -----------------------------------------------------------------------------
# OPEN-NAV HARNESS CONFIG
# -----------------------------------------------------------------------------
_C.OPENNAV_HARNESS = CN()
_C.OPENNAV_HARNESS.ENABLED = False
_C.OPENNAV_HARNESS.ENABLE_HARNESS_LOGGING = False
_C.OPENNAV_HARNESS.ENABLE_DECISION_EFFECT = False
_C.OPENNAV_HARNESS.TRACE_DIR = "logs/harness_traces"
_C.OPENNAV_HARNESS.TRACE_FORMAT = "jsonl"
_C.OPENNAV_HARNESS.LOG_IMAGES = False
_C.OPENNAV_HARNESS.FAIL_OPEN = True

_C.OPENNAV_HARNESS.GEOMETRY_QUERY = CN()
_C.OPENNAV_HARNESS.GEOMETRY_QUERY.ENABLED = True
_C.OPENNAV_HARNESS.GEOMETRY_QUERY.LOG_ONLY = True

_C.OPENNAV_HARNESS.GROUNDER_DIAGNOSTIC = CN()
_C.OPENNAV_HARNESS.GROUNDER_DIAGNOSTIC.ENABLED = True
_C.OPENNAV_HARNESS.GROUNDER_DIAGNOSTIC.LOG_ONLY = True

_C.OPENNAV_HARNESS.MEMORY_DIAGNOSTIC = CN()
_C.OPENNAV_HARNESS.MEMORY_DIAGNOSTIC.ENABLED = True
_C.OPENNAV_HARNESS.MEMORY_DIAGNOSTIC.LOG_ONLY = True

_C.OPENNAV_HARNESS.CONTEXT_BUILDER = CN()
_C.OPENNAV_HARNESS.CONTEXT_BUILDER.ENABLED = True
_C.OPENNAV_HARNESS.CONTEXT_BUILDER.LOG_ONLY = True

_C.OPENNAV_HARNESS.ORACLE_METRICS = CN()
_C.OPENNAV_HARNESS.ORACLE_METRICS.ENABLED = True
_C.OPENNAV_HARNESS.ORACLE_METRICS.LOG_ONLY = True

_C.OPENNAV_HARNESS.VISUAL_EVIDENCE = CN()
_C.OPENNAV_HARNESS.VISUAL_EVIDENCE.ENABLED = False
_C.OPENNAV_HARNESS.VISUAL_EVIDENCE.LOG_ONLY = True
_C.OPENNAV_HARNESS.VISUAL_EVIDENCE.BASE_URL = "http://127.0.0.1:23333/v1"
_C.OPENNAV_HARNESS.VISUAL_EVIDENCE.MODEL = "/root/models/Qwen3.5-4B"
_C.OPENNAV_HARNESS.VISUAL_EVIDENCE.MAX_CANDIDATES = 4
_C.OPENNAV_HARNESS.VISUAL_EVIDENCE.MAX_IMAGE_EDGE = 384
_C.OPENNAV_HARNESS.VISUAL_EVIDENCE.MAX_TOKENS = 1024
_C.OPENNAV_HARNESS.VISUAL_EVIDENCE.TIMEOUT_SECONDS = 60.0
_C.OPENNAV_HARNESS.VISUAL_EVIDENCE.IMAGE_JPEG_QUALITY = 80
_C.OPENNAV_HARNESS.VISUAL_EVIDENCE.METADATA_OBSERVATION_CHARS = 700
_C.OPENNAV_HARNESS.VISUAL_EVIDENCE.COMPACT_JSON = False

_C.OPENNAV_HARNESS.VISUAL_TARGET_VERIFIER = CN()
_C.OPENNAV_HARNESS.VISUAL_TARGET_VERIFIER.ENABLED = False
_C.OPENNAV_HARNESS.VISUAL_TARGET_VERIFIER.LOG_ONLY = True
_C.OPENNAV_HARNESS.VISUAL_TARGET_VERIFIER.CONFIDENCE_THRESHOLD = 0.6
_C.OPENNAV_HARNESS.VISUAL_TARGET_VERIFIER.REQUIRE_ARRIVAL_EVIDENCE = True
_C.OPENNAV_HARNESS.VISUAL_TARGET_VERIFIER.REJECT_ON_MISSING_FINAL_LANDMARKS = True
_C.OPENNAV_HARNESS.VISUAL_TARGET_VERIFIER.REJECT_ON_UNCERTAIN = True
_C.OPENNAV_HARNESS.VISUAL_TARGET_VERIFIER.MIN_STEPS_BEFORE_ALLOW = 0
_C.OPENNAV_HARNESS.VISUAL_TARGET_VERIFIER.REQUIRE_FULL_COVERAGE_FOR_ALLOW = False
_C.OPENNAV_HARNESS.VISUAL_TARGET_VERIFIER.BLOCK_GENERIC_FINAL_TERMS_FOR_ALLOW = True
_C.OPENNAV_HARNESS.VISUAL_TARGET_VERIFIER.REQUIRE_COMPLETION_FOR_SELECTOR_STOP = True
_C.OPENNAV_HARNESS.VISUAL_TARGET_VERIFIER.REQUIRE_CURRENT_VIEW_TEXT_CORROBORATION_FOR_ALLOW = False

# Endgame tilt view (staged, default OFF -> a disabled run stays byte-identical to
# clean_baseline_v1: no sensor is added and no hook fires).
#
# Why: get_camera_orientations() pitches every one of the 12 rig cameras at 0.0, so
# the agent has full 360 yaw coverage and ZERO vertical coverage. With ~1.25m eye
# height and a 90 deg VFOV, floor closer than ~1.25m falls outside every frame. A
# large share of R2R goals are floor-level or low furniture (rug / bed side / stairs
# / chair), so part of the documented 46% near-distance under-detection is geometric
# -- the target is not in any image, not mis-read.
#
# Cost control: stop_verification fires ~1.55x per step (1008 events / 650 steps in
# clean_baseline_v1), so an ungated hook would roughly double VLM cost. The tilt view
# is observed ONLY when the endgame is armed AND a stop is proposed, then rate-limited.
_C.OPENNAV_HARNESS.ENDGAME_TILT_VIEW = CN()
_C.OPENNAV_HARNESS.ENDGAME_TILT_VIEW.ENABLED = False
_C.OPENNAV_HARNESS.ENDGAME_TILT_VIEW.LOG_ONLY = True
_C.OPENNAV_HARNESS.ENDGAME_TILT_VIEW.PITCH_DEG = -30.0
# Arm conditions -- any one arms the endgame. All are oracle-free: instruction
# structure, the agent's own perception, and step count. Deliberately NOT keyed on
# phase_evidence.phase: phase_evidence.py branches "recover" on recent_distance_gains
# (= simulator geodesic goal-distance deltas = GT). completed/action_count come from
# _completed_action_count(estimation, actions), which is text-only and clean.
#
# ARM_ON_FINAL_CLAUSE defaults OFF as of the 20260719 ep100 measurement: the LLM's
# self-reported completed_action_count is already at action_count-1 on step 1 in
# 81/100 episodes (median 4 of 5 "done" before the agent has moved), so the signal
# is clean of oracle but is not actually informative about the endgame. Leaving it
# on burned the whole per-episode budget in steps 1-3 and covered the terminal stop
# in only 21/100 episodes.
_C.OPENNAV_HARNESS.ENDGAME_TILT_VIEW.ARM_ON_FINAL_CLAUSE = False
_C.OPENNAV_HARNESS.ENDGAME_TILT_VIEW.ARM_ON_FINAL_LANDMARK = True
_C.OPENNAV_HARNESS.ENDGAME_TILT_VIEW.ARM_STEP_FRAC = 0.7
_C.OPENNAV_HARNESS.ENDGAME_TILT_VIEW.MAX_PER_EPISODE = 4
# Minimum steps between two in-episode fires. Together with MAX_PER_EPISODE this
# spreads the budget over the episode instead of spending it on the first gates.
_C.OPENNAV_HARNESS.ENDGAME_TILT_VIEW.MIN_STEP_GAP = 3
# Fire once at the committed stop, outside the budget. The stop pose is the sample
# the anchor threshold has to be calibrated on, and it cannot be recognised in
# advance -- only the commit point knows the episode is ending. Costs +1 call/episode
# and takes terminal coverage from 21/100 to 100/100.
_C.OPENNAV_HARNESS.ENDGAME_TILT_VIEW.ALWAYS_ON_STOP = True
_C.OPENNAV_HARNESS.ENDGAME_TILT_VIEW.DEDUPE_DIST_M = 0.5
_C.OPENNAV_HARNESS.ENDGAME_TILT_VIEW.DEDUPE_HEADING_DEG = 15.0
_C.OPENNAV_HARNESS.ENDGAME_TILT_VIEW.CENTER_FRAC = 0.25

# Arm C -- observable-feature prior over waypoint candidates.
# Fixed linear model, weights fit offline against an oracle label (which candidate is
# closest to the goal) on ep100_series_m420260719_001700. Legal offline supervision:
# inference reads only distance / turn angle / raw_rank / candidate count / number of
# matched instruction terms, never a goal distance.
# Out-of-fold (GroupKFold-5 by episode, 5 seeds): 44.1% +/- 1.6 vs LLM selector 36.7%
# vs best single heuristic 35.7% vs random 27.1%.
# LOG_ONLY=True computes and logs scores into the candidate records without touching
# the order handed to the selector; flip to False to present candidates prior-first.
_C.OPENNAV_HARNESS.CANDIDATE_PRIOR = CN()
_C.OPENNAV_HARNESS.CANDIDATE_PRIOR.ENABLED = False
_C.OPENNAV_HARNESS.CANDIDATE_PRIOR.LOG_ONLY = True
# Override gate when LOG_ONLY=False. The prior can only reach the decision by
# overriding the selector's emitted choice: the prompt's candidate ordering carries
# no position bias (first-slot pick 27.1% vs 25.8% uniform, flat across slots), so
# reranking the list is a no-op, and prompt-side geometry injection was already
# measured net-negative. Gate is the softmax probability gap between the prior's top
# candidate and the selector's pick.
#
# REVISED 20260720: default lowered 0.10 -> 0.00. The earlier note here read
# "0.10 buys 96% of the benefit for half the intervention" off the scan
# 41.8 / 41.5 / 39.5. Re-run out-of-fold (GroupKFold-5 by episode, 5 seeds, weights
# refit per fold, n=656 decision points on the fitting run):
#
#   MIN_PROB_GAP   0.00      0.05      0.10      0.15      0.20      0.30
#   out-of-fold   43.8+-.6  44.2+-.8  42.5+-.4  39.7+-.5  39.2+-.6  38.0+-.2
#   in-sample      45.3      44.6      42.7       --       40.1       --
#   (LLM selector alone 36.7%, random 25.8%)
#
# 0.00 and 0.05 tie within seed noise; both beat 0.10 by more than the seed spread,
# so the gate costs accuracy and the free parameter is dropped. Note the in-sample
# row runs ~1.5 points optimistic -- always read the out-of-fold row.
#
# Two caveats kept deliberately: (a) the out-of-fold 43.8 does not reconcile with the
# previously recorded 41.8 for full override; the fitting protocol here (hand-rolled
# logistic, 600 iters, lr 0.3, standardized) is not provably the one used then, and
# the discrepancy is unexplained. (b) The measurement run is a near-duplicate of the
# fitting run (temperature 0), so this is NOT an independent test set.
#
# Candidate world position, for any future candidate-level analysis:
#     theta = -heading - angle_rad;  x + d*sin(theta),  z - d*cos(theta)
# validated at 0.000 m median error against the next step's recorded position.
# STOP / MOVE_BACK are never overridden -- the prior has no termination label.
_C.OPENNAV_HARNESS.CANDIDATE_PRIOR.MIN_PROB_GAP = 0.00

# Progress estimator mode. "llm" calls the text LLM every step (15.7 s, 29% of a
# full ep100 run); "rule" derives the executed-action list from the step counter at
# zero cost.
#
# Measured on ep100_series_m420260719_130559 (n=635 steps with usable progress):
# correlation between the reported progress fraction and true progress toward goal --
#     LLM estimation      r = 0.122      rule k=min(step, a)   r = 0.325
# and, controlling for step number (within-step, weighted over steps 1-10),
# the LLM contributes  r = +0.037  -- i.e. nearly all of its apparent signal is the
# step counter, which is free. It turns negative from step 6 on (step 9: -0.208).
# Failure modes behind that number: 69/94 episodes report the instruction 100%
# complete at step 1 before moving; 414/687 steps saturate at 1.0; 18/93 episodes
# emit a byte-identical string all episode; and the executed-action count DECREASES
# between adjacent steps 75 times, which is impossible and shows it re-invents the
# answer each step rather than tracking state.
#
# This matters beyond cost: the estimate drives 47.7% of phase assignments, so
# 70/95 episodes enter phase "verify" at step 1 -- the state machine starts at its
# terminal state. See docs/experiment_record_20260719.md section 10.
_C.OPENNAV_HARNESS.COMPLETION_ESTIMATION = CN()
_C.OPENNAV_HARNESS.COMPLETION_ESTIMATION.MODE = "llm"

_C.OPENNAV_HARNESS.VISUAL_EVIDENCE_MEMORY = CN()
_C.OPENNAV_HARNESS.VISUAL_EVIDENCE_MEMORY.ENABLED = False
_C.OPENNAV_HARNESS.VISUAL_EVIDENCE_MEMORY.LOG_ONLY = True
_C.OPENNAV_HARNESS.VISUAL_EVIDENCE_MEMORY.MAX_HISTORY = 64
_C.OPENNAV_HARNESS.VISUAL_EVIDENCE_MEMORY.MAX_NOTES_CHARS = 240

_C.OPENNAV_HARNESS.MULTIMODAL_SELECTOR_CONTEXT = CN()
_C.OPENNAV_HARNESS.MULTIMODAL_SELECTOR_CONTEXT.ENABLED = False
_C.OPENNAV_HARNESS.MULTIMODAL_SELECTOR_CONTEXT.LOG_ONLY = True
_C.OPENNAV_HARNESS.MULTIMODAL_SELECTOR_CONTEXT.MAX_SUMMARY_CHARS = 200
_C.OPENNAV_HARNESS.MULTIMODAL_SELECTOR_CONTEXT.INCLUDE_MEMORY_SUFFIX = False
_C.OPENNAV_HARNESS.MULTIMODAL_SELECTOR_CONTEXT.DECISION_MODE = "phase_gated_u1"
_C.OPENNAV_HARNESS.MULTIMODAL_SELECTOR_CONTEXT.APPLY_PHASES = [
    "search",
    "approach",
]
_C.OPENNAV_HARNESS.MULTIMODAL_SELECTOR_CONTEXT.SUPPRESS_TARGET_ARRIVAL_FOR_SELECTOR = True
_C.OPENNAV_HARNESS.MULTIMODAL_SELECTOR_CONTEXT.MIN_CONFIDENCE_FOR_TARGET_HINT = 0.9

_C.OPENNAV_HARNESS.LLM_RUNTIME = CN()
_C.OPENNAV_HARNESS.LLM_RUNTIME.COMPLETION_MAX_TOKENS = 0
_C.OPENNAV_HARNESS.LLM_RUNTIME.NAVIGATOR_MAX_TOKENS = 0
_C.OPENNAV_HARNESS.LLM_RUNTIME.THOUGHT_FUSION_MAX_TOKENS = 0
_C.OPENNAV_HARNESS.LLM_RUNTIME.DECISION_MAX_TOKENS = 0

_C.OPENNAV_HARNESS.NAVIGATION_RUNTIME = CN()
_C.OPENNAV_HARNESS.NAVIGATION_RUNTIME.SHORT_ACTION_STEP_LIMIT = 10
_C.OPENNAV_HARNESS.NAVIGATION_RUNTIME.LONG_ACTION_STEP_LIMIT = 12
_C.OPENNAV_HARNESS.NAVIGATION_RUNTIME.SHORT_ACTION_COUNT_THRESHOLD = 6
_C.OPENNAV_HARNESS.NAVIGATION_RUNTIME.BLOCK_WEAK_FINAL_TARGET_COMPLETION_STOP = True
_C.OPENNAV_HARNESS.NAVIGATION_RUNTIME.MIN_STEPS_FOR_WEAK_FINAL_TARGET_COMPLETION_STOP = 8

_C.OPENNAV_HARNESS.U_SERIES = CN()
_C.OPENNAV_HARNESS.U_SERIES.ENABLED = False
_C.OPENNAV_HARNESS.U_SERIES.DECISION_EFFECT_UNIT = "none"

_C.OPENNAV_HARNESS.U_SERIES.PHASE_EVIDENCE = CN()
_C.OPENNAV_HARNESS.U_SERIES.PHASE_EVIDENCE.ENABLED = False
_C.OPENNAV_HARNESS.U_SERIES.PHASE_EVIDENCE.LOG_ONLY = True
_C.OPENNAV_HARNESS.U_SERIES.PHASE_EVIDENCE.UNKNOWN_CONFIDENCE_THRESHOLD = 0.4
_C.OPENNAV_HARNESS.U_SERIES.PHASE_EVIDENCE.LATE_STEP_THRESHOLD = 4
_C.OPENNAV_HARNESS.U_SERIES.PHASE_EVIDENCE.FAILURE_SIGNAL_TTL_STEPS = 2
_C.OPENNAV_HARNESS.U_SERIES.PHASE_EVIDENCE.CLEAR_FAILURE_ON_POSITIVE_GAIN_COUNT = 2

_C.OPENNAV_HARNESS.U_SERIES.STOP_EVIDENCE_VERIFIER = CN()
_C.OPENNAV_HARNESS.U_SERIES.STOP_EVIDENCE_VERIFIER.ENABLED = False
_C.OPENNAV_HARNESS.U_SERIES.STOP_EVIDENCE_VERIFIER.LOG_ONLY = True
_C.OPENNAV_HARNESS.U_SERIES.STOP_EVIDENCE_VERIFIER.ENABLE_RESCUE = False
_C.OPENNAV_HARNESS.U_SERIES.STOP_EVIDENCE_VERIFIER.ENABLE_RELATION_CHECK = False
_C.OPENNAV_HARNESS.U_SERIES.STOP_EVIDENCE_VERIFIER.ENABLE_WEAK_TARGET_ADJUSTMENT = False
_C.OPENNAV_HARNESS.U_SERIES.STOP_EVIDENCE_VERIFIER.RESCUE_CONFIDENCE_THRESHOLD = 0.95
_C.OPENNAV_HARNESS.U_SERIES.STOP_EVIDENCE_VERIFIER.RESCUE_MIN_STEP = 3
_C.OPENNAV_HARNESS.U_SERIES.STOP_EVIDENCE_VERIFIER.RESCUE_MAX_NON_POSITIVE_GAINS = 1
_C.OPENNAV_HARNESS.U_SERIES.STOP_EVIDENCE_VERIFIER.RESCUE_REQUIRE_POSITIVE_RECENT_GAIN = True
_C.OPENNAV_HARNESS.U_SERIES.STOP_EVIDENCE_VERIFIER.RESCUE_ALLOW_PHASE_VERIFY = True

_C.OPENNAV_HARNESS.U_SERIES.FAILURE_RECOVERY = CN()
_C.OPENNAV_HARNESS.U_SERIES.FAILURE_RECOVERY.ENABLED = False
_C.OPENNAV_HARNESS.U_SERIES.FAILURE_RECOVERY.LOG_ONLY = True
_C.OPENNAV_HARNESS.U_SERIES.FAILURE_RECOVERY.ENABLE_RESELECT = False
_C.OPENNAV_HARNESS.U_SERIES.FAILURE_RECOVERY.MAX_RECOVERY_PER_EPISODE = 2
_C.OPENNAV_HARNESS.U_SERIES.FAILURE_RECOVERY.NEGATIVE_GAIN_WINDOW = 2

# -----------------------------------------------------------------------------
# INFERENCE CONFIG
# -----------------------------------------------------------------------------
_C.INFERENCE = CN()
_C.INFERENCE.SPLIT = "test"
_C.INFERENCE.LANGUAGES = ["en-US", "en-IN"]
_C.INFERENCE.SAMPLE = False
_C.INFERENCE.USE_CKPT_CONFIG = True
_C.INFERENCE.CKPT_PATH = "data/checkpoints/CMA_PM_DA_Aug.pth"
_C.INFERENCE.PREDICTIONS_FILE = "predictions.json"
_C.INFERENCE.INFERENCE_NONLEARNING = False
_C.INFERENCE.NONLEARNING = CN()
_C.INFERENCE.NONLEARNING.AGENT = "RandomAgent"
_C.INFERENCE.FORMAT = "r2r"  # either 'rxr' or 'r2r'
# -----------------------------------------------------------------------------
# IMITATION LEARNING CONFIG
# -----------------------------------------------------------------------------
_C.IL = CN()
_C.IL.lr = 2.5e-4
_C.IL.batch_size = 5
_C.IL.epochs = 4
_C.IL.use_iw = True
# inflection coefficient for RxR training set GT trajectories (guide): 1.9
# inflection coefficient for R2R training set GT trajectories: 3.2
_C.IL.inflection_weight_coef = 3.2
# load an already trained model for fine tuning
_C.IL.load_from_ckpt = False
_C.IL.ckpt_to_load = "data/checkpoints/ckpt.0.pth"
# if True, loads the optimizer state, epoch, and step_id from the ckpt dict.
_C.IL.is_requeue = False
# it True, start training from the saved epoch
# -----------------------------------------------------------------------------
# IL: RXR TRAINER CONFIG
# -----------------------------------------------------------------------------
_C.IL.RECOLLECT_TRAINER = CN()
_C.IL.RECOLLECT_TRAINER.preload_trajectories_file = True
_C.IL.RECOLLECT_TRAINER.trajectories_file = (
    "data/trajectories_dirs/debug/trajectories.json.gz"
)
# if set to a positive int, episodes with longer paths are ignored in training
_C.IL.RECOLLECT_TRAINER.max_traj_len = -1
# if set to a positive int, effective_batch_size must be some multiple of
# IL.batch_size. Gradient accumulation enables an arbitrarily high "effective"
# batch size.
_C.IL.RECOLLECT_TRAINER.effective_batch_size = -1
_C.IL.RECOLLECT_TRAINER.preload_size = 30
_C.IL.RECOLLECT_TRAINER.use_iw = True
_C.IL.RECOLLECT_TRAINER.gt_file = (
    "data/datasets/RxR_VLNCE_v0/{split}/{split}_{role}_gt.json.gz"
)
# -----------------------------------------------------------------------------
# IL: DAGGER CONFIG
# -----------------------------------------------------------------------------
_C.IL.DAGGER = CN()
_C.IL.DAGGER.iterations = 10
_C.IL.DAGGER.update_size = 5000
_C.IL.DAGGER.p = 0.75
_C.IL.DAGGER.expert_policy_sensor = "SHORTEST_PATH_SENSOR"
_C.IL.DAGGER.expert_policy_sensor_uuid = "shortest_path_sensor"
_C.IL.DAGGER.load_space = False
# if True, load saved observation space and action space
_C.IL.DAGGER.lmdb_map_size = 1.0e12
# if True, saves data to disk in fp16 and converts back to fp32 when loading.
_C.IL.DAGGER.lmdb_fp16 = False
# How often to commit the writes to the DB, less commits is
# better, but everything must be in memory until a commit happens/
_C.IL.DAGGER.lmdb_commit_frequency = 500
# If True, load precomputed features directly from lmdb_features_dir.
_C.IL.DAGGER.preload_lmdb_features = False
_C.IL.DAGGER.lmdb_features_dir = (
    "data/trajectories_dirs/debug/trajectories.lmdb"
)
# -----------------------------------------------------------------------------
# RL CONFIG
# -----------------------------------------------------------------------------
_C.RL = CN()
_C.RL.POLICY = CN()
_C.RL.POLICY.OBS_TRANSFORMS = CN()
_C.RL.POLICY.OBS_TRANSFORMS.ENABLED_TRANSFORMS = [
    "CenterCropperPerSensor",
]
_C.RL.POLICY.OBS_TRANSFORMS.CENTER_CROPPER_PER_SENSOR = CN()
_C.RL.POLICY.OBS_TRANSFORMS.CENTER_CROPPER_PER_SENSOR.SENSOR_CROPS = [
    ("rgb", (224, 224)),
    ("depth", (256, 256)),
]
_C.RL.POLICY.OBS_TRANSFORMS.RESIZER_PER_SENSOR = CN()
_C.RL.POLICY.OBS_TRANSFORMS.RESIZER_PER_SENSOR.SIZES = [
    ("rgb", (224, 224)),
    ("depth", (256, 256)),
]
# -----------------------------------------------------------------------------
# MODELING CONFIG
# -----------------------------------------------------------------------------
_C.MODEL = CN()
_C.MODEL.policy_name = "CMAPolicy"  # or "Seq2SeqPolicy"
_C.MODEL.ablate_depth = False
_C.MODEL.ablate_rgb = False
_C.MODEL.ablate_instruction = False

_C.MODEL.INSTRUCTION_ENCODER = CN()
_C.MODEL.INSTRUCTION_ENCODER.sensor_uuid = "instruction"
_C.MODEL.INSTRUCTION_ENCODER.vocab_size = 2504
_C.MODEL.INSTRUCTION_ENCODER.use_pretrained_embeddings = True
_C.MODEL.INSTRUCTION_ENCODER.embedding_file = (
    "data/datasets/R2R_VLNCE_v1-2_preprocessed/embeddings.json.gz"
)
_C.MODEL.INSTRUCTION_ENCODER.dataset_vocab = (
    "data/datasets/R2R_VLNCE_v1-2_preprocessed/train/train.json.gz"
)
_C.MODEL.INSTRUCTION_ENCODER.fine_tune_embeddings = False
_C.MODEL.INSTRUCTION_ENCODER.embedding_size = 50
_C.MODEL.INSTRUCTION_ENCODER.hidden_size = 128
_C.MODEL.INSTRUCTION_ENCODER.rnn_type = "LSTM"
_C.MODEL.INSTRUCTION_ENCODER.final_state_only = True
_C.MODEL.INSTRUCTION_ENCODER.bidirectional = False

_C.MODEL.spatial_output = True
_C.MODEL.RGB_ENCODER = CN()
_C.MODEL.RGB_ENCODER.cnn_type = "TorchVisionResNet50"
_C.MODEL.RGB_ENCODER.output_size = 256

_C.MODEL.DEPTH_ENCODER = CN()
_C.MODEL.DEPTH_ENCODER.cnn_type = "VlnResnetDepthEncoder"
_C.MODEL.DEPTH_ENCODER.output_size = 128
# type of resnet to use
_C.MODEL.DEPTH_ENCODER.backbone = "resnet50"
# path to DDPPO resnet weights
_C.MODEL.DEPTH_ENCODER.ddppo_checkpoint = (
    "data/ddppo-models/gibson-2plus-resnet50.pth"
)

_C.MODEL.STATE_ENCODER = CN()
_C.MODEL.STATE_ENCODER.hidden_size = 512
_C.MODEL.STATE_ENCODER.rnn_type = "GRU"

_C.MODEL.SEQ2SEQ = CN()
_C.MODEL.SEQ2SEQ.use_prev_action = False

_C.MODEL.PROGRESS_MONITOR = CN()
_C.MODEL.PROGRESS_MONITOR.use = False
_C.MODEL.PROGRESS_MONITOR.alpha = 1.0  # loss multiplier


def purge_keys(config: CN, keys: List[str]) -> None:
    for k in keys:
        del config[k]
        config.register_deprecated_key(k)


def get_config(
    config_paths: Optional[Union[List[str], str]] = None,
    opts: Optional[list] = None,
) -> CN:
    r"""Create a unified config with default values. Initialized from the
    habitat_baselines default config. Overwritten by values from
    `config_paths` and overwritten by options from `opts`.
    Args:
        config_paths: List of config paths or string that contains comma
        separated list of config paths.
        opts: Config options (keys, values) in a list (e.g., passed from
        command line into the config. For example, `opts = ['FOO.BAR',
        0.5]`. Argument can be used for parameter sweeping or quick tests.
    """
    config = CN()
    config.merge_from_other_cfg(habitat_baselines.config.default._C)
    purge_keys(config, ["SIMULATOR_GPU_ID", "TEST_EPISODE_COUNT"])
    config.merge_from_other_cfg(_C.clone())

    if config_paths:
        if isinstance(config_paths, str):
            if CONFIG_FILE_SEPARATOR in config_paths:
                config_paths = config_paths.split(CONFIG_FILE_SEPARATOR)
            else:
                config_paths = [config_paths]

        prev_task_config = ""
        for config_path in config_paths:
            config.merge_from_file(config_path)
            if config.BASE_TASK_CONFIG_PATH != prev_task_config:
                config.TASK_CONFIG = get_task_config(
                    config.BASE_TASK_CONFIG_PATH
                )
                prev_task_config = config.BASE_TASK_CONFIG_PATH

    if opts:
        config.CMD_TRAILING_OPTS = opts
        config.merge_from_list(opts)

    config.freeze()
    return config
