#!/usr/bin/env python3
import os
import hashlib
import json
import re
import torch
import random
import argparse
import numpy as np
from datetime import datetime
from habitat import logger
import habitat_extensions  # noqa: F401
import vlnce_baselines     # noqa: F401
from vlnce_baselines.config.default import get_config
from habitat_baselines.common.baseline_registry import baseline_registry


def _episode_group_name(episode_count) -> str:
    try:
        value = int(episode_count)
    except (TypeError, ValueError):
        return "ep_unknown"
    if value < 0:
        return "ep_all"
    return "ep{}".format(value)


def _apply_run_log_layout(base_dir: str, episode_group: str, run_date: str) -> str:
    return os.path.join(str(base_dir).rstrip(os.sep), episode_group, run_date)


def _apply_trace_log_layout(trace_dir: str, episode_group: str, run_date: str) -> str:
    trace_root = os.path.join("logs", "harness_traces")
    trace_dir = str(trace_dir).rstrip(os.sep)
    if trace_dir == trace_root:
        return os.path.join(trace_root, episode_group, run_date)
    trace_prefix = trace_root + os.sep
    if trace_dir.startswith(trace_prefix):
        variant = trace_dir[len(trace_prefix) :]
        return os.path.join(trace_root, episode_group, run_date, variant)
    return os.path.join(trace_dir, episode_group, run_date)



def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _write_run_manifest(config, exp_config: str, opts) -> dict:
    """Persist the resolved, secret-redacted config before an evaluation starts."""
    resolved_config = re.sub(
        r"(?m)^(\s*(?:API_KEY|DASHSCOPE_API_KEY)\s*:\s*).*$",
        r"\1<redacted>",
        config.dump(),
    )
    config_path = os.path.abspath(exp_config)
    try:
        with open(config_path, "rb") as config_file:
            source_hash = _sha256_bytes(config_file.read())
    except OSError:
        source_hash = None

    result_dir = os.path.abspath(config.RESULTS_DIR)
    os.makedirs(result_dir, exist_ok=True)
    snapshot_path = os.path.join(result_dir, "resolved_config.yaml")
    with open(snapshot_path, "w", encoding="utf-8") as snapshot_file:
        snapshot_file.write(resolved_config)

    resolved_hash = _sha256_bytes(resolved_config.encode("utf-8"))
    harness = config.OPENNAV_HARNESS
    manifest = {
        "schema_version": "opennav.run_manifest.v1",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "exp_config": config_path,
        "exp_config_sha256": source_hash,
        "resolved_config": snapshot_path,
        "resolved_config_sha256": resolved_hash,
        "command_line_overrides": list(opts or []),
        "model": {
            "llm": config.LLM,
            "visual_evidence_model": os.environ.get("OPENNAV_LLM_MODEL"),
            "visual_evidence_base_url": os.environ.get("OPENNAV_LLM_BASE_URL"),
        },
        "evaluation": {
            "split": config.EVAL.SPLIT,
            "episode_count": config.EVAL.EPISODE_COUNT,
            "seed": config.TASK_CONFIG.SEED,
            "task_config": config.BASE_TASK_CONFIG_PATH,
        },
        "harness": {
            "trace_dir": harness.TRACE_DIR,
            "progress_provider": harness.PIPELINE.PROGRESS_PROVIDER,
            "stop_policy": harness.PIPELINE.STOP_POLICY,
            "action_compiler": harness.PIPELINE.ACTION_COMPILER,
            "decision_effect_enabled": harness.ENABLE_DECISION_EFFECT,
            "geometry_injection": harness.GEOMETRY_INJECTION,
            "min_steps_before_allow": harness.VISUAL_TARGET_VERIFIER.MIN_STEPS_BEFORE_ALLOW,
            "short_action_step_limit": harness.NAVIGATION_RUNTIME.SHORT_ACTION_STEP_LIMIT,
            "long_action_step_limit": harness.NAVIGATION_RUNTIME.LONG_ACTION_STEP_LIMIT,
            "progress_locator_log_only": harness.PROGRESS_LOCATOR.LOG_ONLY,
            "terminal_gate_log_only": harness.TERMINAL_GATE.LOG_ONLY,
            "phase_evidence_log_only": harness.U_SERIES.PHASE_EVIDENCE.LOG_ONLY,
        },
    }
    manifest_path = os.path.join(result_dir, "run_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as manifest_file:
        json.dump(manifest, manifest_file, ensure_ascii=False, indent=2, sort_keys=True)
        manifest_file.write("\n")

    os.environ["OPENNAV_CONFIG_SNAPSHOT"] = snapshot_path
    os.environ["OPENNAV_CONFIG_SHA256"] = resolved_hash
    os.environ["OPENNAV_EXP_CONFIG_SHA256"] = source_hash or ""
    return manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--exp_name",
        type=str,
        default="test",
        required=True,
        help="experiment id that matches to exp-id in Notion log",
    )
    parser.add_argument(
        "--exp-config",
        type=str,
        required=True,
        help="path to config yaml containing info about experiment",
    )
    parser.add_argument(
        "opts",
        default=None,
        nargs=argparse.REMAINDER,
        help="Modify config options from command line",
    )
    parser.add_argument('--local_rank', type=int, default=0, help="local gpu id")
    parser.add_argument(
        "--llm",
        type=str,
        required=True,
        help="The LLM model to be used (e.g., gpt-4o-2024-08-06, Qwen/Qwen3.5-4B)",
    )
    parser.add_argument(
        "--api_key",
        type=str,
        required=True,
        help="API key for accessing the LLM service",
    )
    args = parser.parse_args()
    run_exp(**vars(args))
    
def run_exp(exp_name: str, exp_config: str, 
            opts=None, local_rank=None,
            llm: str = None, api_key: str = None, episodes_to_load: int = 0) -> None:
    r"""Runs experiment given mode and config

    Args:
        exp_config: path to config file.
        run_type: "train" or "eval.
        opts: list of strings of additional config options.
        llm: The LLM model to be used (e.g., gpt-4o-2024-08-06).
        api_key: API key for accessing the LLM service.
    Returns:
        None.
    """
    config = get_config(exp_config, opts)
    config.defrost()

    run_date = datetime.now().strftime("%Y%m%d")
    episode_group = _episode_group_name(config.EVAL.EPISODE_COUNT)
    os.environ["OPENNAV_RUN_DATE"] = run_date
    os.environ["OPENNAV_EPISODE_GROUP"] = episode_group

    config.CHECKPOINT_FOLDER += exp_name
    if os.path.isdir(config.EVAL_CKPT_PATH_DIR):
        config.EVAL_CKPT_PATH_DIR += exp_name
    config.RESULTS_DIR = os.path.join(
        _apply_run_log_layout(config.RESULTS_DIR, episode_group, run_date),
        exp_name,
    )
    config.OPENNAV_HARNESS.TRACE_DIR = _apply_trace_log_layout(
        config.OPENNAV_HARNESS.TRACE_DIR,
        episode_group,
        run_date,
    )
    config.LOG_FILE = exp_name + '_' + config.LOG_FILE

    config.TASK_CONFIG.SEED = 0

    config.local_rank = local_rank

    if llm is not None:
        config.LLM = llm
    if api_key is not None:
        config.API_KEY = api_key

    _write_run_manifest(config, exp_config, opts)

    config.freeze()
    
    # Check if the 'logs/running_log' directory exists; if not, create it
    log_dir = _apply_run_log_layout("logs/running_log", episode_group, run_date)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # Add the file handler for logging
    logger.add_filehandler(os.path.join(log_dir, config.LOG_FILE))

    random.seed(config.TASK_CONFIG.SEED)
    np.random.seed(config.TASK_CONFIG.SEED)
    torch.manual_seed(config.TASK_CONFIG.SEED)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = False
    if torch.cuda.is_available():
        torch.set_num_threads(1)

    trainer_init = baseline_registry.get_trainer(config.TRAINER_NAME)
    assert trainer_init is not None, f"{config.TRAINER_NAME} is not supported"
    trainer = trainer_init(config)
    trainer.eval()
    
if __name__ == "__main__":
    __spec__ = None 
    main()
