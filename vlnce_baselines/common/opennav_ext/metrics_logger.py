import json
import os
import re
import time
import traceback
from datetime import datetime
from dataclasses import asdict, is_dataclass
from typing import Any, Dict, Optional

try:
    import numpy as np
except Exception:  # pragma: no cover - optional dependency guard
    np = None

try:
    import torch
except Exception:  # pragma: no cover - optional dependency guard
    torch = None


def to_jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return to_jsonable(asdict(value))
    if torch is not None and isinstance(value, torch.Tensor):
        value = value.detach().cpu()
        if value.numel() == 1:
            return to_jsonable(value.item())
        return to_jsonable(value.tolist())
    if np is not None:
        if isinstance(value, np.ndarray):
            return to_jsonable(value.tolist())
        if isinstance(value, np.generic):
            return to_jsonable(value.item())
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value))


class MetricsLogger:
    schema_version = "a1.trace.v1"

    def __init__(
        self,
        trace_dir: str,
        run_id: str,
        rank: int = 0,
        fail_open: bool = True,
        logger: Optional[Any] = None,
    ) -> None:
        self.trace_dir = trace_dir
        self.run_id = run_id
        self.rank = rank
        self.fail_open = fail_open
        self.logger = logger
        self.active_episode_id = None
        self.active_split = "unknown"
        self.active_path = None
        self.disabled = False
        self.run_dir = os.path.join(self.trace_dir, _safe_name(self.run_id))
        self._ensure_dir(self.run_dir)

    def start_episode(
        self,
        episode_id: str,
        split: str,
        instruction: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        if self.disabled:
            return
        self.active_episode_id = str(episode_id)
        self.active_split = str(split)
        episode_dir = os.path.join(
            self.run_dir, _safe_name(split), "rank_{}".format(self.rank)
        )
        self._ensure_dir(episode_dir)
        if self.disabled:
            return
        self.active_path = os.path.join(
            episode_dir, "{}.jsonl".format(_safe_name(self.active_episode_id))
        )
        self._truncate_active_episode()
        self.log_event(
            "episode_start",
            0,
            {
                "episode_id": self.active_episode_id,
                "split": self.active_split,
                "instruction": instruction,
                "metadata": metadata or {},
            },
        )

    def log_event(
        self, event_type: str, step_id: int, payload: Optional[Dict[str, Any]]
    ) -> None:
        if self.disabled:
            return
        try:
            event = {
                "schema_version": self.schema_version,
                "run_id": self.run_id,
                "episode_id": self.active_episode_id,
                "step_id": step_id,
                "event_type": event_type,
                "timestamp": time.time(),
                "timestamp_iso": datetime.now().isoformat(timespec="seconds"),
                "payload": to_jsonable(payload or {}),
            }
            self._write(event)
        except Exception as exc:
            self._handle_error("Harness trace event failed: {}".format(exc))

    def log_tool_failure(
        self,
        tool_name: str,
        step_id: int,
        error: Exception,
        context: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.log_event(
            "tool_failure",
            step_id,
            {
                "tool_name": tool_name,
                "error_type": type(error).__name__,
                "error": str(error),
                "traceback": traceback.format_exc(limit=3),
                "context": context or {},
            },
        )

    def end_episode(
        self, episode_id: str, metrics: Dict[str, Any], step_id: int = 0
    ) -> None:
        self.log_event(
            "episode_end",
            step_id,
            {"episode_id": str(episode_id), "metrics": metrics},
        )
        self.active_episode_id = None
        self.active_path = None

    def _ensure_dir(self, path: str) -> None:
        try:
            os.makedirs(path, exist_ok=True)
        except Exception as exc:
            self._handle_error("Harness trace directory failed: {}".format(exc))

    def _truncate_active_episode(self) -> None:
        try:
            with open(self.active_path, "w", encoding="utf-8"):
                pass
        except Exception as exc:
            self._handle_error("Harness trace truncate failed: {}".format(exc))

    def _handle_error(self, message: str) -> None:
        if self.logger is not None:
            self.logger.info(message)
        if self.fail_open:
            self.disabled = True
            return
        raise RuntimeError(message)

    def _write(self, event: Dict[str, Any]) -> None:
        if self.disabled:
            return
        path = self.active_path
        if path is None:
            path = os.path.join(
                self.run_dir, "rank_{}_run.jsonl".format(self.rank)
            )
        try:
            with open(path, "a", encoding="utf-8") as file:
                file.write(json.dumps(event, ensure_ascii=False) + "\n")
        except Exception as exc:
            self._handle_error("Harness trace write failed: {}".format(exc))
