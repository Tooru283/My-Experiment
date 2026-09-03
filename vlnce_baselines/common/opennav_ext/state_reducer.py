"""Episode-scoped owner for L0/L1/M1/M2/L4 route-state modules.

M1 centralizes lifecycle and mutation ownership while preserving the legacy event and
call order.  It intentionally does not choose actions or change decision permissions.
"""

from typing import Any, Dict, Iterable, Optional

from vlnce_baselines.common.opennav_ext.anchor_chain import build_anchor_chain


STATE_REDUCER_SCHEMA_VERSION = "opennav.state_reducer.v1"


class RouteStateReducer:
    def __init__(
        self,
        *,
        anchor_chain_enabled: bool,
        progress_locator: Any = None,
        spatial_graph: Any = None,
        landmark_pool: Any = None,
        terminal_gate: Any = None,
    ) -> None:
        self.anchor_chain_enabled = bool(anchor_chain_enabled)
        self.progress_locator = progress_locator
        self.spatial_graph = spatial_graph
        self.landmark_pool = landmark_pool
        self.terminal_gate = terminal_gate
        self.episode_id: Optional[str] = None
        self.plan: Dict[str, Any] = {}

    def _require_episode(self) -> None:
        if self.episode_id is None:
            raise RuntimeError("RouteStateReducer must be reset before step updates")

    def reset_episode(
        self, episode_id: Any, actions: Any, landmarks: Any
    ) -> Dict[str, Any]:
        """Reset every owned mutable module exactly once for a new episode."""
        episode_key = str(episode_id)
        if self.episode_id == episode_key:
            return self.plan
        self.episode_id = episode_key
        self.plan = (
            build_anchor_chain(actions, landmarks)
            if self.anchor_chain_enabled
            else {}
        )
        anchors = self.plan.get("anchors", []) if isinstance(self.plan, dict) else []
        if self.progress_locator is not None:
            self.progress_locator.reset_episode(anchors)
        if self.spatial_graph is not None:
            self.spatial_graph.reset_episode()
        if self.landmark_pool is not None:
            vocabulary = [
                anchor.get("key") for anchor in anchors if anchor.get("key")
            ] + [
                anchor.get("landmark")
                for anchor in anchors
                if anchor.get("landmark")
            ]
            self.landmark_pool.reset_episode(vocabulary)
        return self.plan

    def update_spatial(
        self, step_id: int, position: Any, heading: Any, candidates: Iterable[Any]
    ) -> Dict[str, Any]:
        self._require_episode()
        if self.spatial_graph is None:
            return {}
        return self.spatial_graph.update(step_id, position, heading, candidates)

    def update_progress(
        self,
        step_id: int,
        position: Any,
        heading: Any,
        view_tags: Dict[str, Any],
        view_geometry: Dict[str, Any],
    ) -> Dict[str, Any]:
        self._require_episode()
        if self.progress_locator is None:
            return {}
        return self.progress_locator.update(
            step_id, position, heading, view_tags, view_geometry
        )

    def update_landmarks(
        self,
        step_id: int,
        position: Any,
        heading: Any,
        view_tags: Dict[str, Any],
        view_geometry: Dict[str, Any],
    ) -> Dict[str, Any]:
        self._require_episode()
        if self.landmark_pool is None:
            return {}
        return self.landmark_pool.update(
            step_id, position, heading, view_tags, view_geometry
        )

    def evaluate_stop(
        self,
        locator_state: Dict[str, Any],
        _legacy_landmark_pool: Any,
        final_landmark_term: Any,
    ) -> Dict[str, Any]:
        self._require_episode()
        if self.terminal_gate is None:
            return {}
        return self.terminal_gate.evaluate(
            locator_state, self.landmark_pool, final_landmark_term
        )

    def completion_text(self) -> Optional[str]:
        self._require_episode()
        if self.progress_locator is None:
            return None
        return self.progress_locator.completion_text()

    def snapshot(self) -> Dict[str, Any]:
        self._require_episode()
        return {
            "schema_version": STATE_REDUCER_SCHEMA_VERSION,
            "episode_id": self.episode_id,
            "plan": self.plan,
            "modules": {
                "progress": self.progress_locator is not None,
                "spatial_graph": self.spatial_graph is not None,
                "landmark_memory": self.landmark_pool is not None,
                "terminal_gate": self.terminal_gate is not None,
            },
        }
