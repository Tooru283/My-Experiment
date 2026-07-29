---
name: project-m3-harness
description: "M3 ep100 result SR=20%, E3 abstain mechanism implemented 2026-06-30 to fix premature stop + RC3"
metadata: 
  node_type: memory
  type: project
  originSessionId: d7e660c1-ead5-4844-8398-9823f61067f3
---

M3 `proactive_stop_gate` implemented (2026-06-29). **ep100 M3 result (20260629)**: SR=20% (↓1 from max_config 21%), OSR=24% (↓2).

**Why M3 degraded:**
- ep259: M3 committed at 3.34m (outside 3m radius) → OSR lost. Could reach 2.11m if continued.
- ep321: M3 committed at 3.15m → OSR lost. Could reach 1.91m.
- Root cause: COMMIT_DIST_THRESHOLD was missing; any dist<3.5m could commit STOP.

**How to apply:** When analyzing near-goal failures, check: (1) does selector output STOP? (2) if not, does M3 proactive_stop_gate fire? (3) is dist within COMMIT_DIST_THRESHOLD?

**E3 implemented 2026-06-30** (config-controlled, all params in `run_OpenNav.yaml`):
- **E3-for-M3** (`PROACTIVE_STOP_GATE.COMMIT_DIST_THRESHOLD: 2.0`): V2 re-eval fires for dist<3.5m but STOP committed only when dist<2.0m (inside success radius)
- **E3-B** (`STOP_EVIDENCE_VERIFIER.E3_ARRIVAL_OVERRIDE_DIST: 2.5`): when dist<2.5m AND arrival_evidence=True AND not_visible → force allow_stop=True
- **E3-A** (`STOP_EVIDENCE_VERIFIER.E3_ABSTAIN_DIST: 2.0`): when dist<2.0m AND not_visible AND no arrival_evidence → abstain instead of hard-reject (agent continues)
- **E3-C** (tagged in StopEvidenceVerifier): E3-A variant where recent 3 steps had ftv=True

**E3 pre-analysis**: 0-3m ftv rate only 11-45% vs arr_rate=1.0 → systematic unreliability.

**RC3 failures**: ep824/ep1084/ep1106 still need ep5 validation to confirm if E3-B helps.

**Next**: ep5 validation → ep100 M4.

**Related:** [[project_a0_baseline]], [[project_max_config_20260628]]
