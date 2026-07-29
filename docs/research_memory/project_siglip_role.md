---
name: project-siglip-role
description: SigLIP is the vision encoder inside SpatialBot3B and is actively used every navigation step for spatial scene description
metadata: 
  node_type: memory
  type: project
  originSessionId: c35b5734-2607-4e3f-a063-51feb1431660
---

SigLIP (`google/siglip-so400m-patch14-384`) is the `mm_vision_tower` inside SpatialBot3B and IS actively used in every navigation step.

**Why:** SpatialBot3B/config.json sets `mm_vision_tower: "google/siglip-so400m-patch14-384"`. `builder.py` dispatches to `SigLipVisionTower` when 'sig' is in the name. `spatialNavigator.py` calls `spatial.observe_view()` → `spatialbot_description()` every step → SigLIP extracts image features → SpatialBot generates distance-aware scene descriptions fed to Qwen3.5-4B.

**How to apply:** The naming convention `*_qwen_siglip_local_*` reflects the two-model architecture: SigLIP+SpatialBot3B for visual perception (per-step depth-aware scene description), Qwen3.5-4B for reasoning/navigation decisions. Observation 20 in claude-mem ("SigLIP Vision Model Not Active") was INCORRECT and should be disregarded.
