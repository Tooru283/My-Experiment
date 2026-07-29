# patches/

存放**无法从任何远端恢复**的本地改动。这些文件所在目录都被 `.gitignore` 排除，
且其上游仓库不含这些改动，旧机器一旦丢失就不可重建。迁移时必须手工应用。

## `habitat-lab-v0.1.7-vln-discrete.patch`

目标：`external/habitat-lab-v0.1.7`（上游 `facebookresearch/habitat-lab` @ `d6ed1c0a0`）

旧机器上该 clone 的 origin 指向已删除的 `/tmp/habitat-lab-d6-2`，改动从未提交。
内容是 `habitat/tasks/vln/vln.py` 中 `InstructionSensor.observation_space`
由 `Discrete(0)` 改为 `Discrete(1)`——gym >= 0.26 拒绝 `Discrete(0)`，不打模拟器起不来。

```bash
git -C external/habitat-lab-v0.1.7 apply ../../patches/habitat-lab-v0.1.7-vln-discrete.patch
```

## `SpatialBot3B/`

目标：`SpatialBot3B/`（上游 `https://huggingface.co/RussRobin/SpatialBot-3B`）

这里存的是**整份文件**而非 diff，因为旧机器上该目录的 git 仓库没有任何 commit，
无基线可供 diff。从 HF 下载完成后，用这两个文件覆盖同名文件：

```bash
cp patches/SpatialBot3B/modeling_bunny_phi.py patches/SpatialBot3B/__init__.py SpatialBot3B/
```

- `modeling_bunny_phi.py` — 增加了 `DEFAULT_LOCAL_SIGLIP_PATH` 与
  `_resolve_vision_tower_path()`，使 vision tower 从本地目录加载而非 HF hub，
  并支持 `OPENNAV_SIGLIP_PATH` 环境变量覆盖。上游版本只认 hub id，离线环境加载会失败。
- `__init__.py` — 空文件，使该目录可作为 Python 包导入。

其余文件（`configuration_bunny_phi.py`、`config.json`、tokenizer 等）与 HF 下载原件一致，
无需覆盖。判定依据：旧机器上只有这两个文件的 mtime 晚于 2026-05-27 的下载批次。
