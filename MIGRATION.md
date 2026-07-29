# 迁移指南

本仓库只提交源码、实验日志和小体积标注数据。第三方源码库、模型权重和场景资产不入库，
需在新机器上按本文档重新获取。所有链接与版本号均取自旧机器上的 remote 配置和下载日志。

## 0. 一句话流程

```bash
git clone git@github.com:Tooru283/My-Experiment.git Open-Nav   # 主仓库
cd Open-Nav
# 然后依次执行下面第 2 / 3 / 4 / 5 节
```

## 1. 目录布局约定

代码中若干处以 `/root/models/...` 作为默认路径。**新机器沿用同一布局最省事**；
否则需覆盖以下位置：

| 文件 | 变量 | 覆盖方式 |
| --- | --- | --- |
| `run_OpenNav.yaml:78` | `MODEL` | 直接改 yaml |
| `serve_qwen.sh:26` | `MODEL` | 环境变量 `MODEL=` |
| `scripts/smoke/*.sh` | `OPENNAV_LLM_MODEL` / `OPENNAV_SIGLIP_PATH` | 环境变量 |
| `vlnce_baselines/common/navigator/api.py:30` | `DEFAULT_LOCAL_QWEN_MODEL` | 改常量 |
| `vlnce_baselines/common/opennav_ext/visual_evidence.py:18` | `DEFAULT_VISUAL_EVIDENCE_MODEL` | 改常量 |
| `vlnce_baselines/config/default.py:75` | `VISUAL_EVIDENCE.MODEL` | 改配置 |
| `paper_analysis/*/{m1_probe,collect_hidden_states}.py` | `DEFAULT_MODEL` | 改常量 |

## 2. 第三方源码库

均被 `.gitignore` 排除，需 clone 到仓库内对应路径。版本号是旧机器上的实际 HEAD。

```bash
git clone https://github.com/BAAI-DCAI/SpatialBot.git SpatialBot
git -C SpatialBot checkout 775ad8c                 # 旧机器状态，无本地改动

git clone https://github.com/xinyu1205/recognize-anything.git recognize_anything
git -C recognize_anything checkout 7cb804a         # 旧机器状态，无本地改动

git clone https://huggingface.co/RussRobin/SpatialBot-3B SpatialBot3B
# ⚠️ 必须覆盖两个本地改动文件，否则 SigLIP vision tower 会去连 HF hub 而非本地目录
cp patches/SpatialBot3B/modeling_bunny_phi.py patches/SpatialBot3B/__init__.py SpatialBot3B/
```

`SpatialBot3B` 的本地改动无 diff 可用（旧机器上该目录的 git 无任何 commit），
故 `patches/SpatialBot3B/` 直接存整份文件。详见 `patches/README.md`。

### habitat-lab —— 必须打补丁

旧机器上 `external/habitat-lab-v0.1.7` 的 origin 指向已删除的 `/tmp/habitat-lab-d6-2`，
且含一处**未提交**改动。不打补丁模拟器无法启动（gym >= 0.26 拒绝 `Discrete(0)`）。

```bash
mkdir -p external
git clone https://github.com/facebookresearch/habitat-lab.git external/habitat-lab-v0.1.7
git -C external/habitat-lab-v0.1.7 checkout d6ed1c0a0    # v0.1.7
git -C external/habitat-lab-v0.1.7 apply ../../patches/habitat-lab-v0.1.7-vln-discrete.patch
```

补丁内容见 `patches/habitat-lab-v0.1.7-vln-discrete.patch`（改 `habitat/tasks/vln/vln.py`
中 `InstructionSensor.observation_space`，`Discrete(0)` → `Discrete(1)`）。

## 3. Conda 环境

两个环境相互隔离：`opennav` 依赖 Habitat 0.1.7 + Python 3.8，`qwen35-serve` 需 Python 3.10+。

```bash
conda env create -f environment.yml                    # opennav
conda env create -f environment_qwen35_serve.yml       # qwen35-serve
```

`qwen35-serve` 的 vLLM 是 nightly 快照 `0.21.1rc1.dev315+g0b68f21e7`。
若该版本已从 nightly 索引中过期，按原部署文档重装最新 nightly：

```bash
uv pip install vllm --torch-backend=cu128 --extra-index-url https://wheels.vllm.ai/nightly
```

精确版本清单另见 `requirements_qwen35_serve.txt`。
完整部署说明见 `Controlled-Navigation-Harness/docs/qwen35_4b_local_deployment.md`。

`opennav` 中 habitat 是 editable 安装，第 2 节 clone 完后需重新安装：

```bash
conda run -n opennav pip install -e external/habitat-lab-v0.1.7
conda run -n opennav pip install -e SpatialBot          # 提供 bunny 包
```

## 4. 模型权重

```bash
hf download Qwen/Qwen3.5-4B --local-dir /root/models/Qwen3.5-4B --max-workers 4   # 8.8G
hf download Qwen/Qwen3.5-9B --local-dir /root/models/Qwen3.5-9B --max-workers 4   # 19G

# SigLIP：旧机器走的镜像站
wget -P /root/models/google-siglip-so400m-patch14-384 \
  https://hf-mirror.com/google/siglip-so400m-patch14-384/resolve/main/model.safetensors
# 同目录还需 config.json / preprocessor_config.json / tokenizer* 等小文件
```

## 5. 检查点与数据集

以下 4 项在旧机器上**没有留下下载来源记录**，按 Open-Nav 原仓库
<https://github.com/YanyuanQiao/Open-Nav> 的 setup 说明获取：

| 路径 | 大小 | sha256（旧机器实测，用于校验） |
| --- | --- | --- |
| `recognize_anything/pretrained/ram_swin_large_14m.pth` | 5.6G | `15c729c793af28b9d107c69f85836a1356d76ea830d4714699fb62e55fcc08ed` |
| `waypoint_prediction/checkpoints/check_val_best_avg_wayscore` | 1.1G | `fc2a1b92d25a9de8f947f3a6cbb125a7d559503c5c02418b252c551e7158beb1` |
| `data/pretrained_models/ddppo-models/gibson-2plus-resnet50.pth` | 48M | `a6a600277efacf5fd98e293267221185d843eb3012aeff62fabfeee24c2bcdad` |
| `data/scene_datasets/mp3d/` | 21G | 目录，未计算；MP3D 需先签署使用协议 |

下载后建议先校验：

```bash
sha256sum -c <<'EOF'
15c729c793af28b9d107c69f85836a1356d76ea830d4714699fb62e55fcc08ed  recognize_anything/pretrained/ram_swin_large_14m.pth
fc2a1b92d25a9de8f947f3a6cbb125a7d559503c5c02418b252c551e7158beb1  waypoint_prediction/checkpoints/check_val_best_avg_wayscore
a6a600277efacf5fd98e293267221185d843eb3012aeff62fabfeee24c2bcdad  data/pretrained_models/ddppo-models/gibson-2plus-resnet50.pth
EOF
```

## 6. 可重新生成的派生产物

无需迁移，跑脚本即可重建：

- `paper_analysis/scoring_head/hidden_states.npy` — 由 `paper_analysis/scoring_head/collect_hidden_states.py` 生成
- `cache_files/`、`image_show/`、各处 `__pycache__/` — 运行时产物

## 7. 研究上下文

实验决策链记录（oracle 泄漏排查、M1/G1 结算、选择层瓶颈定量、SOTA 差距分析等）
已归档至 `docs/research_memory/`，随本仓库迁移，无需另行备份。
入口是 `docs/research_memory/MEMORY.md`（索引），其余为逐条记录。

原位置是旧机器的 `~/.claude/projects/-root/memory/`。若新机器继续使用同一套
记忆机制，可拷回该目录：

```bash
mkdir -p ~/.claude/projects/-root/memory
cp docs/research_memory/*.md ~/.claude/projects/-root/memory/
```

## 8. 仍需单独备份的

以下不在本仓库内，也未归档，按需自行拷贝：

- shell / 工具配置：`~/.bashrc`、`~/.tmux.conf`、`~/.condarc`
- `~/.ssh/`（GitHub 部署密钥等）——**不要入库**
