# 2026-04-20 实验日报

## 今日目标
读取 Open-Nav README，梳理复现路径，补齐视觉模块和环境依赖的关键入口，为后续 smoke run 做准备。

## 今日完成
1. 阅读并确认仓库复现链路
   - 核对 `README.md`，明确 Open-Nav 依赖以下关键部分：
     - Habitat 0.1.7
     - Matterport3D 场景
     - OpenNav_R2R-CE_100 数据
     - waypoint predictor 权重
     - DDPPO depth encoder
     - SpatialBot 与 RAM 视觉模块
   - 明确推理入口为 `run_OpenNav.bash` 与 `run.py`。

2. 完成视觉相关仓库接入
   - 接入/放置：
     - `recognize_anything/`
     - `SpatialBot3B/`
   - 为 SpatialBot 当前代码路径补齐兼容包装：
     - `SpatialBot3B/configuration_bunny_phi.py`
     - `SpatialBot3B/modeling_bunny_phi.py`
   - 目的：打通 `api.py` 中 Bunny/Phi 相关导入。

3. 开始环境依赖排查
   - 对照 `environment.yml` 与 `requirements.txt` 梳理依赖。
   - 识别出 Open-Nav 不是单一 Python 包问题，而是 Habitat / Habitat-Sim / RAM / SpatialBot 多套依赖组合问题。

## 关键产出
- 明确了 README 对复现所需外部资源的完整要求。
- 补齐 SpatialBot 兼容导入层，为后续模型加载扫清第一层导入障碍。
- 确认后续工作的核心是：先让环境和模型加载通，再做 smoke run。

## 今日问题
1. Open-Nav 对外部资源依赖重，README 只给出了入口，没有现成“一键复现”环境。
2. 视觉链路依赖第三方仓库实现，导入路径与当前仓库引用存在兼容差异。
3. 后续运行预计会受到 Habitat 版本、GPU 依赖和模型权重路径共同影响。

## 当前结论
- 代码尚未进入有效推理阶段。
- 但今天已经把“复现需要什么”和“先修哪里”梳理清楚，属于复现准备阶段的关键推进。

## 下一步
1. 继续补齐 `opennav` 环境缺失依赖。
2. 检查 Habitat / Habitat-Sim 版本兼容。
3. 准备 smoke run 所需数据、场景和权重目录。
4. 尝试进入首次可执行推理。
