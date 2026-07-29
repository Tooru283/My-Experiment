---
name: project-vlm-call-count
description: 每步调用的真实构成与各自产出的实测信息量——completion_estimation 自相矛盾(r=0.12)、observe_view 距离是语言先验(r=0.175)
metadata: 
  node_type: memory
  type: project
  originSessionId: 492084f4-1f0f-4063-8a92-2e46db3bcfff
  modified: 2026-07-19T15:45:21.623Z
---

**20260719 实测更正**（数据源：ep100_series_m420260719_130559，687 步）。
此前本条写的"正常每步 2 次 VLM（V1+selector）"**是错的**：选择器和
completion_estimation 都是**纯文本**调用（`spatialNavigator.py:122` 与 `:622`
都是 `gpt_infer(system, user)`，无图像参数；trace 里 category=`text_llm`）。
选择器读的是 observe_view 产出的**文字**场景描述，不是图片。

## 每步真实构成（均值 4.0 个候选，单步 ~54 s）

| 操作 | 耗时 | 模态 | 次数 |
|---|---|---|---|
| `observe_environment`→`observe_view` (`api.py:153`) | 7.2 s | **VLM** | **每候选一次** |
| `visual_evidence` | 10.3 s | VLM | 1 |
| `completion_estimation` | 15.7 s | 纯文本 | 1 |
| `navigator_move_to_next_vp`（选择器） | 15.6 s | 纯文本 | 1 |
| `stop_current_view_evidence` | 4.7 s | VLM | 54% 的步 |

真实 VLM 次数是**每步 5~6 次**（主力是 observe_view 的每候选一次），不是 2 次。
两个最贵的（31.3 s，58%）都是纯文本 → **想省时间的杠杆不在视觉侧。**

## 各自产出的实测信息量

**`completion_estimation` 不可信（信心高，自证伪）**：
- 69/94 集在**第 1 步**（尚未移动）就自报指令 100% 完成；414/687 步自报饱和 1.0
- 自报进度 vs 真实靠近目标进度 **Pearson r = 0.120**
- 18/93 集整集输出一字未变
- **75 次"已执行动作数"相邻步减少** —— 逻辑不可能，说明它每步独立重编而非跟踪状态
- 它被塞进选择器 prompt，所以是**往决策注入噪声**，不只是浪费
- 附带：这就是 tilt `final_clause` 在 81/100 集第 1 步误触发的根因，
  当时按"预算 bug"修触发条件，其实是在给这个模块的幻觉打补丁

**`observe_view` 距离是语言先验不是感知（信心高）**：
64% 距离值落在 1/2/3/4/5 整数上（n=8397）；37% 的方向描述是完美整数阶梯
（"门1米、楼梯2米、墙3米"，n=2166）；VLM 最近物距 vs 实测航点距离 **r=0.175**、
MAE 1.74 m、系统高估（2.52 vs 1.47）。`spatialNavigator.py:68` 注释早已写明不可靠，
P0 曾为此注入实测距离但 `GEOMETRY_INJECTION` 现为关，**选择器现在只看得到幻觉距离**。
物体标签部分**未证伪**（信心中）：间接证据是几何四维先验（完全不看场景描述）
命中 36.7% **恰等于** LLM 选择器 —— 但命中率定义是"选直线最近候选"，
真用语义的选择器可能故意选非最近且正确，所以这是暗示不是证明。

**`visual_evidence` 是唯一确认有用的视觉调用**：`nmatch` 权重 +0.2999，
逼近 `dist` 的 +0.3030。但同模块 `grounding_score` 的 s_room/s_dir/s_rel/s_dst
权重精确 0.000 —— 10 秒产出里真正有信息的是**一个整数**，利用率极低。

**`visual_target_verifier`**：634/1069 = 59% 返回 `not_applicable` 空转，
但计时 0.00 s 不花钱，可放着。（`depth_stop_veto` AUROC 0.531 已判死，见
[[project-selector-bottleneck-20260719]]）

## 结论

每步 54 s 中 **22.9 s（43%）花在经不起检验的产出上**（completion 15.7 + observe 7.2）。
首选实验：stub 掉 `completion_estimation` 跑 ep100 —— 省 ~3 h/轮（11h→8h），
SR 不降就直接删，降了则说明其价值不在进度准确性上，本身是发现。
**这是目前唯一一个失败也不亏的改动。** observe_view 消融更贵更险，等 completion 结论后再说。

见 [[project-arm-c-20260719]]、[[project-selector-bottleneck-20260719]]、[[feedback-no-autonomous-runs]]
