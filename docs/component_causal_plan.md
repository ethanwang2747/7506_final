# 成分因果分析执行计划

> 2026-11-18。方法细节见 `docs/component_causal_protocol.md`。
> 原则：先修测量，再做因果；每个结论必须过 necessity / sufficiency / specificity 三关。

## Stage 0 — 测量可信化（进行中）

- [x] 单框规则 + same-half/diff-half 分解（0.871 / 0.804；same-half 0.721）
- [x] 灰图/打乱/遮目标 probe 仲裁（0.95 → 灰图 0.99，判定为位置泄漏）
- [x] 参照物反事实（base/full 聚合结果）
- [x] **QP 指标修正**：主指标改为 pair preference（true vs opposite），不再用逐样本 accuracy
- [x] **RC 指标修正**：保存 per-pair margins，主指标改为 pair preference（clean > corrupted），带 bootstrap
- [x] 主表固化：base / full × {QP preference, RC specificity gap, natural AUC}

**Stage 0 结果（2026-11-18）：**

| 指标 | base | full | 读法 |
|---|---:|---:|---|
| QP pair preference | 0.615 [0.578, 0.652] | 0.492 [0.443, 0.544] | full 在位置受控任务上没有关系词偏好 |
| RC hflip preference | 0.658 [0.596, 0.713] | 0.625 [0.567, 0.688] | 都高于 0.5 |
| RC vflip（对照） | 0.529 [0.471, 0.592] | 0.583 [0.521, 0.646] | full 对竖直移动也变敏感 |
| **RC specificity gap** | **0.129** | **0.042** | SFT 降低了“水平特异”的参照敏感性 |
| 自然题 AUC | 0.600 | 0.799 | 但单框规则 0.871 |

结论：自然题的提升主要来自绝对位置读出；位置受控指标上，full 没有优于 base，且 RC 特异性下降。Stage 0 完成，进入 Stage 1。

## Stage 1 — 内容 vs 路由（进行中）

- [x] Token ablation（S1/S6，20 对/组，均值替换）
- [x] Attention knockout：text→目标/参照，early/mid/late/all；随机 token 对照
- [ ] 指标：RC preference、Δmargin、翻转率
- [ ] 产出：内容 vs 路由的因果表

**Stage 1 token ablation 结果（base，120 对）：**

| 条件 | S1 pair_pref | S1 Δmargin | S6 pair_pref | S6 Δmargin |
|---|---:|---:|---:|---:|
| 原图 | 0.708 | −0.261 | 0.708 | −0.261 |
| 消融目标 token | **0.567** | −0.080 | **0.583** | −0.089 |
| 消融参照 token | 0.667 | −0.103 | 0.608 | −0.105 |
| 消融背景 token | 0.625 | −0.204 | 0.642 | −0.230 |
| 消融随机 token | 0.700 | −0.213 | 0.642 | −0.200 |

读法：S1 主要靠目标内容，S6 参照内容开始起作用；背景/随机对照保持较高，效应是特异的。

**Stage 1 attention knockout 结果（base，60 对）：**

| 层组 | 目标 token | 参照 token | 随机 token |
|---|---:|---:|---:|
| early (0–7) | 0.750 | 0.667 | 0.667 |
| mid (8–15) | 0.717 | 0.717 | 0.733 |
| late (16–23) | 0.733 | 0.650 | 0.700 |
| all | 0.633 | 0.650 | 0.700 |

读法：只有全层阻断目标/参照时偏好略降（0.633/0.650 vs 随机 0.700），但效应小、特异性弱；late 参照 0.650 也接近随机。结论：**在这个模型上，text→object attention 不是 RC 偏好的主要载体**；内容替换（token ablation）的效应更清楚。这是与 Grounding Isn't Knowing 的模型不同的地方。

## Stage 2 — 精细定位（进行中）

- [x] Tuned lens：last prompt token，dev 训练 affine translator，test 评测
- [x] Head-level CMA：clean=原图，corrupted=参照移动；层级别 + L15 单 head
- [ ] Cumulative head ablation（nAUC）
- [ ] Tuned lens：answer token 版本；与 CMA 层对齐
- [ ] （可选）DAS：参照侧别子空间 interchange

**Stage 2a tuned lens 结果（480 test examples，dev 训练）：**

| 层 | 14 | 15 | 16 | 17 | 18 | 19 | 23 |
|---|---:|---:|---:|---:|---:|---:|---:|
| raw logit lens | 0.501 | 0.522 | 0.537 | 0.544 | 0.578 | 0.607 | 0.606 |
| tuned lens | 0.568 | 0.599 | 0.612 | 0.613 | 0.614 | 0.614 | 0.607 |

读法：tuned lens 把可读性提前到 L15–16，说明“L18–19 才出现”部分是 raw logit lens 的伪影；但天花板仍只有 0.61，模型整体很弱。

**Stage 2b CMA 结果（60 对，median R，bootstrap CI）：**

| 干预 | median R | 95% CI |
|---|---:|---|
| 替换 L15 全部 head | **0.499** | [0.401, 0.526] |
| 替换 L17 全部 head | 0.212 | [0.012, 0.349] |
| 替换 L16 全部 head | 0.134 | [0.003, 0.300] |
| 替换 L15 单个 head（16 个） | 全部 ≈ 0 | — |

读法：L15 是明确的 mediation layer，但效应**分布在多个 head 上**，没有单个“关系 head”；这与 Grounding Isn't Knowing 的稀疏 head 结论不同，是模型特异的。

## Stage 3 — 纠正

- [ ] 只对 Stage 1/2 三关指认的成分做一次干预
- [ ] 候选：红框/双 crop、attention 重加权、side-SFT
- [ ] 在新确认集 + QP/RC 上验证
- [ ] 报告重写：成分 × 效应表 + 可操作结论

## 当前状态

- QP 的逐样本 accuracy 与 pair preference 矛盾（full：0.70 vs 0.49），说明 accuracy 被 yes/no 偏置污染；Stage 0 先修。
- RC 已有聚合结果（hflip Δmargin −0.32/−0.74，flip 16%/19%），但缺 per-pair preference 与 CI。
