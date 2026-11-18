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

## Stage 1 — 内容 vs 路由（第 2 周）

- [ ] Token ablation：目标/参照/上下文 × shrink/dilate × 随机 token 对照
- [ ] Attention knockout：text→目标/参照，4 层一组 + 全层；随机 token 对照
- [ ] 指标：RC preference、Δmargin、翻转率
- [ ] 产出：内容 vs 路由的因果表

## Stage 2 — 精细定位（第 3 周）

- [ ] Head-level CMA：clean=原图，corrupted=参照移动；MF；随机/负 MF 对照
- [ ] Cumulative head ablation（nAUC）
- [ ] Tuned lens：last vs answer token；与 CMA 层对齐
- [ ] （可选）DAS：参照侧别子空间 interchange

## Stage 3 — 纠正

- [ ] 只对 Stage 1/2 三关指认的成分做一次干预
- [ ] 候选：红框/双 crop、attention 重加权、side-SFT
- [ ] 在新确认集 + QP/RC 上验证
- [ ] 报告重写：成分 × 效应表 + 可操作结论

## 当前状态

- QP 的逐样本 accuracy 与 pair preference 矛盾（full：0.70 vs 0.49），说明 accuracy 被 yes/no 偏置污染；Stage 0 先修。
- RC 已有聚合结果（hflip Δmargin −0.32/−0.74，flip 16%/19%），但缺 per-pair preference 与 CI。
