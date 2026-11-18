# 项目审查报告：十个视角的交叉审查

> 审查日期：2026-11-18
> 审查对象：`/home/yucheng/ris/7506project`
> 方法：10 个子 agent 独立读取代码、结果 JSON、日志和报告，分别从统计、因果、数据、probe、SFT、跨模型、文献、叙事、复现、红队十个角度审查；主 agent 对关键数字做了独立复核。

---

## 0. 一句话结论

**有真正的东西，但不是现在这个标题讲的东西。**

真正成立、且最有价值的，是一个可复现的负结果：

> 在这套 GQA 派生的左右关系对上，一个只使用目标框、完全不看参照物的「绝对 x」规则，就能达到 **AUC 0.871 / 准确率 0.804**；base 模型是 **0.600 / 0.569**，full adapter 是 **0.799 / 0.721**。模型和 SFT 都没有超过「只看一个框」的诊断上界。

这不是「实验没做好」，而是这套 benchmark 的标签本身主要由单个物体的绝对位置决定。项目真正发现的，是**评测效度问题**，而不是「方位读出瓶颈」。

如果继续按现在的叙事（「视觉里有 0.95 的方位信息，决策层只有 0.65，教方位报告可以修复关系判断」）写，审稿人会用三句话打穿：

1. 0.95 是用真值框池化出来的；位置残差化后只剩 0.687，position-only 是 0.961。
2. 0.799 没有超过单框规则 0.871；在真正需要比较的 same-half 子集上，full adapter 只比单框规则高 0.022。
3. logit lens 没有前移；full adapter 的最佳层是 L23（0.860），不是 L17–18。

---

## 1. 项目实际做了什么

项目以 `SmolVLM2-2.2B-Instruct` 为对象，研究它在 GQA 派生空间关系 yes/no 题上的失败位置。已完成的工作量是扎实的：

- **数据**：从 GQA scene graph 重建受控 source/target 配对（同一问题、两张图、一正一负），图像级划分；另有 strict、confirm、同图水平翻转、细粒度存在性/属性/行为对照。
- **诊断**：E0 校验、E1 行为图、E2 区域消融、E3 patching、E4 替代机制；逐 token probe、9×9 网格 probe、last-token 侧别 probe、logit lens、几何 vs 模型对照。
- **干预**：LoRA-SFT 的 2×2 对照（side-only / relation-only / mixed / base）、full epoch、两个种子、cross-group、aug/rev、空白图、POPE/TIFA 回归。
- **对照**：Qwen2-VL-2B、Qwen2.5-VL-3B（4bit）、SmolVLM2-500M；InternVL2.5-2B 与 LLaVA-OneVision-0.5B 尝试失败。
- **工程**：40+ 实验脚本、168 个结果 JSON、SFT 14 分钟 / 5.55GB 峰值、完整日志。

这套东西作为课程项目的**工作量和技术栈**是够的，甚至偏多。问题不在「做得少」，而在「主叙事选错了」。

---

## 2. 十人审查团判决

| # | 审查视角 | 判决 | 最致命的一条 |
|---|---|---|---|
| 1 | 统计与实验设计 | **弱** | n=20/组，6 组里只有 1 组 CI 排除 0.5；720 个 cell 未做多重比较校正；图像簇被当独立样本 |
| 2 | 因果机制 | **部分成立** | R 的分母病（61% 样本 \|Δm\|<0.5）；mean 里混入 target token；L18–19 的跃迁 CI 含 0 |
| 3 | 数据与基准效度 | **有保留** | 单框绝对位置就能到 AUC 0.804；core 240 只有 57 对进 strict；strict 5,348 对从未整评 |
| 4 | 表征 probe | **部分成立** | 0.95 是 oracle box 位置泄漏；残差化后 0.687；「best cell 0.61–0.71」是 81 格取最大 |
| 5 | SFT | **有条件可信** | raw margin 把增益放大约 2–4 倍；same-half 上 side-only 仅 +0.026（CI 跨 0）；LR schedule 有 bug |
| 6 | 跨模型与外部效度 | **部分成立** | token 预算 81 vs 165 未对齐；Qwen2.5-VL-3B 与 500M 结果未进报告；两个失败对照未披露 |
| 7 | 文献与新颖性 | **课程够，workshop 勉强** | 七条主张没有一条是全新机制；最危险的是 0.95 的 probe 解读，与 2026 年「decodability ≠ grounding」正面冲突 |
| 8 | 报告叙事 | **大修** | 0.95 未标 oracle box；logit lens 前移句与数据相反；三份文稿数字互相矛盾 |
| 9 | 可复现性 | **C 级** | README 主路径因缺 `PYTHONPATH` 直接报错；168 个 JSON 无命令/种子/commit；0.799 的评测无日志 |
| 10 | 红队 / so-what | **weak reject**（课程可 weak accept） | 单框捷径 + same-half 分解表明：模型学的是半区归属，不是序关系比较 |

十个视角的共识非常一致：**工程可信、方向有趣、主结论过强，且真正的发现被写成了配角。**

---

## 3. 真正做出来的东西

### 3.1 硬结论（可以写进报告，经得起追问）

1. **受控配对协议本身是资产。** 同问题、两图、一正一负，图像级隔离；47.9 万条 GQA 标注关系里只有 0.05% 违背框中心序；confirm 480/480 复算一致。这个协议比直接刷 GQA 原始 yes/no 干净得多。
2. **失败是任务特异的。** 同图细粒度对照：存在性 0.865、属性 0.765、行为 0.865，空间关系 AUC 0.600；空白图控制下视觉增益 +0.41 / +0.32 / +0.15 / +0.10。这不是「这个 VLM 不会看图」。
3. **模型不能报告方位。** 水平 960/960 回答 "left"；选项顺序反转后 933/960 仍是 "left"；竖直 689/960 回答 "top"，准确率 0.492，低于真值多数类基线 0.596。这是常数先验，不是读框。
4. **给文字方位后，模型会跟随。** 真方位 0.760（两物不同半区 0.921），反方位 0.271（不同半区 0.120）。文字侧的组合/跟随是存在的，但这不等于视觉前向里的组合电路完好。
5. **SFT 在 same-half 子集上有真实提升。** base 0.531 → side-only 0.622 → relation-only 0.727 → full 0.737（AUC，n=224）。这是项目里最实的正结果：模型确实从「完全不会比较」被推到了「能做一些比较」。
6. **单框捷径（最重要的新发现）。** 一个只用目标框绝对 x、完全不看参照物的规则，在 480 个样本上 AUC 0.871 / 准确率 0.804；在 same-half 子集上仍有 0.715。这说明这套 benchmark 的分数，大部分可以由「一个物体的绝对位置」解释。
7. **工程负结果。** attention head 消融/放大、latent steering、learnable region reader 都没有可靠增益。但这些实验的功效不足（4/768 head、clean acc 0.517、reader 0.561 [0.511,0.609]），只能作为「未发现」而不是「不存在」。
8. **跨模型的初步信号。** Qwen2-VL-2B 同任务 0.746/0.826，决策层侧别 0.838/0.801；SmolVLM2-2.2B 是 0.600，决策层 0.653/0.544。方向值得保留，但不能说「model-specific」（见 §5）。

### 3.2 半硬结论（需要加限定）

- **视觉 token 里有物体身份。** 9×9 逐格 probe 的 name pointing 0.54–0.62，两个都对 0.317–0.398；但 chance 基线 0.205 偏松，固定中心格就能到 0.31，所以真实 lift 是 1.7–2×，不是 3×。
- **视觉 token 里有绝对位置。** 0.95 是 oracle box 的上界；残差化后 0.687，position-only 0.961。可以说「位置可解码」，不能说「关系可解码」。
- **决策层侧别弱。** L19 target-left 0.653、reference-left 0.544；但这是「绝对侧别」而不是「关系答案」，而且 target-left 与 answer probe 0.589 不是同一件事。
- **外部 benchmark 不回归。** POPE 0.783→0.850（主要是 yes rate 0.33→0.517），TIFA cued 0.753→0.867 但 coverage 0.788→0.588（长度匹配后 0.702）。这是真实的 trade-off，不是「全面变好」。
- **同尺寸 Qwen 更强。** 但 SmolVLM2 是 81 个视觉 token、Qwen2-VL-2B 是 165 个；分辨率/ token 预算混杂未排除。

### 3.3 不成立 / 必须降级

| 现在的说法 | 为什么不成立 | 应该改成 |
|---|---|---|
| 「pooled visual tokens decode side at 0.95」 | oracle box 池化 + 标签就是 box 在左半；残差化 0.687；position-only 0.961 | 「oracle-box 上界 0.95；无框全局 reader 0.474–0.492；残差化 0.687」 |
| 「0.600 → 0.799/0.802，Δ+0.271」 | 单框规则 0.871/0.804 更高；same-half 只比规则高 0.022；flip 口径只有 +0.071 | 「raw margin +0.199；flip +0.071；same-half +0.206；仍低于单框规则」 |
| 「logit lens moves to L17–18」 | full 最佳层 L23=0.860；L17=0.667，L18=0.831，L19=0.847；信号变强没前移 | 「信号增强，未前移；L19 与最终 logit 的一致率 0.81，但准确率仅 0.53」 |
| 「readout bottleneck」 | 0.95 与 0.65 是不同特征、不同任务、不同 CV、不同容量；latent steering 无效反而削弱「方向存在但没读出」 | 「localization-to-readout gap；binding、reference-side representation、late-layer routing 仍未排除」 |
| 「model-specific」 | Qwen2.5-VL-3B 4bit 0.817/0.687/0.619；SmolVLM2-500M 0.541；同族不复现；InternVL/LLaVA 失败未披露 | 「family/training-specific；单 checkpoint 比较，不能分离架构、数据、规模」 |
| 「composition intact」 | 给真方位时 same-half 只有 0.559，diff-half 0.921；模型学的是「X 在左、Y 在右 → yes」的半区模式 | 「文字侧跟随可用；视觉序关系比较未建立」 |
| 「strict 5,348 对 / confirm 隔离」 | core 240 只有 57 对进 strict；strict 从未整评；confirm 的 bbox 未存，无法做 same-half 分析 | 「strict 是抽样诊断，不是全量评测；confirm 只做了图像级隔离」 |

---

## 4. 核心证据一：0.95 vs 0.65 为什么不成立

这是整个项目最关键的论证，也是十个审查者中至少五个独立指出的同一问题。

**0.95 是什么？** 用 GQA 真值框把目标区域的视觉 token 池化，再训练一个线性 probe 预测「这个框在不在左半」。标签就是框的位置，特征是用框选出来的 token——这几乎是同义反复。`results/diagnostics_summary.md:251-252` 自己写了 "(oracle box)"。`results/readout_summary.md:106` 给了残差化后的数字：0.687（test）/ 0.705（flip）；`results/readout_summary.md:116` 给了 position-only oracle：0.961（test）/ 0.950（flip）。也就是说，**0.95 还不如纯位置基线，残差化后只剩 0.687**。

**0.65 是什么？** 是 L19 last-token 的 target-left 侧别 probe，不是关系答案；reference-left 只有 0.544；answer probe 是 0.589。它和 0.95 不是同一特征空间、不是同一 token 集、不是同一任务、不是同一 CV 容量。把它俩直接相减，得不到「信息在、读不出」的结论。

**logit lens 也不是「晚决策」的证据。** base 的 L19 与最终 logit 一致率 0.81，但准确率只有 0.53——它只是在复刻最终答案（包括错误），不是在正确答案出现时突然点亮。L17→L18 的 AUC 增量 +0.069，CI 含 0；strict 480 例里 L18/L19/final = 0.525/0.530/0.548，没有 18–19 峰。full adapter 的最佳层是 L23，不是 L17–18。

**latent steering 的负结果反而拆台。** 如果「方位方向存在但下游没读」，那么把 side probe 方向注入 last token 应该能改变行为；但现有 steering 用的是 source−target answer direction，不是 side direction，且 clean acc 0.517 / AUC 0.549，本来就没有 headroom。这个实验既不能支持、也不能反驳 readout 假说，属于「测错了方向」。

**结论：** 项目可以保留「视觉 token 里有绝对位置、决策层侧别弱」这一现象，但不能把它包装成「信息在、只是读不出」的强因果链。后者需要 position-matched probe、灰图/打乱图仲裁、以及 probe-direction injection 三个实验才能成立，目前一个都没有。

---

## 5. 核心证据二：0.799 为什么不是「关系提升」

我用 `results/sft/eval_relation_full.json` 的 480 个 meta 逐样本对齐了 GQA 真值框，计算了一个**单框绝对 x 规则**：

- 对「X 在 Y 左边」：分数 = `0.5 - x_center(X)`；对「X 在 Y 右边」：分数 = `x_center(X) - 0.5`。
- 这个规则**完全不看参照物 Y**，只用目标 X 的绝对位置。

结果：

| 指标 | base | side-only | relation-only | full | 单框规则 |
|---|---:|---:|---:|---:|---:|
| 总体 AUC（n=480） | 0.600 | 0.717 | 0.777 | **0.799** | **0.871** |
| 总体准确率 | 0.569 | — | — | 0.721 | **0.804** |
| same-half AUC（n=224） | 0.531 | 0.622 | 0.727 | **0.737** | 0.715 |
| diff-half AUC（n=256） | 0.673 | 0.814 | 0.827 | **0.859** | 0.999 |

三个读法：

1. **总体 0.799 低于单框规则 0.871。** 一个不看参照物的规则，比整个 adapter 还好。说明这套题的分数上限，很大一部分来自「目标物绝对位置」，而不是「两物比较」。
2. **same-half 上 full 只比规则高 0.022（0.737 vs 0.715）。** same-half 是真正需要比较的子集（两个物体在同一半区），但 full adapter 也只是刚刚追平「读绝对 x」的水平。base 在这里是 0.531，所以 SFT 确实学到了东西——但学到的是「把绝对 x 读得更准」，不一定是「比较两个框」。
3. **diff-half 上 full 0.859，规则 0.999。** 一旦两个物体分处不同半区，单框规则几乎完美；模型反而差了一截，说明它的定位/读出并不稳定。

**最诚实的正结果**不是「0.799 修复了关系判断」，而是：

> SFT 把模型在 same-half 子集上的 AUC 从 0.531 提到 0.737（+0.206），这是真实的进步；但它只是把模型推到「单框绝对 x」规则的水平（0.715），并没有证明模型学会了序关系比较。

这也解释了为什么 side-only 在 same-half 上只有 +0.026（CI 跨 0），而 relation-only 有 +0.118：**直接教关系标签，比教方位更能让模型在 same-half 上动一点；但两者都还没有跳出绝对位置的框架。**

---

## 6. 现在能回答 / 不能回答的问题

| 问题 | 现在的答案 | 置信度 |
|---|---|---|
| SmolVLM2 在这套 6 组左右关系上弱吗？ | 是：AUC 0.600，生成 acc 0.569；SFT 后 0.799/0.721 | 高 |
| 是通用视觉失败吗？ | 不是：存在性 0.865、属性 0.765、行为 0.865 | 高 |
| 视觉 token 里有物体身份吗？ | 有，但 name pointing 的真实 lift 只有 1.7–2×，不是 3× | 中高 |
| 视觉 token 里有绝对位置吗？ | 有；0.95 是 oracle box 上界，残差化 0.687，position-only 0.961 | 高 |
| 决策层读到了方位吗？ | 弱：L19 target-left 0.653、reference-left 0.544；但这是绝对侧别，不是关系答案 | 中 |
| 模型能报告方位吗？ | 不能：水平常数 "left"，竖直 "top" 先验低于多数类 | 高 |
| 给文字方位后能组合吗？ | 能跟随，但 same-half 只有 0.559；不是「序关系比较」 | 中高 |
| SFT 改善关系判断了吗？ | raw margin 改善；same-half +0.206；但总体低于单框规则，flip 只 +0.071 | 中 |
| 瓶颈是读出、定位、绑定还是校准？ | **不能回答**；三者未分离 | — |
| SFT 学到的是比较还是绝对位置？ | **不能回答**；证据更偏向绝对位置 | — |
| 是否 model-specific？ | 不能；Qwen2.5-VL-3B 与 500M 削弱了「model-specific」，更像 family/training-specific | — |
| 能推广到其他关系/模型/数据吗？ | 不能；6 组左右关系、单模型、GQA 派生 | — |

---

## 7. 如果继续：三条路线

### 路线 A（推荐）：把负结果做成主线

**新定位：** 不是「诊断并修复空间读出」，而是「一个空间关系 benchmark 到底测了什么」。

**核心贡献（按重要性）：**

1. **单框捷径 + same-half/diff-half 分解。** 已有数据，1 天可完成。把 §5 的表放进主文，作为第一个结果。
2. **reference-only counterfactual（决定性的实验）。** 目标框不动，只把参照物移到另一侧（或只 patch 参照区域），看 margin 是否翻转。如果 margin 不翻，模型就不是在比较；如果翻，才说明有比较。2–3 天，8GB 可做。
3. **同图换问句对。** 同一张图，分别问「X 在 Y 左边吗？」和「X 在 Y 右边吗？」，标签由构造保证 1/0；这样 position-only AUC 被压到 0.5，才能真正测关系。3 天。
4. **统一评测协议。** 主表用同一 480 例 + flip；raw margin、flip-consistency、same-half/diff-half 三列并列；image-cluster bootstrap + Holm 校正。
5. **报告重写。** 删掉「readout bottleneck」「composition intact」「model-specific」，改成「single-box shortcut / what SFT actually learns」。

**预期结论：** 模型学的是半区归属/绝对位置，不是序关系；SFT 只教会前者。这是一个干净、可复现、有方法学价值的负结果。课程报告可以拿高分，workshop 也有机会。

**时间：** 3 周够。第 1 周补单框/same-half/reference-only；第 2 周同图换问句 + 统计修正；第 3 周重写 + 图表 + 交稿。

### 路线 B（坚持机制故事）：补三个因果实验

如果一定要保留「readout bottleneck」：

1. **灰图/打乱图/遮目标 probe 仲裁**（2–3 天）：在灰图、打乱 patch、遮目标区域三种输入上重跑 0.95 侧别 probe 与 0.65 决策层 probe。掉到随机 → 读的是 bbox，claim 改写；不掉 → 这是对「decodability ≠ grounding」最有力的回答。
2. **L19 probe-direction injection**（1 周）：估计 target-left/reference-left 方向，向 last token 注入 ±α，测 yes/no margin；与随机方向、answer 方向对照。这是「信息被下游使用」的直接因果检验。
3. **position-matched probe / residualized**（3–5 天）：同 box 换问句/换身份，使 position-only AUC=0.5，再看 region/last-token probe 是否还 >0.6。

**风险：** 3 周内可能全负；如果全负，故事就变成路线 A。**建议：先做路线 A 的 1 和 2，它们同时也能为路线 B 提供证据。**

### 路线 C（最小可交）：删掉「修复」叙事

保留诊断链 + 负结果，SFT 只作为「半区捷径」的证据，不声称修好空间关系。1 周内可交，风险最低，但浪费了已有的 SFT 工作。

---

## 8. 报告修改清单

### P0（必须改，否则结论不成立）

1. **全篇 0.95 标 oracle box。** 加 position-only 0.961 / residual 0.687；删「information is there but unread」的强断言。涉及 `report/report_v6_draft.md:17`、`:251`、`:294`。
2. **加单框基线 0.871/0.804 + same-half 表。** 把 0.799 明确写成「低于单框规则」；把「Δ+0.271」改成「raw margin +0.199；flip +0.071；same-half +0.206」。
3. **删 logit lens 前移句。** `report/report_v6_draft.md:33-34` 与 `:246-247` 与 `logit_lens_core_full.json` 矛盾；改成「信号增强、未前移；L19 与最终 logit 一致率 0.81，但准确率 0.53」。
4. **改 model-specific → family/training-specific。** 补 Qwen2.5-VL-3B（4bit）0.817/0.687/0.619、SmolVLM2-500M 0.541，披露 InternVL/LLaVA 失败。
5. **统一 base 数字。** 0.600（eval_relation 480）、0.626（readout core）、0.531（confirm）、0.689（dev）是四套口径；主表必须用同一套，其他放附录。
6. **修数字：** POPE F1 0.739（不是 0.732）；Δacc 0.190（不是 0.192）；reference-left 0.544（不是 0.56）；竖直 1384/1920（不是 689/960）；best-cell 0.61–0.71 是 81 格选择偏差，不能当「有弱信号」的证据。
7. **复现修复：** README 加 `export PYTHONPATH=.`；主路径加 `eval_relation.py`；补 `peft`/`scipy`/`av`；每个结果写 `run_meta.json`（argv/seed/git commit/时间/模型路径）。

### P1（应该改）

- 统计：image-cluster bootstrap、Holm/FDR、n=40/组；把「6 组只有 1 组显著」如实写成「组级 n 不足」。
- 三稿统一：`readout_gap_section.md` 仍保留 0.812 表头和「composition intact」，与 v6 冲突；`findings_cn.md` 的中文归因比英文强。
- 图表：v6 正文 0 图；补图号、图注（n、95% CI、控制条件）；重做 dashboard panel B（旧 per-token 0.53–0.54 vs 正文 0.61–0.71）。
- Related Work：至少补 What's Up、BLINK、SpatialEval、Learning to Localize、EmbSpatial、Why Is Spatial Reasoning Hard、Decodable Is Not Grounded、Dual Mechanisms。

### P2（可选）

- 外部 benchmark 的 n/CI：TIFA spatial n=26、TempCompass direction n=6–20、SSv2 30 条；不要用它们支撑「任务特异」的强结论。
- TIFA 长度：matched-length 0.702，仍低 0.086；写清楚是风格+少量内容，不是知识退步。
- POPE：yes rate 0.33→0.517；提升主要来自校准，不是知识。

---

## 9. 三周时间表（11/18–12/10）

| 周 | 任务 | 产出 |
|---|---|---|
| 第 1 周（11/18–11/24） | 单框基线 + same-half/diff-half 重评；reference-only counterfactual；image-cluster bootstrap + Holm | 新主表；决定「比较 vs 绝对位置」的关键图 |
| 第 2 周（11/25–12/1） | 同图换问句对；修统计/复现；报告重写（路线 A 叙事） | 干净的 position-matched 评测；P0 修改完成 |
| 第 3 周（12/2–12/9） | 最终评测、图表、三稿统一、交稿 | 课程报告 + 可复现 repo |

如果第 1 周的 reference-only counterfactual 显示「margin 会翻转」，说明模型确实有比较能力，只是不稳定——那可以保留一部分「readout」叙事，但仍需 position-matched 证据。如果 margin 不翻，路线 A 的负结果就是最终故事。

---

## 10. 最后判断

**这个项目有没有意义？** 有，但不是现在这个意义。

- **作为课程项目：** 有意义，工作量、工程、数据审计都超过一般课程项目；只要把结论改诚实，可以交得很好。
- **作为「机制发现 + 修复」：** 不成立。0.95 是 oracle box 的位置泄漏，0.799 没有超过单框规则，logit lens 没有前移，「model-specific」被同族 500M 和 Qwen2.5 削弱。
- **作为 workshop paper：** 有潜力，但必须重写成「一个空间关系 benchmark 到底测了什么 + SFT 实际学到了什么」的负结果 + 方法学论文。单框捷径 + same-half 分解是真正的新东西。
- **作为主会论文：** 不够。需要第三族模型、position-matched 评测、reference-only 因果实验，以及更强的统计。

**如果只做一件事：** 做 reference-only counterfactual——目标不动，只移参照物。它能在 2–3 天内回答「模型到底有没有比较两个物体」，比再加一个 adapter 有价值得多。

**如果只删一句话：** 删掉「文字侧组合完好，瓶颈只是方位读出」。数据支持的版本是：「模型学的是半区归属/绝对位置；序关系比较没有建立；SFT 只教会了前者。」

这不是一个失败的项目。这是一个**标题写错了的好项目**。把标题改对，东西就出来了。
