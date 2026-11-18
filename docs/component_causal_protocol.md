# 成分因果分析协议（Component Causal Protocol）

> 2026-11-18。目的不是“再找一个新机制”，而是回答：**这条推理链上，哪个成分对答案有因果作用、有多大、能不能被替代/纠正。**
> 方法参照 [Grounding Isn't Knowing](https://arxiv.org/abs/2608.23074)、[Decodable Is Not Grounded](https://arxiv.org/abs/2606.31257)、[Dual Mechanisms](https://arxiv.org/abs/2603.22278)、[Why Is Spatial Reasoning Hard](https://arxiv.org/abs/2503.01773)，以及 activation patching / DAS / tuned lens 的标准做法。

---

## 0. 先修测量：不用被绝对位置污染的题做因变量

原 6 组左右题的单框规则 AUC 0.871 / 准确率 0.804，高于 full adapter 0.799 / 0.721。在这套题上做成分分析，涨跌分不清是“比较”还是“绝对位置读得更准”。

因此主因变量换成两个**位置受控**任务：

1. **同图换问句对（QP）**：同一张图、同一对物体，只把“左/右”换掉，标签由构造保证 1/0。绝对位置在两个问题里相同。
2. **参照物反事实（RC）**：目标不动，只把参照物水平移到另一侧；竖直移动做对照。指标：Δmargin、翻转率、恢复率。

自然 6 组题只作为“外部行为”报告，不作为机制结论的主表。

**基线**：单框规则、position-only、同图 QP 的 pair preference、灰图/打乱图。

---

## 1. 协议总览：五层 × 三类干预

```text
输入/分辨率 → 视觉 token（内容） → 定位/绑定 → 路由（attention） → 整合 → 读出 → 先验
```

对每个候选成分做三类干预：

- **必要性（necessity）**：去掉/破坏它，答案掉多少？
- **充分性（sufficiency）**：从正确样本把它补回/换入，答案恢复多少？
- **特异性（specificity）**：同样大小/位置的随机成分，效应是否更小？

只有必要性和充分性都成立，且特异性对照显著更弱，才能说“这个成分是瓶颈”。

---

## 2. 各层的具体做法

### 2.1 视觉 token 内容（token ablation）

参照 Grounding Isn't Knowing 的做法：

- 用目标/参照 mask（我们只有框，可用框内 token）选择视觉 token；
- 把选中 token 替换为**全局平均视觉 embedding**（不是零，保持分布）；
- mask 做 **shrink / dilate**（0/1/2 token padding），区分“物体内部”与“周围上下文”；
- 对照：同数量随机 token，3 个种子；
- 指标：关系准确率、Δmargin、方位报告准确率。

能回答：关系判断依赖目标内部、参照内部，还是周围上下文？

### 2.2 定位/绑定（input-level）

- 原图 vs 红框标出目标+参照 vs 只给两个 crop vs 目标贴到空白画布已知位置；
- 指标：方位报告、QP、RC；
- 对照：标无关物体、随机框。

能回答：失败是“找不到/绑不定”，还是“找到了但读不出”。

### 2.3 路由（attention knockout + head CMA）

**Attention knockout**（Geva et al. 2023；Grounding Isn't Knowing）：

- 阻止所有 post-image token 对目标/参照视觉 token 的 attention；
- 按 4 层一组、全层两组设置；
- 对照：同数量随机视觉 token；
- 指标：关系准确率、Δmargin。

**Head-level CMA**（Grounding Isn't Knowing 的 MF）：

- clean = 原图（关系真）；corrupted = 参照物移到另一侧（关系假）；
- 只保留 clean 正确、corrupted 错误的 pair；
- 逐 head 把 clean 的 prompt-token 激活 patch 到 corrupted run；
- MF = (L_base − L_patched) / (L_base − L_src)，L 为答案 token 的 NLL；
- 对照：随机 head、负 MF head、同层其他 head；
- 用 MF 排序做 cumulative head ablation，报告 nAUC。

能回答：哪些 head 真正把参照物信息送进答案；是稀疏还是弥散。

### 2.4 整合（relation composition）

- 同图 QP 的 pair preference（true question margin > opposite）；
- 参照物反事实的翻转率；
- 目标/参照同时移动 vs 只移动一个；
- 指标：pair preference、flip rate、Δmargin。

能回答：模型有没有把两个物体放在一起比较，还是只读一个。

### 2.5 读出（tuned lens + answer-token）

- **Tuned lens**（Belrose et al. 2023）：每层训一个 affine translator 到最终 hidden，再用 LM head 读 yes/no；比 raw logit lens 可靠；
- 同时读 **last prompt token** 与 **answer token**（teacher forcing 下答案前一位）；
- 对照：raw logit lens、随机 translator；
- 指标：每层 AUC、与最终答案一致率、校准。

能回答：答案在哪一层变得可读；是读出晚，还是 lens 伪影。

### 2.6 子空间（DAS，可选但推荐）

- 在最佳层取 last-token hidden；
- 用 DAS 学一个低秩旋转，把“参照在左/右”对齐到一个子空间；
- 做 interchange intervention：把 clean 的子空间换到 corrupted，反之亦然；
- 对照：随机子空间、同维度其他子空间；
- 指标：答案翻转率、Δmargin。

能回答：是否存在一个可干预的“参照侧别”子空间；比 mean-difference steering 更原则。

---

## 3. 统计与决策规则

- 主指标预先指定：QP 的 pair preference、RC 的 flip rate、自然题的 AUC（仅外部行为）。
- 每个条件 n≥40 对/组；image-cluster bootstrap（按图像重采样）；Holm/FDR 校正。
- 报告效应量 + 95% CI + 成本（额外前向次数、token 数）。
- 决策规则：
  1. token ablation 目标/参照 → 谁必要？
  2. attention knockout 到目标/参照 → 路由是否必要？
  3. head CMA → 哪些 head 充分？随机对照是否更弱？
  4. tuned lens → 答案何时可读？与 CMA 的层是否一致？
  5. DAS → 子空间干预是否翻转答案？
  6. 只有以上都指向同一层/同一成分，才做干预。

---

## 4. 三周执行表

| 周 | 任务 | 产出 |
|---|---|---|
| 第 1 周 | QP + RC 评测固化；单框/position-only/灰图基线；image-cluster bootstrap | 位置受控主表；确认因变量可信 |
| 第 2 周 | token ablation（目标/参照/上下文 × shrink/dilate × 随机对照）；attention knockout（层组 × 随机对照） | 内容 vs 路由的因果表 |
| 第 3 周 | head CMA + tuned lens +（可选）DAS；只对最强成分做一次干预验证；报告重写 | “哪一层/哪个成分是瓶颈 + 什么手段能救”的结论 |

---

## 5. 预期能写出的结论

- 一张 **成分 × 必要性/充分性/特异性** 的表；
- 一个明确的“瓶颈层/成分”，以及它在 QP/RC 上的效应量；
- 一个**可操作的纠正**（标框、attention 重加权、side-SFT 之一），并说明它改变了哪个成分；
- 如果所有干预都无效，也是一个干净的负结果：**这套基准的失败不在这些成分上**。

---

## 6. 参考

- Li et al. *Grounding Isn't Knowing: Do VLMs Need Object Localization for Spatial Reasoning?* [arXiv:2608.23074](https://arxiv.org/abs/2608.23074)
- Liao et al. *Decodable Is Not Grounded: A Vision-Ablation Arbiter for VLM Spatial Reasoning.* [arXiv:2606.31257](https://arxiv.org/abs/2606.31257)
- Cui et al. *The Dual Mechanisms of Spatial Variable Binding in VLMs.* [arXiv:2603.22278](https://arxiv.org/abs/2603.22278)
- Chen et al. *Why Is Spatial Reasoning Hard for VLMs? An Attention Mechanism Perspective.* [arXiv:2503.01773](https://arxiv.org/abs/2503.01773)
- Geva et al. *Dissecting Recall of Factual Associations in Auto-Regressive LMs.* EMNLP 2023.
- Meng et al. *Locating and Editing Factual Associations in GPT.* NeurIPS 2022.
- Wang et al. *Interpretability in the Wild: a Circuit for Indirect Object Identification.* ICLR 2023.
- Geiger et al. *Finding Alignments Between Interpretable Causal Variables and Distributed Neural Representations.* [arXiv:2303.02536](https://arxiv.org/abs/2303.02536)
- Belrose et al. *Eliciting Latent Predictions from Transformers with the Tuned Lens.* [arXiv:2303.08112](https://arxiv.org/abs/2303.08112)
- Heimersheim & Nanda. *How to Use Activation Patching: A Practical Guide.* 2024.
