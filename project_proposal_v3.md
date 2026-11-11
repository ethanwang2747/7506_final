# 项目构想 v3：模型理解与底层机制研究

> 状态：供队员讨论。第三版已将研究重心从视觉 token 剪枝与推理效率，调整为多模态 Transformer 的信息组织、token 干预机制和空间信息保留。所有假设均为待检验命题，不将预期现象写成已有发现。

## 一、暂定题目

**What Does a Visual Token Represent? Controlled Interventions on Information Aggregation and Spatial Preservation in Multimodal Transformers**

**视觉 Token 究竟表征了什么？多模态 Transformer 中信息汇聚与空间保留的受控干预研究**

## 二、研究动机

在视觉语言模型中，图像首先被划分为 patch，并转换为视觉 token，随后经过视觉编码器和语言模型的多层注意力、残差与 MLP。虽然这些 token 通常仍能映射回二维图像坐标，但经过多层信息混合后，一个位于坐标 $(x,y)$ 的 token 未必只表示对应图像区域。

它可能同时包含：

- 原 patch 的局部视觉属性；
- 邻近区域或全局场景的信息；
- 由其他 token 汇聚而来的语义；
- 与当前文本问题相关的任务信息；
- 对后续输出重要、但无法从二维 attention 图直接看出的隐式信息。

因此，将 token 可视化叠加到原图上，并不能自动证明模型正在“读取该区域”；高 attention 也不能单独证明对应 token 对输出具有决定性作用。另一方面，已有 VLA 工作表明，对中间层特定 token 加噪或掩蔽，可能改变策略关注的区域和最终动作。这进一步提出了一个值得研究的机制问题：

> **中间层视觉 token 到底保留了哪些位置、语义和任务信息？针对 token 的干预为什么会改变模型输出，其效果又如何受到跨层信息汇聚、token 冗余和空间关系的制约？**

本项目不以提出更快的剪枝算法或降低显存为目标，而是将删除、mask、替换和加噪作为**受控干预工具**，研究多模态 Transformer 的内部信息组织。

## 三、核心研究问题与假设

### RQ1：跨层传播后，视觉 token 与原图像区域还具有怎样的对应关系？

考察不同层 token 中能够读取的局部属性、邻域信息和全局语义，分析信息是否随深度汇聚，以及二维位置是否仍能有效解释 token 表征。

- **H1a**：随着层数加深，token 对原 patch 局部属性的可解码性可能下降，而对非局部信息的可解码性增强。
- **H1b**：不同层可能保留不同类型的信息，因此不能简单描述为“信息逐层丢失”。

### RQ2：为什么 attention 排名与输出干预效果会在特定条件下分离？

不将“attention 不是因果解释”当作新发现，而是研究具体机制：信息汇聚、head 聚合、跨层变换、冗余和集合交互分别在什么条件下影响二者关系。

- **H2a**：信息冗余较高时，单个 token 的 attention 与删除影响可能较弱相关。
- **H2b**：固定聚合规则可能掩盖不同层和 head 的功能差异。
- **H2c**：单 token 干预无法充分预测多 token 联合干预。

### RQ3：删除、mask、替换与加噪是否构成不同的信息干预？

这些操作经常被统称为“去除 token”，但它们实际改变的是不同对象：序列结构、注意力可见性、特征内容或输入分布。

- **H3**：不同干预产生的输出变化不能互相替代；部分性能变化可能来自分布偏移，而不只是目标信息被移除。

### RQ4：语义信息与空间关系信息是否具有不同的干预鲁棒性？

研究物体类别、整体语义、局部文字、目标位置与物体间关系在相同干预下的变化。

- **H4**：模型可能在整体语义保持正确时，已经丢失目标—参照物或夹爪—物体之间的空间关系；该现象是否成立需要真实实验检验。

## 四、总体研究设计

采用“**可控玩具模型发现机制，真实 VLM 检验机制**”的两级设计。

1. 在合成任务中明确每个 patch 的属性、空间关系和答案所需信息，系统控制冗余与干扰；
2. 在真实 VLM 中使用公开视觉问答数据，检验相同机制是否能解释真实模型行为；
3. 使用相同的层选择、干预定义和输出变化指标；
4. 将 probe 作为表征证据，将输出干预作为行为证据，二者不互相替代；
5. 不开展真实 VLM/VLA 微调，不将推理加速作为研究贡献。

## 五、模型基准

### 5.1 可控多模态 Transformer

实现一个小型 encoder-only 多模态 Transformer：

- 输入为规则网格图像的 patch token 与离散文本问题 token；
- 视觉与文本 token 拼接后进行 self-attention；
- 默认 4 层、`d_model=128`、4 heads，根据任务学习稳定性微调；
- 输出为属性、位置或空间关系分类；
- 使用固定输入 token 化与位置编码，确保 patch 坐标可追踪。

模型只需稳定学会受控任务，不追求复杂架构或大规模训练。FiLM、cross-attention、Mamba、秩退化与新 VLA 架构均不属于本项目范围。

### 5.2 真实视觉语言模型

**主模型：`HuggingFaceTB/SmolVLM2-2.2B-Instruct`。**

选择理由：

- 支持图像—文本输入和视觉问答；
- 基于 Idefics3，包含 SigLIP 视觉编码器与 SmolLM2 文本解码器；
- 官方模型卡报告视频推理约需 5.2GB GPU RAM，适合 RTX 3070 8GB 的推理期研究；
- 权重、代码和模型接口公开，Apache 2.0 许可；
- 无需微调，实验集中于中间表征与输出干预。

提取 attention 时使用支持显式权重输出的 eager 实现。模型中间层索引、视觉 token 范围、特殊 token 和位置编码行为以实际代码为准，在 E0 中确认后冻结。

LLaVA-1.5-7B 仅作为可选外部验证，不进入关键路径。

## 六、数据集与任务

### 6.1 可控合成数据

构造 `8×8` 或 `16×16` 网格场景，每个 patch 具有明确的颜色、形状、物体或背景属性。生成三类任务：

1. **局部属性任务**：答案只依赖指定目标的颜色、形状或类别；
2. **空间关系任务**：答案同时依赖目标和参照物的位置；
3. **冗余与集合任务**：多个 patch 重复或互补地提供答案信息。

默认数据规模约 `20k train / 2k validation / 4k test`，实际在最小训练验证后固定。测试条件包括：

- 同分布；
- 未见过的位置与空间组合；
- 不同目标尺寸和干扰物数量；
- 不同信息冗余度；
- 关键 token 单独与联合干预。

玩具模型使用 3 个训练随机种子。

### 6.2 真实公开数据

| 数据集 | 研究用途 | 设置 | 行为指标 |
|---|---|---|---|
| **GQA** | 通用物体、属性与组合关系 | 从公开 balanced 划分固定抽取约 500 题，保留问题类型 | 标准化答案 exact-match |
| **TextVQA validation** | 局部文字与细粒度信息 | 从 5,000 道验证题中固定抽取约 500 题 | 官方 VQA soft accuracy |
| **CV-Bench** | 空间关系、计数及可选深度/距离 | 核心使用 2D relation 与 count，固定公开样本清单 | Multiple-choice accuracy |

另设 dev 样本用于调试 prompt 与输出解析。正式测试样本、提示和层配置冻结后，不再根据结果挑选最优组合。

除全样本结果外，可在“原始模型答对子集”上分析干预后的答案保留率，但该条件子集不能替代全样本结果。

### 6.3 具身视觉案例

在与学姐确认研究边界和使用许可后，选择其 VLA 工作中的一个典型现象作为讨论案例，例如“中间层 token 加噪后，策略从干扰物转向正确目标”。我们只研究：

- 相关层 token 的位置—信息对应；
- attention 与干预敏感性的差异；
- 单 token 与区域集合干预的差异；
- 目标与参照物的空间信息保留。

学姐的 RL 方法、训练流程、成功率和已有可视化均不归属本项目。若无法获得许可或可靠接口，则改用 GiTBench 中允许分析的机器人帧，构造小规模人工核对的空间问答与可视化案例；两种情况都不声称验证 VLA 闭环控制。

## 七、核心实验

### E0：接口与干预语义验证

在正式实验前确认：

- 视觉 token 与原始 patch 坐标的映射；
- 可提取的层、hidden states 与 attention；
- 删除、mask、置零、均值替换和加噪分别修改了计算图的哪一部分；
- 删除后 position embedding、RoPE、特殊 token 与缓存如何处理；
- 相同输入下基线输出是否可复现。

只有计算图验证通过的干预才能进入主实验。物理删除与 mask 的差异用于保证机制结论正确，不展开部署效率研究。

### E1：跨层位置—信息对应研究

在玩具模型和真实 VLM 的早、中、晚三个代表层提取 token hidden states。

#### Probe 设计

训练简单线性 probe，分别预测：

- 原 patch 的颜色、形状、类别或文字相关属性；
- 邻域与全局场景属性；
- 空间关系任务所需的目标/参照物信息。

同时使用随机标签 control task，报告：

\[
\mathrm{Selectivity}=\mathrm{Acc}_{probe}-\mathrm{Acc}_{control}.
\]

Probe 只回答“该信息能否被简单读取”，不证明模型输出实际使用了该信息。需要结合 E2 干预形成证据链。

#### 主要比较

- 层深与局部/非局部信息可解码性的变化；
- 不同任务与信息冗余条件下的变化；
- token 表征与其二维坐标的一致程度。

### E2：输出依赖与干预机制研究

对选定 token 或空间区域执行：

1. attention mask；
2. 零向量或数据集均值替换；
3. 不同强度的高斯噪声；
4. 物理删除（仅当位置与序列行为已验证）。

输出变化使用：

- 正确答案 log-probability 变化；
- 预测答案是否翻转；
- 输出分布的 Jensen–Shannon divergence；
- 任务准确率变化。

对每种干预单独报告，不用一种干预替代另一种的结论。通过“目标区域干预 + 无关区域干预 + 随机区域干预”比较输出依赖，并检验 attention 排名能否预测这些变化。

### E3：冗余与集合交互研究

在合成任务中构造信息重复与信息互补的 token 集合，在真实模型中选择空间相邻、语义相似或任务相关的 token 小组。

令删除或替换单个 token 的输出影响为 $\delta_i$，联合干预集合 $S$ 的影响为 $\delta_S$，定义一个诊断量：

\[
I_S=\delta_S-\sum_{i\in S}\delta_i.
\]

- $I_S>0$：联合影响大于单点影响之和，提示互补或超加性交互；
- $I_S<0$：联合影响小于单点之和，提示冗余或补偿；
- 接近 0：在当前指标下近似可加。

该量是有限干预下的诊断，不声称完整刻画高阶因果。重点分析不同集合结构、层和任务如何改变交互，而不是穷举所有 token 子集。

### E4：语义与空间信息的差异化退化

在相同区域和相同干预强度下，比较：

- 物体/属性识别；
- 文字或局部细节；
- 目标—参照物空间关系；
- 可选机器人图像中的夹爪—目标关系。

通过针对性构造样本，区分：

- 保留目标但删除参照物；
- 保留参照物但删除目标；
- 保留显著物体但破坏全局布局；
- 保留空间覆盖但抑制高语义区域。

最终回答模型是否出现“语义仍正确、空间关系先失效”的现象，并用 E1–E3 的表征与交互结果解释可能原因。

## 八、Attention 在项目中的位置

Attention 只作为一种局部层内信号，与输出干预结果进行对照。

主分析预先固定：

- 代表层；
- head 聚合方式；
- 文本 query 或视觉 token 接收 attention 的定义；
- 特殊 token 是否排除。

报告 Spearman 相关、Top-k 重合和分层案例，但不把低相关直接解释为“attention 无用”，也不把高相关解释为因果证明。层与 head 差异用于分析机制，不用测试集事后挑选最优规则。

## 九、评价标准与统计

### 表征证据

- Probe accuracy 与 selectivity；
- 各层局部/非局部信息可解码性；
- 必要时的置信区间与跨种子稳定性。

### 行为证据

- $\Delta$ 正确答案 log-probability；
- 答案翻转率；
- Jensen–Shannon divergence；
- 绝对准确率及 $\Delta\mathrm{Acc}$；
- 全样本准确率与原始正确子集保留率。

### 集合与空间证据

- 单点与联合干预差异及 $I_S$；
- 目标、参照物与无关区域的干预差异；
- 空间覆盖率与任务变化的关系；
- 代表性成功案例和失败案例。

### 统计协议

- 同一输入上配对比较所有干预；
- 使用 paired bootstrap 估计差值的 95% 置信区间；
- 对二值正误结果可补充 McNemar 检验；
- 玩具模型使用 3 个训练种子；
- 随机区域/集合干预至少使用 5 个选择种子；
- 保存逐样本预测和干预结果，避免只保留汇总均值。

## 十、预期贡献

1. **位置—信息对应证据**：描述跨层 token 对局部、邻域和全局信息的可解码性变化，避免将二维 token 坐标直接等同于信息来源。
2. **干预机制证据**：区分 mask、替换、加噪与删除的计算语义，解释局部 attention 与输出依赖在特定条件下分离的原因。
3. **集合交互证据**：展示冗余与互补如何使单 token 重要性无法直接预测联合干预。
4. **空间保留证据**：检验语义与空间关系在相同干预下是否具有不同鲁棒性，并分析目标与参照物的共同作用。
5. **具身机制讨论**：为 token 级 VLA 干预为什么可能改变动作提供可检验的解释路径，但不将 VLA 方法或成功率归为本项目成果。

项目不预设所有假设成立。若真实 VLM 未复现玩具模型中的现象，或空间关系并未更早退化，这些边界本身就是研究结果。

## 十一、资源与范围控制

- 主实验使用 RTX 3070 8GB；
- 玩具模型完成训练，真实 VLM 仅推理和有限 probe 训练；
- RTX 4090 最多两天，仅用于可选外部验证；
- 一个玩具架构、一个真实主模型、三个公开数据集；
- 三个代表层、四类核心干预，不遍历所有层和 head 组合；
- 集合干预只覆盖受控小组，不穷举子集；
- 不开展 VLA 微调、RL 或完整 GiTBench 闭环实验；
- 不研究 FLOPs、kernel、显存优化或部署加速。

### 止损顺序

若工作量过大，依次删除：

1. 可选 VLA/LLaVA 外部验证；
2. 物理删除之外的复杂位置编码扩展；
3. Gradient/Taylor 等额外重要性指标；
4. 部分真实数据集中的非核心任务。

必须保留 E0、E1、E2、E3 和最小化的 E4，因为它们共同构成项目的机制主线。

## 十二、与相关工作的边界

FastV、SparseVLM 和 VisionZip 将 attention 等信号用于视觉 token 缩减，属于本项目背景，不是我们要复现或超越的效率基线。

已有 attention 可解释性研究指出 attention 权重不能直接等同于因果解释。本项目不重复宣称这一结论，而是进一步研究多模态视觉 token 中造成排序与输出依赖差异的具体条件。

已有 VLA token 加噪工作以学习方法与任务成功率提升为目标。本项目只研究干预机制，并在获得许可后将其作为具身案例；不声称提出该 VLA 方法，不使用其结果冒充本项目实验。

## 十三、初步分工

- **成员 A：可控模型与集合实验**——合成数据、玩具模型、probe、E1 和 E3；
- **成员 B：真实 VLM 与干预实验**——E0、E2、公开数据评测和可视化；
- **共同完成**——E4、统计分析、具身案例讨论和报告。

两部分必须共享层选择、token 坐标、干预定义、输出变化指标和逐样本结果格式，避免形成两个独立项目。

## 十四、暂定论文结构

1. Introduction
2. Related Work
3. Visual Token Representation and Intervention Protocol
4. Cross-Layer Information Aggregation
5. Output Dependence and Set-Level Interactions
6. Semantic versus Spatial Information Preservation
7. Embodied-Vision Discussion
8. Limitations and Conclusion

## 十五、参考资料与公开资源

1. Hugging Face. **SmolVLM2-2.2B-Instruct Model Card.** https://huggingface.co/HuggingFaceTB/SmolVLM2-2.2B-Instruct
2. Hudson and Manning. **GQA: A New Dataset for Real-World Visual Reasoning and Compositional Question Answering.** https://cs.stanford.edu/people/dorarad/gqa/
3. Singh et al. **Towards VQA Models That Can Read / TextVQA.** https://textvqa.org/
4. Tong et al. **Cambrian-1 and CV-Bench.** https://cambrian-mllm.github.io/
5. Belinkov. **Probing Classifiers: Promises, Shortcomings, and Advances.** https://arxiv.org/abs/2102.12452
6. Hewitt and Liang. **Designing and Interpreting Probes with Control Tasks.** https://arxiv.org/abs/1909.03368
7. Jain and Wallace. **Attention is not Explanation.** https://arxiv.org/abs/1902.10186
8. Chen et al. **FastV: An Image is Worth 1/2 Tokens After Layer 2.** https://arxiv.org/abs/2403.06764
9. Zhang et al. **SparseVLM: Visual Token Sparsification for Efficient Vision-Language Model Inference.** https://proceedings.mlr.press/v267/zhang25s.html
10. Yang et al. **VisionZip: Longer is Better but Not Necessary in Vision Language Models.** https://arxiv.org/abs/2412.04467
