# VLM 课程项目：研究讨论与路线记录

> 本文档记录项目方向的讨论过程、文献调研、模型选择和推荐实验路线。它是供后续设计与执行使用的工作笔记，不是已经批准的最终 proposal，也不把尚未实施的改进写成已有结果。
>
> **当前进度、诊断结论与执行计划以 [`docs/progress_and_plan.md`](docs/progress_and_plan.md) 为准；本文保留讨论过程。**

## 1. 项目目标与硬性约束

### 1.1 课程要求

根据 `requirement.txt`：

- 选题与课程相关即可，LLM、推理模型、LLM agents 只是示例。
- 最终交付一份短会议或 workshop 风格的书面报告。
- 报告必须描述 problem、approach、results 和 limitations。
- 评分看重：
  - 问题设计的创新性，**或者**技术执行的可靠性；
  - 报告的清晰度与质量。
- 不要求口头展示。
- 截止日期为 2026 年 12 月 10 日。
- 不要求提出全新 attention、提升模型性能、训练 VLA 或给出必然有效的解决方案。

因此，本项目首先是研究训练：学会提出假设、构造对照、实施干预、分析负结果并限定结论。复现和借鉴已有方法是合理且必要的。

### 1.2 当前资源与边界

- GPU：RTX 3070 8GB。
- 只做 VLM / 多模态模型。
- 不做 VLA、真机或机器人仿真。
- 不做全模型预训练或大规模微调。
- 可以做推理期特征提取、probe、activation patching，以及冻结特征上的轻量读取器训练。
- 优先复用现有 SmolVLM2 代码、数据和实验结果。
- 目标是形成一个能用于课程报告和面试讲述的完整研究闭环，而不是追求大规模 SOTA。

## 2. 当前仓库已经有什么

### 2.1 项目现状

当前仓库研究的是：

- 模型：`HuggingFaceTB/SmolVLM2-2.2B-Instruct`。
- 任务：GQA 派生的空间关系与语义属性 yes/no 判断。
- 方法：行为测试、区域均值替换、activation patching、线性 probe 和多种对照。
- 主要文档：
  - 项目与复现说明：`README.md`
  - 实验命令：`experiments/README.md`
  - 报告草稿：`report/report_v5_draft.md`
  - 更新的实验笔记：`results/dashboard.md`
  - 汇总表：`results/summary.md`

当前没有 SmolVLA、LIBERO、action expert 或机器人训练代码。

### 2.2 模型流水线与干预阶段

当前使用五个代表阶段：

| 阶段 | 位置 | 含义 |
|---|---|---|
| S0 | `vision_model.post_layernorm` 输出 | 视觉编码器输出，27×27 网格 |
| S1 | `connector` 输出 | 进入语言模型前的视觉 token，9×9、共 81 个 |
| S6 | 文本层 6 输入 | 约 25% 深度 |
| S12 | 文本层 12 输入 | 约 50% 深度 |
| S18 | 文本层 18 输入 | 约 75% 深度 |

无 image splitting 时，图像缩放到 384×384，S1 为 81 个视觉 token。splitting 后会产生多个 crop 和一个 global block，token 数与网格需要单独处理。

### 2.3 已有主要结果

以下是已有实验结果，不是新方案的预期结果：

- 语义属性任务整体较稳定，test AUC 大致为 0.69–0.89。
- 空间任务表现不均匀，并受到对象组合、分辨率和样本质量影响。
- 无 splitting 时，只有 chair/table 的空间 AUC 置信区间明显排除随机水平；splitting 后，部分组提高到约 0.72–0.79。
- 目标区域均值替换在 S0–S12 常影响答案裕度，在 S18 接近零。
- 完整视觉块 patching 在 S0/S1 几乎完全恢复源答案裕度，S6 较高，S12 部分恢复，S18 接近零。
- 完整视觉块恢复首先是正对照，不能解释为唯一的信息存储位置。
- 区域消融影响输出，不等于该区域正确编码了空间关系。
- 区域激活能转移部分答案证据，不等于模型能够正确完成源—目标关系判别。
- 自由生成存在明显 yes/no 偏置，候选裕度排序与最终阈值决策需要分开报告。

### 2.4 新结果与报告草稿不完全同步

`results/dashboard.md` 比 `report/report_v5_draft.md` 更新，正式写作前必须统一：

- 严格过滤低质量或含义含糊的配对后，部分空间组 AUC 达到 0.76–0.92。
- 早先“模型不会做 above/below”的强结论已经撤回。
- 40 对同图翻转实验显示，各组都可能有正向区域迁移，但迁移强弱与干净行为不相关。
- position-residual probe 和 position-matched 分析已经补做；报告限制部分仍有旧表述。
- 后期测试集上的筛选和任务选择具有探索性，不能继续边看 test 结果边挑选正式任务。

当前仓库的价值不仅是某个机制结论，也包括：发现了数据配对、标签有效性、分辨率和输出校准等替代解释。

## 3. 讨论中逐渐澄清的问题

### 3.1 最初担心的问题

项目容易变成：

> 先选择一个宽泛问题，再对许多层和区域做消融，最后根据结果寻找解释。

这种流程能生成实验记录，但如果没有稳定现象和核心假设，就难以回答：

- 到底定位了什么实际问题？
- 为什么这些干预能支持当前解释？
- 能否提出与诊断对应的改进？
- 结论对其他样本或模型是否成立？

### 3.2 一个更合理的研究闭环

> 稳定的问题现象 → 排除替代解释 → 定位受限环节 → 根据诊断做最小改进 → 在未见数据上验证。

改进不是课程要求中的必需项。即使最终没有提升性能，只要问题清楚、对照可靠、结论有边界，也能成为完整项目。

### 3.3 不应预设答案一定是 attention

最初的想法逐渐从“研究 attention 策略”转为：

> 模型怎样形成、选择、组合和读取视觉证据？

Attention 是候选机制或候选改进方式，而不是预先确定的瓶颈。

## 4. 必须区分的核心概念

### 4.1 大白话定义

- **表征 / hidden state**：模型内部的向量，可能包含图像、文字和任务信息。
- **Probe**：额外训练的小读取器，用于检查某种信息能否从向量中恢复。
- **Ablation / 消融**：破坏或替换某部分，观察输出变化。
- **Activation patching**：把输入 A 的激活写入输入 B 的前向过程，观察能否转移 A 的答案证据。
- **Readout / 读取**：将内部向量转化为答案或其他输出的过程。
- **Counterfactual / 反事实**：尽量只改变一个条件，例如只翻转左右关系而尽量保持场景内容不变。
- **Attention**：根据查询选择和混合 token 信息的计算方式，不等于模型里的全部信息，也不自动构成解释。

### 4.2 四个不能互相替代的命题

1. **Encoding / 表征形成**：信息是否进入隐藏状态？
2. **Decodability / 可读取性**：外部读取器是否能恢复信息？
3. **Causal use / 实际使用**：干预这些信息或连接是否改变输出？
4. **Readout improvement / 读取改进**：另一种读取方式能否在未见输入上改善任务表现？

因此：

- probe 能读出来，不表示原模型在使用；
- 原模型输出差，不表示信息不存在；
- attention 路由提高性能，不自动证明它修复了假设中的机制；
- activation patch 有效，不表示该位置是天然、唯一的计算路径。

### 4.3 Attention 不是全部信息的“压缩中介”

简化写为：

\[
\operatorname{Attention}(Q,K,V)=\operatorname{softmax}(QK^\top)V
\]

- Q/K 影响信息路由；
- V 和后续变换影响实际读取的内容；
- residual stream、MLP、layer norm、connector 等也保存和改变信息；
- loss 降低不要求 attention 变得稀疏或符合人类直觉。

应该区分“attention 图上关注哪里”和“改变某条连接是否影响任务输出”。

## 5. 文献调研：别人怎样开展类似研究

以下用于学习研究结构和方法，不要求全部精读。2026 年工作目前主要应视为新预印本，而非已经被充分复现的定论。

### 5.1 从失败现象到诊断和改进

- [Eyes Wide Shut? Exploring the Visual Shortcomings of Multimodal LLMs](https://arxiv.org/abs/2401.06209)
  - 从视觉失败出发，构造 CLIP-blind pairs 和 MMVP。
  - 检查视觉编码器与 MLLM 失败之间的关系。
  - 最后用 Mixture of Features（MoF）补充视觉特征。
  - 值得学习的是“现象—假设—受控验证—改进”的结构。

### 5.2 多层视觉特征与读取接口

- [Dense Connector for MLLMs](https://arxiv.org/abs/2405.13800)
  - 不只使用最后一层，而是利用多层视觉特征。
  - 说明“中间层包含有用信息”本身不是新问题。

- [Cambrian-1](https://arxiv.org/abs/2406.16860)
  - 系统比较视觉表示，提出 CV-Bench 和 Spatial Vision Aggregator。
  - 说明视觉表示、空间聚合和评测应共同考虑。

- [Mixture of Layers: Dynamic Layer Routing for Visual Reasoning](https://arxiv.org/abs/2610.09440)
  - 根据文本指令，在图像级或 patch 级动态选择视觉编码器层。
  - 与“任务条件化多层读取”的想法非常接近。
  - 包含图像路由、文本条件路由等对照。

### 5.3 可解码不等于可控制

- [Decoding Task Progress from VLA Representations](https://arxiv.org/abs/2608.13474)
  - 任务进度可以被线性 probe 读取，但不能据此有效控制策略。
  - 将 weak decodability、strong decodability 和 steerability 分开。
  - 最终把 probe 用于运行监测，说明不一定需要提升主任务性能才能产生用途。

### 5.4 VLA 相关工作带来的启发，但不进入本项目实验

- [SmolVLA](https://arxiv.org/abs/2506.01844)：曾触发“保留多少层、action expert 怎样读取”的讨论。
- [VLA-Adapter](https://arxiv.org/abs/2509.09372)：研究 VL 条件如何接入动作。
- [LayerRoute](https://arxiv.org/abs/2609.06079)：根据动作状态动态路由 VLM 层。
- [SpatialVLA](https://arxiv.org/abs/2501.15830)：使用 Ego3D Position Encoding 和 Adaptive Action Grids。
- [LIBERO-PRO](https://arxiv.org/abs/2510.03827)：通过受控扰动暴露 VLA 鲁棒性问题。

这些工作帮助澄清了“读取”“空间表征”和“因果使用”的概念，但本项目已明确不进入 VLA。

### 5.5 Attention sink 与视觉 token 异常

- [Vision Transformers Need Registers](https://arxiv.org/abs/2309.16588)
  - 高范数 token 常出现在低信息背景区域。
  - 这些 token 可能被重新用于全局计算，而不是保留局部位置内容。
  - register token 可以承接这类计算角色。

- [A Unifying View of Attention Sinks](https://arxiv.org/abs/2606.08105)
  - 相似 sink 图案可能对应不同功能：
    - adaptive NOP：近似抑制当前 head 的更新；
    - broadcast：聚合和广播全局信息。
  - 因此不能看到亮点就认定它是错误或应该删除。

## 6. 现代 VLM 的改进位置与可借鉴方式

现代 VLM 的进步不只是修改视觉编码器，也包括数据、语言模型、训练和输出。

| 改进位置 | 典型做法 | 对应瓶颈 | 本项目的借鉴方式 |
|---|---|---|---|
| 视觉编码器 | SigLIP、DINOv2、InternViT、多编码器 | 原特征缺少信息 | 暂不换 encoder；表征不足时再考虑 |
| 层选择 | Dense Connector、多层融合 | 最后一层丢失局部信息 | 比较单层和多层冻结特征 |
| 动态读取 | MoL、任务条件化路由 | 不同问题需要不同层或区域 | 轻量 query-conditioned readout |
| 分辨率 | 动态分辨率、splitting | 输入阶段丢失细节 | 作为控制因素 C |
| Connector | MLP、pixel shuffle、Q-Former、DeepStack | 视觉特征接入语言模型的方式 | 研究读取接口，不重训整个 VLM |
| 输出 | 候选评分、阈值校准 | 有排序证据但决策错误 | 分开报告 margin 与生成 |

核心决策原则：

> 缺信息 → 改视觉输入或编码器；  
> 有信息但没读到 → 改 connector/readout；  
> 有答案排序但输出错误 → 改决策与校准。

## 7. 研究方向如何收缩

### 7.1 不再研究的宽泛对象

- 泛化的“图片理解能力”。
- 同时研究 OCR、计数、深度、3D、语义属性和所有空间关系。
- 直接训练或替换视觉编码器。
- 大规模比较不同 VLM 的 benchmark 性能。
- 直接进入 VLA 或机器人动作。

### 7.2 选定的任务领域

主任务限定为：

> **二维图像中的目标—参照物空间关系判断。**

例如：`Is the woman to the left of the man?`

选择它的原因：

- 能明确区分目标、参照物和关系。
- 有 scene graph 和 bbox，可进行区域分析。
- 能构造严格配对与同图翻转反事实。
- 能区分对象识别、位置表征、关系组合、任务读取和输出校准。
- 已有代码和数据基础设施都围绕它构建。

语义属性任务保留为对照，用于比较同一模型和干预流程下的属性判断与关系判断，不作为第二条主线。

### 7.3 编码器相关决策

- 不直接替换视觉编码器，避免把参数量、训练数据和外部知识混入结论。
- 不训练新的视觉编码器。
- 将编码器视为固定特征来源。
- 研究编码器各层“有什么”是方法，不是最终宽泛目标。
- 多编码器或 MoF 只作为“现有表征确实缺少信息”时的备用拓展。

## 8. 推荐主线：A 为主，B/C 为补充

### 8.1 暂定题目

**Seeing Is Not Answering: Diagnosing and Improving Spatial Readout in a Small Vision-Language Model**

中文：

**看得见却答不对：小型视觉语言模型中空间证据的读取差距与轻量改进**

### 8.2 核心研究问题

> **SmolVLM2 的空间关系判断失败，在多大程度上来自视觉证据缺失，又在多大程度上来自已有证据没有被有效读取？**

子问题：

1. 控制位置捷径后，不同阶段的冻结特征是否包含可泛化的关系信息？
2. 区域干预和 activation patching 是否影响答案，这种影响是否与正确判别一致？
3. 冻结模型后，轻量多层或任务条件化读取器能否改善未见配对与同图翻转上的表现？

### 8.3 A、B、C 如何组成一条信息链

```text
输入分辨率与视觉证据（C）
          ↓
视觉 token 如何路由、是否出现 sink（B）
          ↓
问题如何读取这些表征并形成答案（A）
```

- **C：行为控制**——空间证据不足是否只是分辨率或输入处理造成的。
- **A：核心研究**——信息是否存在、能否读取，以及轻量读取器能否改善。
- **B：机制补充**——attention 异常集中是否具有实际因果作用。

它们不是三个并列项目。

## 9. 推荐实验路线

### Phase 0：冻结配置并验证实现

- 固定主模型、prompt、候选评分和图像预处理。
- 固定 train/dev/test，不用 test 结果选择任务。
- 使用严格配对并抽样人工核对。
- 确认视觉 token 坐标、hook 位置、identity patch 和缓存行为。
- 先阅读并同步 `dashboard.md` 与报告中的新旧结果。

### E1：可信行为地图

保留：

- 原始图像与严格 source/target 配对；
- 同图水平翻转；
- no-split 与 splitting；
- 空间任务与语义属性对照；
- 候选答案 AUC、配对裕度差、生成准确率和 yes/no 偏置。

回答：哪些失败在分辨率提高后消失，哪些仍然存在？

### E2：信息存在性与读取对照

在 S0、S1、S6、S12 提取冻结特征，在相同训练数据下比较：

- 原始 VLM 零样本行为；
- 最终 token 上的线性读取头；
- 单层视觉特征的线性 probe；
- MLP probe；
- 后续必要时再加入 attention-pooling。

必要对照：

- position-only baseline；
- 随机标签；
- 图像级 train/test 隔离；
- 未见对象组合或关系组；
- 相同监督预算和明确的模型容量说明。

回答：控制位置捷径后，是否仍有可泛化的关系信息？

### E3：因果证据与 B 补充

复用区域消融和 patching，并强调：

- 输出依赖不等于正确关系编码；
- 区域转移不等于正确行为；
- source-only 裕度变化可能是答案偏置变化，优先考虑 bilateral 指标。

若稳定观察到 sink：

1. 确认层、head 和 token；
2. 检查跨图像、跨目标位置是否稳定；
3. 检查 token norm、value norm 和 attention 分布；
4. 对 sink、目标区域和随机区域实施等 token 数量干预；
5. 比较候选答案和读取器输出变化。

若 sink 没有特异因果作用，将其写成负结果并停止扩展。

### E4：由诊断驱动的轻量改进

只有在 E2 支持“信息存在但原模型利用不足”时，才重点实施读取改进。

冻结 SmolVLM2，仅训练小型读取器，比较：

- 均值池化；
- 线性读取；
- 固定多层融合；
- 任务条件化 attention 或多层路由。

输入为问题表示与一层或少数几层视觉 token，输出为 yes/no。可提前缓存特征，以适应 8GB 显存。

评估不能只看随机划分准确率，还应包括：

- 未见对象组合；
- 同图翻转；
- 是否仍依赖固定位置；
- 是否仍受 yes/no 先验支配；
- 参数量、训练预算和推理成本。

严格结论应是“改进下游读取接口可以利用冻结表征”，不能直接写成“修复了原模型内部 attention”。

## 10. 结果决策树

### 情况 1：特征读取得好，原模型答得差

- 支持存在 readout gap 的候选证据。
- 继续比较多层/任务条件化读取器。
- 检查改进是否泛化到反事实和未见组合。

### 情况 2：所有读取器都差

- 不支持简单的读取不足解释。
- 更可能是表征、关系组合或任务构造仍有问题。
- 优先复查分辨率、目标—参照物联合信息和标签有效性。
- 不应直接添加复杂 attention 模块。

### 情况 3：候选裕度排序好，但自由生成差

- 更像输出校准或阈值问题。
- 可单独做轻量校准，不包装成视觉表征机制。

### 情况 4：复杂读取器不优于简单读取器

- 报告简单读取已经足够。
- 不需要为了故事继续增加模块复杂度。

## 11. 主模型选择与候选模型

### 11.1 推荐决定

主模型继续使用 `SmolVLM2-2.2B-Instruct`。

理由不是它最新或最强，而是：

- 已经实际跑通现有实验；
- 官方模型卡称视频推理约需 5.2GB GPU RAM；
- 27 层视觉编码器、24 层文本层；
- 无 splitting 时固定 81 个视觉 token；
- connector 较简单，区域映射、hook 和 patching 已验证；
- 更换模型会要求重做 E0、配对形状、区域映射和主要结果。

### 11.2 候选比较

| 模型 | 8GB 适配性 | 项目适配性 | 决定 |
|---|---|---|---|
| SmolVLM2-2.2B | 已有实际运行基础 | 简单 connector、固定网格、已有全套干预 | 主模型 |
| Qwen3-VL-2B-Instruct | BF16 权重粗略约 4GB，实际取决于 token 数 | 空间能力更新更强，但动态分辨率、Interleaved-MRoPE、DeepStack 增加复杂度 | 可选第二模型 |
| InternVL3.5-2B-HF | 总参数约 2.3B，BF16 权重粗略约 4.6GB | ViT–MLP–LLM，动态高分辨率，需要重做映射 | 可选但无必要 |
| Qwen2.5-VL-3B-Instruct | BF16 权重粗略约 6GB，余量较紧 | 动态分辨率与窗口 attention，工程更复杂 | 不推荐专门切换 |

官方资料：

- [SmolVLM2-2.2B](https://huggingface.co/HuggingFaceTB/SmolVLM2-2.2B-Instruct)
- [Qwen3-VL-2B](https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct)
- [Qwen3-VL 配置](https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct/raw/main/config.json)
- [InternVL3.5-2B-HF](https://huggingface.co/OpenGVLab/InternVL3_5-2B-HF)
- [Qwen2.5-VL-3B](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct)

Qwen3-VL 已通过 DeepStack 融合多层 ViT 特征。如果用它做可选验证，可以问：在已有多层融合的模型中，SmolVLM2 上观察到的读取差距是否减弱？

外部验证不是必需项；时间不足时只保留一个主模型。

## 12. 统计与推断注意事项

- 同一输入上的干净、消融和 patch 结果尽量使用配对比较。
- 同图多问题或 source/target 配对不能当作完全独立样本。
- 报告效应量和置信区间，而不只报告点估计。
- 小样本翻转实验的排名可能不稳定，应优先使用较大的冻结样本集。
- 多组、多层、多区域比较存在多重比较问题，不应把单个置信区间当作全项目显著性证明。
- 镜像关系组不是独立重复验证。
- 位置标签由 bbox 中心定义时，position-only baseline 不可省略。
- 使用额外训练的 probe 与零样本原模型比较时，要加入同监督预算的读取基线，避免把训练差异误当成机制差异。
- 若使用真实 bbox 或其他标准答案信息，应称为 oracle diagnostic，不包装成实际可用的感知方案。

## 13. 最小交付、可选拓展与止损

### 13.1 最小可交付项目

- 严格冻结的空间关系与语义对照数据。
- 可信行为地图。
- 分阶段的读取/位置对照。
- 现有区域消融和 patching 的谨慎解释。
- 清楚说明证据能支持和不能支持的结论。

### 13.2 可选加分项

按优先级：

1. 一个由诊断支持的轻量读取器实验。
2. 同图翻转和未见对象组合上的泛化验证。
3. attention sink 的短小因果补充。
4. Qwen3-VL-2B 的少量外部验证。

### 13.3 止损标准

- 没有稳定行为现象：先修任务，不解释中间层。
- probe 对照不清楚：先修数据和对照，不训练复杂读取器。
- sink 没有因果作用：删除该支线。
- 读取器没有优势：保留诊断结果，不继续堆模块。
- 第二模型需要重做大量基础设施：删除外部验证。
- 不把 VLM 与 VLA 两套故事塞入同一报告。

## 14. 报告与面试叙事

可以按以下顺序讲：

1. 小型 VLM 能识别不少对象和属性，但空间关系判断不稳定。
2. 最终准确率无法区分“没有形成证据”和“证据没有被读取”。
3. 我们首先修正了配对、标签有效性、分辨率和输出偏置等替代解释。
4. 然后用冻结特征读取、区域干预和 activation patching 提供不同等级的证据。
5. 如果存在读取差距，再训练轻量读取接口，并在反事实和未见组合上验证。
6. 无论改进是否成功，都明确结论成立的条件、负结果和局限。

比“我做了很多层消融”更清楚的概括是：

> 我们研究的是视觉证据为什么存在却没有稳定转化为答案，并用对照实验区分表征、读取和校准问题。

之前的 VLA 经历最多作为研究动机或概念来源，不需要进入实验，也不需要把结论推广到动作控制。

## 15. 到训练机后的建议检查顺序

### 15.1 先恢复环境，而不是立即写新方法

```bash
cd /path/to/7506_final
git status
nvidia-smi
```

确认：

- 原始 GQA 图像和 scene graph 是否仍在；
- `data/derived/` 中的配对和 mean vectors 是否存在；
- 模型缓存是否可用；
- 当前 Python 环境和 CUDA 是否正常；
- 本地结果是否与仓库一致。

### 15.2 阅读顺序

1. `research_direction_notes.md`（本文档）
2. `results/dashboard.md`
3. `report/report_v5_draft.md`
4. `experiments/e0_validate.py`
5. `experiments/e4_stage_probe.py`
6. `experiments/flip_counterfactual.py`

### 15.3 最小实现验证

在已有 dev 配置上优先复用 E0，而不是先跑新的 test：

```bash
python experiments/e0_validate.py \
  --task spatial \
  --group "woman|to the left of|man" \
  --split dev \
  --splitting 0
```

确认 identity patch、张量形状和 token 映射仍然成立。

### 15.4 下一步真正需要做的 pilot

先完成“信息存在性”对照，而不是先写 attention 改进模块：

- 选择冻结的 dev 任务与对照；
- 提取 S0/S1/S6/S12 特征；
- 使用相同训练数据比较最终 token 线性读取、视觉特征线性 probe、position-only 和随机标签；
- 检查现有样本量是否足够，尤其是 chair/table 等训练样本很少的组；
- 根据 pilot 结果进入第 10 节的决策树。

第一轮最重要的产出不是性能提升，而是三张可信的表/图：

1. 各任务在 no-split/splitting 下的行为结果；
2. 各阶段的读取结果及 position/random 对照；
3. 在同图翻转或未见组合上的泛化结果。

## 16. 当前一句话决定

> **以 SmolVLM2-2.2B 为固定主模型，只研究二维目标—参照物空间关系；以“视觉证据存在但没有被有效读取”为核心候选解释，用语义属性、分辨率和输出校准控制替代解释，以 attention sink 为可选机制补充，并只在诊断支持时实现轻量多层或任务条件化读取器。**
