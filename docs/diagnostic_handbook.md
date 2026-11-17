# VLM 机制诊断手册

> 目标：不是“找一个大招提升准确率”，而是先建立一套可复用的诊断流程，回答：
>
> **信息在不在？能不能解码？有没有被使用？在哪一层/哪个 token 被使用？是路由问题还是 readout 问题？**
>
> 本手册面向视觉语言模型（VLM），但大部分方法可直接迁移到 LLM、VLM、VLA。

---

## 0. 先记住五个区分

诊断中最常见的错误，是把下面五件事混为一谈。

| 概念 | 问题 | 典型证据 | 不能证明 |
|---|---|---|---|
| **Encoding** | 信息有没有进入 hidden state？ | probe、RSA、SAE | 模型是否使用 |
| **Decodability** | 外部 reader 能不能读出来？ | linear/nonlinear probe | 模型内部是否这样读 |
| **Causal use** | 干预它会不会改变输出？ | ablation、patching、steering | 是否是唯一路径 |
| **Readout** | 最终 token / margin / 阈值有没有把它转成答案？ | logit lens、final-token probe、calibration | 信息是否存在 |
| **Prior / shortcut** | 答案是不是来自语言先验或评测捷径？ | blank/gray、position-only、text-only | 视觉能力 |

一句话：

> **Probe 证明“可解码”，patching 证明“被使用”，steering 证明“可干预”，held-out 证明“可泛化”。**

---

## 1. 诊断的总框架

```text
行为现象
  → 行为地图与对照（blank / gray / flip / position）
  → 可解码性（probe / per-token map / RSA / CKA）
  → 因果必要性（ablation / patching）
  → 因果充分性（patch-in / steering / interchange）
  → 定位（layer / head / token / region）
  → readout（logit lens / final token / calibration）
  → 泛化（held-out groups / images / positions / seeds）
  → 才决定是否干预
```

每一步都要有**对照**，否则结论会被以下因素污染：

- 位置捷径（bbox 位置、图像布局）；
- 物体共现/语言先验；
- probe 容量过大；
- 多层冗余/补偿；
- 测试集选择偏差；
- 多次比较/看 test 调参。

---

## 2. Step 0：先把行为现象定义清楚

### 为什么

如果行为指标本身不稳定，后面的机制分析没有意义。必须先知道“错在哪、错多少、是否稳定”。

### 怎么做

1. **定义任务单位**
   - 单图 yes/no：accuracy、AUC、margin；
   - 同图/同视频 pair：pair accuracy、Δmargin、source-target AUC；
   - 生成任务：factual coverage、hallucination rate、judge score。
2. **分 group / 分类型**
   - 不要只报总体；空间组、存在性、属性、行为、方向、时序要分开。
3. **报告不确定性**
   - bootstrap CI，按 image / pair / video 聚类；
   - 多种子；
   - 配对比较（同一 pair 的原方法和新方法）。
4. **固定 split**
   - dev 用来选层/阈值/方向；test 只跑一次；
   - 图像级/视频级 split，避免同一图的不同问题泄漏。

### 输出

一张行为表：

| group | n | metric | CI | baseline | 备注 |
|---|---:|---|---:|---:|---|

### 常见坑

- 只看 accuracy，不看 AUC / pair ranking；
- 把同一张图的多个问题当独立样本；
- 用 test 结果挑 group；
- 没有 blank/position 对照。

---

## 3. Step 1：行为对照——先排除“没看”和“捷径”

### 为什么

在解释内部机制前，先确认模型是否真的用了输入，以及任务是否被评测捷径解决。

### 三种核心对照

| 对照 | 做法 | 如果结果高，说明 |
|---|---|---|
| **blank / black** | 空白图/黑屏视频，只留问题 | 语言先验/选项先验很强 |
| **gray** | 灰度图，保留形状去掉颜色 | 颜色任务仍高 → 先验；下降 → 用颜色 |
| **position-only** | 只用 bbox 位置/大小训练 classifier | 任务可被位置捷径解决 |
| **flip** | 水平翻转；关系真值按任务反转/保持 | 模型是否镜像一致 |
| **text-only** | 只给问题/选项，不给图 | 纯文本可解程度 |

### 判断

- blank 高 → 先验是主因，先做去偏/校准，不要谈视觉机制；
- position-only 高 → 评测有捷径，必须位置残差化/匹配；
- 全图高、blank 低 → 视觉证据确实被使用；
- 全图低、oracle region 高 → 定位/readout 是瓶颈。

---

## 4. Step 2：可解码性——信息在不在？

### 4.1 Probing 是什么

在 frozen hidden state 上训练一个轻量 classifier，预测某个概念（物体、属性、关系、correctness）。

### 为什么需要

它回答“信息是否线性/非线性可解码”，是机制分析的入口，但不是终点。

### 怎么做

1. 选层/阶段：S0、S1、S6、S12、S18、final token；
2. 选 token：final token、visual token mean、target/ref region mean、每个 token；
3. 标签：object presence、target/ref role、relation truth、correctness；
4. split：image/group-held-out；不要同图泄漏；
5. 标准化特征；logistic regression / MLP；
6. 报告 AUC、accuracy、CI；
7. **控制**：
   - random labels；
   - position-only baseline；
   - 随机 token/region；
   - group-held-out；
   - 相同监督预算的 baseline。

### 能回答 / 不能回答

- 能：信息是否可解码、哪层最强、是否受位置主导；
- 不能：模型是否使用、是否因果、是否唯一表征。

### 常见坑

- probe 容量过大 → 过拟合/记忆；
- 不做 random-label/position control；
- 用同一图的不同问题划分 train/test；
- 把 probe 高准确率直接说成“模型理解了”。

参考：[Probing Classifiers: Promises, Shortcomings, and Advances](https://arxiv.org/abs/2102.12452)；[Designing and Interpreting Probes with Control Tasks](https://arxiv.org/abs/1909.03368)。

---

## 5. Step 3：Latent 怎么“表征出来”？

这是本手册的核心问题：attention 可以覆盖回图像，latent 怎么办？

### 5.1 不要只画 PCA/UMAP

PCA/UMAP/t-SNE 能显示“是否按标签聚类”，但：

- 只描述几何，不证明因果；
- 受预处理、随机种子、perplexity 影响；
- 高维结构可能被压扁；
- 聚类可能来自位置/物体共现，而不是关系。

**结论：特征空间图只能作为“假设生成”，不能作为“诊断结论”。**

### 5.2 Per-token probe map（最像 attention coverage 的 latent 可视化）

做法：

1. 在某一层的每个 visual token 上取 hidden state；
2. 训练一个 probe（target / reference / relation / correctness）；
3. 对每个 token 输出 score；
4. reshape 回 9×9 或 27×27 网格；
5. 叠加到原图；
6. 对比 correct vs incorrect、source vs target、flip vs original；
7. 做 position-residualized 版本。

它能回答：

- 信息在哪些 token 上可解码；
- 高分 token 是否落在目标/参照物；
- 错误样本的 map 是否弥散/错位。

它不能回答：

- 模型是否真的用这些 token；
- 这些 token 是否因果必要。

### 5.3 Probe-direction projection

1. 训练 linear probe，取法向量 `w`；
2. 每个 token 投影 `score_i = w · h_i`；
3. 映射回网格；
4. 得到连续“关系方向热图”。

优点：比分类输出更平滑，能看到 spatial gradient；缺点：仍是相关性。

### 5.4 RSA / CKA：层间结构诊断

- 对每层构造 representation similarity matrix（RSA）；
- 用 CKA/SVCCA 比较不同层/条件；
- 比较 source vs target、flip vs original、correct vs incorrect；
- 找“关系信息开始分化”的层。

能回答：哪一层表征开始分离、层间是否共享结构。
不能回答：是否因果。

参考：[Similarity of Neural Network Representations Revisited](https://arxiv.org/abs/1905.00414)。

### 5.5 Logit lens / tuned lens：什么时候变成答案

- 把每层 hidden state 解码到 yes/no 或候选答案；
- 看答案概率在第几层出现；
- tuned lens 学习一个 per-layer affine translator，比 raw logit lens 更可靠。

能回答：答案在哪一层变得可读；readout 是否滞后。
不能回答：该层是否因果决定答案。

参考：[Eliciting Latent Predictions from Transformers with the Tuned Lens](https://arxiv.org/abs/2303.08112)。

### 5.6 SAE / dictionary learning：特征级分解

- 在 residual stream / visual token 上训练 sparse autoencoder；
- 得到稀疏 feature；
- 找 object/color/relation/direction feature；
- 对 feature 做 ablation/steering。

能回答：latent 由哪些可解释特征组成；是否存在 monosemantic relation feature。
不能回答：feature 是否完整、是否因果（需干预验证）。

参考：[Sparse Autoencoders Find Highly Interpretable Features](https://arxiv.org/abs/2309.08600)。

### 5.7 Latent 可视化方法选择表

| 你想知道 | 方法 | 是否因果 |
|---|---|---|
| 信息在不在 | probe / per-token probe | 否 |
| 在哪些 token | per-token probe map | 否 |
| 哪层开始分化 | RSA/CKA | 否 |
| 何时变成答案 | logit/tuned lens | 否 |
| 由什么特征组成 | SAE | 否（需干预） |
| 有没有被使用 | ablation/patching/steering | 是 |
| 是否唯一路径 | path patching / DAS / circuit | 是（更强） |

---

## 6. Step 4：因果必要性——没有它，输出会不会坏？

### 方法

| 方法 | 做法 | 适合 |
|---|---|---|
| **Zero/mean ablation** | 把 token/region/head 置零或替换为均值 | 快速必要性筛查 |
| **Activation patching** | 用 clean activation 替换 corrupted activation | 定位关键层/token |
| **Path patching** | 只 patch 某条路径（attention/MLP） | 区分路径 |
| **Interchange intervention** | 交换高维变量/子空间 | 因果抽象 |
| **DAS** | 学习低秩子空间并对齐因果变量 | 分布式表征 |

### 必做的对照

- 随机同尺寸 token/region/head；
- 随机方向/正交方向；
- 位置匹配/残差化；
- 多层/多 head 冗余检查；
- 同一干预在无关任务上的特异性。

### 判断

- ablation 掉点、随机不掉 → 必要；
- ablation 不掉、probe 高 → 冗余/未被使用/路径补偿；
- ablation 掉点但 patch 不恢复 → 可能不是充分表征；
- patch 恢复、随机不恢复 → 充分性证据。

参考：[Locating and Editing Factual Associations in GPT](https://arxiv.org/abs/2202.05262)；[Causal Abstractions of Neural Networks](https://arxiv.org/abs/2106.02997)。

---

## 7. Step 5：因果充分性——把它放回去，能不能恢复？

### 方法

1. **Source→target patching**：把正确样本的 activation/region 写入错误样本；
2. **Steering vector**：`h ← h + α·d`，d 为 source−target / correct−error 方向；
3. **Interchange intervention**：交换高维变量；
4. **DAS**：学习低秩子空间，交换子空间坐标。

### 必做对照

- 随机方向；
- shuffled-label 方向；
- 正交方向；
- 不同 α 的 dose-response；
- 位置/组别 held-out；
- 对无关任务的特异性。

### 判断

- 恢复 → 该表征足以支持答案；
- 不恢复 → 表征不充分，或 readout 不匹配；
- 只有高 α 恢复 → 可能是分布外/伪影，需谨慎。

---

## 8. Step 6：定位——层、head、token、区域

### 8.1 Attention 分析

- **Attention rollout/flow**：把多层 attention 传播到输入 token；
- **Relevancy maps**：梯度×attention，比 raw attention 更接近贡献；
- **Head ablation/amplification**：因果检验；
- **Attention pattern patching**：把正确 pattern patch 到错误样本。

必须区分：

- attention 高 ≠ 因果重要；
- attention 低 ≠ 信息不在（可能在 residual/MLP）；
- attention sink / register token 可能吸收大量 attention；
- 多头冗余常见。

参考：[Quantifying Attention Flow in Transformers](https://arxiv.org/abs/2005.00928)；[Transformer Interpretability Beyond Attention Visualization](https://arxiv.org/abs/2012.09838)；[Vision Transformers Need Registers](https://arxiv.org/abs/2309.16588)。

### 8.2 Visual token ablation

- 对单个/一组 visual token 做 zero/mean/ablate；
- 看 margin/accuracy 变化；
- 随机同尺寸 token 对照；
- 映射回图像，得到 causal heatmap。

这比 attention map 更接近“因果覆盖”。

### 8.3 VLM 专用工具

- LVLM-Interpret：层 attention、relevancy maps、CLEANN 因果解释；
- CLIP 文本分解：把 image representation 分解到文本概念；
- POPE/MMVP 等行为 benchmark 用于定位失败类型。

参考：[LVLM-Interpret](https://arxiv.org/abs/2404.03118)；[Interpreting CLIP's Image Representation](https://arxiv.org/abs/2310.05916)；[MMVP](https://arxiv.org/abs/2401.06209)；[POPE](https://arxiv.org/abs/2305.10355)。

---

## 9. Step 7：Readout——信息有没有变成答案

### 方法

| 方法 | 问题 |
|---|---|
| Final-token linear probe | 最终 token 是否线性可读 |
| Logit lens / tuned lens | 哪一层开始可读 |
| Margin / calibration | 排序对但阈值错？ |
| Flip consistency | 单图 readout 是否不稳定 |
| Ensemble / TTA | 多视图是否能稳定 readout |
| Selective prediction | 高置信样本是否更准 |

### 判断

- probe 高、margin 低 → readout/校准问题；
- margin 排序好、accuracy 低 → 阈值问题；
- flip 不一致 → 单图偏置/不稳定；
- logit lens 很晚才出现 → 前面层没有形成可读答案。

---

## 10. Step 8：泛化与稳健性

任何机制结论都要过：

1. **held-out images/videos**；
2. **held-out object/relation groups**；
3. **position-matched / residualized**；
4. **flip / counterfactual**；
5. **multiple seeds**；
6. **multiple comparisons correction**；
7. **effect size + CI**；
8. **cost**（token 数、参数量、额外前向）。

如果只在 dev 上有效、test 不行，不能叫机制。

---

## 11. 诊断决策树

```text
行为有 gap
├─ blank/black 高？
│   └─ 是 → 语言/选项先验；先做去偏/校准
├─ position-only 高？
│   └─ 是 → 位置捷径；位置残差化/匹配
├─ probe 可解码？
│   ├─ 否 → 表征/输入缺失；查 encoder/resolution/connector
│   └─ 是 → 做因果
│       ├─ ablation 掉点？
│       │   ├─ 是 → 必要；做 sufficiency/patching
│       │   └─ 否 → 冗余/未使用/路径补偿；查其他层/DAS/SAE
│       └─ patch/steer 恢复？
│           ├─ 是 → 充分；可做干预
│           └─ 否 → readout/整合问题；查 final token/logit lens
└─ readout 指标？
    ├─ 排序好、阈值差 → calibration
    ├─ flip 不稳定 → consistency/TTA
    └─ 都差 → 回到定位/整合
```

---

## 12. 泛化到模型其他方面

| 方面 | 要诊断什么 | 方法 | 结果含义 |
|---|---|---|---|
| **视觉输入/分辨率** | 信息是否在输入阶段丢失 | 分辨率/splitting/crop 消融；S0 probe | S0 高、S1 低 → connector/分辨率 |
| **Connector** | visual token 是否保留信息 | S0 vs S1 CKA/probe；S1 patching | S0 可解码、S1 不可 → connector loss |
| **定位** | 目标/参照物对应哪些 token | per-token probe map；oracle vs no-box；region ablation | oracle 高、no-box 低 → localization |
| **跨模态路由** | 文本 query 读哪些视觉 token | attention audit；head ablation；attention flow | attention 高且 ablation 掉 → routing |
| **关系/时序整合** | 左右、方向、先后是否组合 | relation/direction/order probe；patching；DAS | 单对象可读、关系不可读 → integration |
| **Readout** | 最终 token/margin/阈值 | final-token probe；logit/tuned lens；calibration | 信息可读但 margin 错 → readout |
| **语言先验** | 答案是否来自文本 | blank/black；text-only；option shuffle | blank 高 → prior |
| **评测捷径** | 位置/共现是否解释结果 | position-only；同图/同视频对照；残差化 | position 高 → shortcut |
| **幻觉/存在性** | 是否存在性判断偏差 | POPE/CHAIR；yes-rate；precision/recall | 高 precision 低 recall → conservative |
| **视频/动作** | 时序/方向/动作是否被使用 | frame ablation；order shuffle；black video；temporal probe | full > black → visual；direction 弱 → integration |
| **鲁棒性/OOD** | 换分布是否仍成立 | group-held-out；corruption；resolution shift | 只在 dev 成立 → 不泛化 |

---

## 13. 我们项目里的具体诊断计划

### 13.1 已有证据

- oracle region probe 0.812；position-only 0.961；
- no-box reader 0.44–0.55；
- attention head ablation 无效应；
- latent source−target steering 无稳定收益；
- flip consistency 0.600→0.690 / 0.633→0.813；
- POPE 0.783，TIFA coverage 0.788，TempCompass direction 弱。

### 13.2 下一步诊断（不是改进）

1. **Per-token probe map**
   - S1/S6/S12，target/reference/relation probe；
   - 映射回 9×9；
   - correct vs incorrect、source vs target、flip；
   - position-residualized。
2. **Layer-wise CKA/RSA**
   - source vs target、flip vs original、correct vs incorrect；
   - 找分化层。
3. **Region patching（必要性）**
   - target/reference/random region；
   - mean-replace / zero；
   - 随机 region 对照。
4. **Region swapping（充分性）**
   - source→target、target→source；
   - 随机 region 对照。
5. **Logit lens / tuned lens**
   - yes/no 在第几层出现；
   - 与 patching 层对齐。
6. **DAS / 低秩子空间**
   - 关系变量是否可因果中介；
   - 如果不行，说明分布式/非线性或未形成。

### 13.3 判断标准

- probe map 高 + ablation 掉 + patch 恢复 + random 不恢复 → 该 region/token 是因果瓶颈；
- probe map 高 + ablation 不掉 → 冗余/未使用；
- probe map 低 + oracle 高 → 定位/readout；
- attention 高 + ablation 不掉 → attention sink/冗余；
- 只有 oracle 有效、no-box 无效 → localization 是主瓶颈。

---

## 14. 常见反模式

1. **“Attention 看图这里，所以模型用了这里”** —— attention ≠ causal。
2. **“Probe 能读出关系，所以模型理解关系”** —— decodability ≠ use。
3. **“PCA 分开了，所以表征编码了关系”** —— 几何 ≠ 因果。
4. **“Ablation 掉点，所以这里唯一存储”** —— 可能是冗余/补偿。
5. **“Patch 恢复，所以这里就是机制”** —— 可能只是充分路径之一。
6. **“Test 上调到最好”** —— 选择偏差；dev 选，test 一次。
7. **“总体涨了就行”** —— 分组/位置/先验可能被平均掉。
8. **“负结果不写”** —— 负结果恰恰是排除候选、定位瓶颈的关键。

---

## 15. 一页 checklist

### 行为
- [ ] 指标定义清楚（accuracy/AUC/pair/coverage）
- [ ] 分组报告
- [ ] CI + 多种子
- [ ] blank/gray/flip/position 对照

### 可解码性
- [ ] linear probe
- [ ] random-label control
- [ ] position-only baseline
- [ ] group/image-held-out
- [ ] per-token probe map

### 几何
- [ ] CKA/RSA 层间比较
- [ ] source/target、correct/incorrect、flip
- [ ] 不把 PCA/UMAP 当因果

### 因果
- [ ] necessity：ablation/patching
- [ ] sufficiency：patch-in/steering
- [ ] specificity：random region/direction/head
- [ ] dose-response
- [ ] held-out

### 定位
- [ ] layer/head/token/region
- [ ] attention vs causal ablation
- [ ] visual token ablation

### Readout
- [ ] final-token probe
- [ ] logit/tuned lens
- [ ] calibration
- [ ] flip consistency

### 结论
- [ ] 写清楚“能支持/不能支持”
- [ ] 负结果也写
- [ ] 成本/可部署性

---

## 16. 参考文献

- Belinkov. *Probing Classifiers: Promises, Shortcomings, and Advances.* [arXiv:2102.12452](https://arxiv.org/abs/2102.12452)
- Hewitt & Liang. *Designing and Interpreting Probes with Control Tasks.* [arXiv:1909.03368](https://arxiv.org/abs/1909.03368)
- Kornblith et al. *Similarity of Neural Network Representations Revisited.* [arXiv:1905.00414](https://arxiv.org/abs/1905.00414)
- Meng et al. *Locating and Editing Factual Associations in GPT.* [arXiv:2202.05262](https://arxiv.org/abs/2202.05262)
- Geiger et al. *Causal Abstractions of Neural Networks.* [arXiv:2106.02997](https://arxiv.org/abs/2106.02997)
- Wang et al. *Interpretability in the Wild: a Circuit for Indirect Object Identification in GPT-2 small.* [arXiv:2211.00593](https://arxiv.org/abs/2211.00593)
- Belrose et al. *Eliciting Latent Predictions from Transformers with the Tuned Lens.* [arXiv:2303.08112](https://arxiv.org/abs/2303.08112)
- Cunningham et al. *Sparse Autoencoders Find Highly Interpretable Features in Language Models.* [arXiv:2309.08600](https://arxiv.org/abs/2309.08600)
- Abnar & Zuidema. *Quantifying Attention Flow in Transformers.* [arXiv:2005.00928](https://arxiv.org/abs/2005.00928)
- Chefer et al. *Transformer Interpretability Beyond Attention Visualization.* [arXiv:2012.09838](https://arxiv.org/abs/2012.09838)
- Darcet et al. *Vision Transformers Need Registers.* [arXiv:2309.16588](https://arxiv.org/abs/2309.16588)
- Ben Melech Stan et al. *LVLM-Interpret: An Interpretability Tool for Large Vision-Language Models.* [arXiv:2404.03118](https://arxiv.org/abs/2404.03118)
- Gandelsman et al. *Interpreting CLIP's Image Representation via Text-Based Decomposition.* [arXiv:2310.05916](https://arxiv.org/abs/2310.05916)
- Tong et al. *Eyes Wide Shut? Exploring the Visual Shortcomings of Multimodal LLMs.* [arXiv:2401.06209](https://arxiv.org/abs/2401.06209)
- Li et al. *Evaluating Object Hallucination in Large Vision-Language Models.* [arXiv:2305.10355](https://arxiv.org/abs/2305.10355)
