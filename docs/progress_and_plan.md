# 项目进度与计划

> 最后更新：2026-11-17（提交 `1ac3b25` 之后）
> 课程截止：2026-12-10
> 用途：给协作者和自己看的单一事实来源。实验数字以 `results/` 下的 JSON 和 summary 为准，
> 本文只做索引和决策记录。

---

## 1. 一句话状态

诊断阶段已经完成，结论稳定：**模型能编码物体身份、会部分使用目标区域，但没有把两个物体的位置组合成关系判断；答案信号只在第 18–19 层出现，且与决定标签的几何几乎不相关。**
下一步从“继续诊断”切换到“有针对性的干预”，主线是分解式提示 → 带中间步骤监督的 LoRA-SFT →（可选）DPO。

---

## 2. 状态总览

| 阶段 | 状态 | 产物 |
|---|---|---|
| 行为评测与同图对照 | 完成 | `results/readout_summary.md` |
| 外部 benchmark（POPE / TIFA / TempCompass / SSv2） | 完成 | `results/external_benchmarks_summary.md` |
| 诊断手册 | 完成 | `docs/diagnostic_handbook.md` |
| 逐 token probe map | 完成 | `results/diagnostics/per_token_probe.json` |
| Logit lens（24 层） | 完成 | `results/diagnostics/logit_lens_*.json` |
| 区域消融 / patching 汇总 | 完成 | `results/diagnostics/region_patching_aggregate.json` |
| 几何 vs 模型对照 | 完成 | `results/diagnostics/geometry_vs_model.json` |
| 诊断总结与图 | 完成 | `results/diagnostics_summary.md`、两张图 |
| 按组固定的逐 token 对照 | 完成（2026-11-18） | `results/diagnostics/per_group_token_probe.json` |
| 侧别 probe（第 18–19 层） | 完成（2026-11-18） | `results/diagnostics/side_probe.json` |
| 直接询问方位 | 完成（2026-11-18） | `results/diagnostics/side_report.json` |
| 给方位后的关系判断 | 完成（2026-11-18） | `results/diagnostics/side_given_relation.json` |
| 分解式提示 / 反转一致性 | **部分完成**：分解提示被证否；反转一致性未做 | — |
| LoRA-SFT 首轮（混合） | 完成（2026-11-18） | `results/sft/adapter_side`、`results/sft/eval_relation_sft.json` |
| 2×2 对照（side-only / relation-only） | 完成（2026-11-18） | `results/sft/bootstrap_deltas.json` |
| 跨组泛化 | 完成（2026-11-18） | `results/sft/eval_relation_crossgroup.json` |
| 全量 1 epoch | 完成（2026-11-18） | `results/sft/eval_relation_full.json` |
| 空白图对照 | 完成（2026-11-18） | `results/sft/eval_blank_base.json`、`eval_blank_full.json` |
| LoRA-SFT（中间步骤监督） | **未做** | — |
| DPO / 轻量 RL | **未做** | — |

---

## 3. 已完成实验与关键数字

### 3.1 行为层面

| 项目 | 结果 |
|---|---|
| 同图 source/target 对（测试核心） | margin AUC 0.600；pair acc 0.604 |
| 翻转一致性（唯一可部署提升） | AUC 0.600 → 0.690；strict 0.633 → 0.813 |
| POPE（300 题） | acc 0.783，AUC 0.890，yes 率 0.33；翻转合并 0.798 |
| TIFA/COCO（466 事实） | cued recall 0.753；描述覆盖 0.788；空间事实 0.538 对 0.692 |
| TempCompass（30 视频） | 是/否 0.605 对黑屏 0.377；direction 0.400/0.200，action 0.833/0.944 |
| Something-Something V2（30 视频） | pair acc 0.667 对黑屏 0.600；AUC 0.729 对 0.601 |

### 3.2 诊断层面

| 诊断 | 关键数字 |
|---|---|
| 几何 probe vs 模型 | 位置 AUC 0.991；模型 AUC 0.600；Spearman 0.183；符号一致 0.562 |
| 命名指向（9×9 逐格） | S1/S6/S12 = 0.540 / 0.595 / 0.619；随机 0.205；两个都对 0.317 / 0.358 / 0.398 |
| 查询物体归属 | AUC 0.582 / 0.584 / 0.587；argmax 命中 0.506 / 0.463 / 0.451（随机 0.390） |
| 关系答案（按关系类型） | 平均格 0.49–0.50；最好格 0.52–0.60；全 token 平均 0.48–0.51 |
| 模型正确性 | 平均 0.542–0.544；最好格 0.583–0.604；全 token 平均 0.582–0.588 |
| Logit lens（test core） | L14 0.493 → L18 0.636 → L19 0.646；最终 0.634 |
| Logit lens（strict 240 对） | 最终 0.548；L22 0.547 |
| 区域消融（S1） | target −0.382 [−0.465, −0.306]；reference −0.141；控制项 ≈0 |
| 区域 patching（S1） | target 0.333 [0.252, 0.464]；reference 0.201；无关 0.072；随机 0.003 |
| 翻转排序准确率 | 0.588；left 0.675，right 0.588，above 0.538，**below 0.357** |
| 匹配的全局读出器 | 全 token 平均 + 问句 0.474；目标+参照 pooled + 问句 0.492 |

### 3.3 能力判断

| 层级 | 判断 |
|---|---|
| 视觉编码 | 可用：物体身份在 token 上可解码 |
| 区域选择 | 部分可用：目标区域因果有效，但只承担约 1/3 的翻转差距 |
| 组合 | 缺失：单 token、全 token 平均都读不出关系 |
| 决策 | 偏晚且弱：第 18–19 层才出现信号 |
| 最终答案 | 与几何几乎不相关（ρ=0.18） |
| 先验/校准 | 偏保守，不是乱说 yes |

### 3.4 已修正的旧结论

- 早期把 per-group oracle region 的 0.81 当作“信息都在、只是没读出来”的证据。
  同一 global reader 下 pooled target+reference 只有 0.492；0.81 主要来自“每组各自学规则 + 读位置”。
- 真正可靠的上界是几何 probe 的 0.99，不是 0.81。
- 已在 `results/readout_summary.md` 和 `report/readout_gap_section.md` 中标注。

---

## 4. 未决问题（2026-11-18 更新）

1. ~~按组固定的逐 token probe~~ **已回答**：问句固定后平均格仍是 0.50–0.54，最好格 0.61–0.71，
   没有单个 token 能稳定回答。
2. ~~侧别概念在哪一层可解码~~ **已回答，并且是混合的**：竖直侧别在 L18–19 可解码
   （0.75–0.81），水平侧别只有 0.65/0.56；连续 x 坐标 R²≈0。视觉 token 本身水平侧别
   AUC 0.95，说明信息在、决策层没读到。
3. **直接询问方位**：模型水平方向 960/960 回答 "left"，竖直方向是 "top" 先验；
   选项顺序反转不改变答案。也就是说它不能报方位。
4. **组合是否完好**：给真方位时关系准确率 0.76（两侧不同半区 0.92），给反方位掉到 0.27/0.12。
   组合是完好的，缺的是把方位读出来。
5. **定位误差 vs 组合误差的占比**：仍未直接测量；区域 patching 只是间接证据。

---

## 5. 计划（2026-11-18 起）

### 阶段 0：补对照（11/18–11/20，约 1 天）

- 按组固定的逐 token probe；
- 第 18–19 层的侧别 probe；
- 冻结评测协议：pair accuracy/AUC、strict、flip、位置残差化、空白图。

**判据**：得到“有概念/没概念”的明确结论，决定阶段 2 是“激发”还是“教学”。

### 阶段 1：零训练干预（11/18 已完成主要判定）

- ~~分解式提示：先问 X 在哪侧、Y 在哪侧，再问关系~~ **证否**：模型不能报方位
  （水平恒定 "left"，竖直 "top" 先验），子问题不可答，分解提示无法成立。
- **给方位后的关系判断**：真方位 0.76（两侧不同半区 0.92），反方位 0.27/0.12。
  组合能力完好，问题被定位到“方位读出”。
- 问题侧反转一致性：仍未做；优先级下调（组合已证明可用）。

**结论**：阶段 1 的“提示激发”路径关闭；直接进入阶段 2，但训练目标改为
**先教方位报告（visual → text），再教关系**。

### 阶段 2：LoRA-SFT（11/18 首轮完成）

**首轮结果（4,000 步，14 分钟，5.55GB）：**

| adapter | 方位 水平/竖直 | 关系 acc / AUC |
|---|---|---|
| base | 0.506 / 0.492 | 0.569 / 0.600 |
| side-only | 0.645 / 0.682 | 0.617 / 0.717 |
| relation-only | 0.551 / 0.492 | 0.683 / 0.777 |
| mixed | 0.689 / 0.704 | 0.656 / 0.750 |

- 方位训练迁移到关系（ΔAUC +0.117，CI 不跨 0）；关系训练不提升方位。
- 直接关系监督更强（ΔAUC +0.177），但机制证据来自 side-only。
- 详见 `docs/sft_experiment_log.md` 与 `results/diagnostics_summary.md` §11。

数据（程序化生成，不需要人工标注）：

- **新增第一阶段目标：方位报告**。输入图像 + “Is the X on the left half or right half?”，
  监督 GQA 框导出的 left/right/top/bottom；这是模型当前完全缺失的能力。
- 第二阶段：关系判断，复用已验证可用的组合能力；“X 在左半；Y 在右半；所以 yes/no”。
- 从 GQA scene graph 生成同图配对、镜像、问句反转、难负例；位置平衡。

训练：

- 冻结视觉塔，只训 LoRA；
- 2.2B + 梯度检查点 + batch 1；先 1–2 个 epoch 的小规模验证；
- 8GB 显存下预计可行；不够则 QLoRA。

**判据（缺一不可）**：

1. held-out pair accuracy / AUC 提升，CI 不跨 0；
2. 位置残差化后的 AUC 也提升；
3. 翻转一致性不下降；
4. logit lens 显示答案信号前移（例如 L19 → L14–15）；
5. POPE / TIFA 不回归。

**回退**：若 SFT 只学到格式、判据 2 或 4 不满足，转比较器模块（冻结主干 + 双区域比较头），
把它作为机制对照写入报告。

### 阶段 3：DPO / 轻量 RL（12/1–12/4，可选）

- 只在阶段 2 证明“会比较但不够稳”时做；
- 奖励：配对正确性 + 翻转一致性 + 格式；
- 优先 DPO（两次前向），GRPO 在 8GB 上代价过高。

### 收尾（12/4–12/9）

- 最终评测与置信区间；
- 报告整合：诊断链 + 干预结果 + 限制；
- 若干预失败，报告仍以诊断研究成立（课程要求不强制性能提升）。

---

## 6. 资源与约束

| 项目 | 现状 | 影响 |
|---|---|---|
| GPU | RTX 3070 8GB | LoRA 可行但紧；GRPO 基本不可行 |
| 本地 GQA 图 | val 子集 293 张；**train 74,256 张已下载**（22 个 parquet 分片，9.8GB，`data/raw/gqa/train_all_images`） | SFT 数据已就绪；其中 73,761 张可生成关系对 |
| 本地配对 | dev 1206 对；test 240 对 | 只够评测，不够训练 |
| GQA train scene graph | 已有（`data/raw/gqa/sceneGraphs.zip`） | 配对脚本可直接复用 |
| 网络 | HF 直连不可用；Stanford 源约 15KB/s；ModelScope 镜像约 10MB/s | 大数据一律走国内镜像（`tools/download_gqa_train_images.sh`） |
| 时间 | 距今约 3 周 | 只够一条主线 + 一个回退 |

---

## 7. 决策点

1. **是否投入数据下载与 SFT？**
   收益：直接针对诊断出的瓶颈；风险：3 周内可能只得到负结果。
   建议：先做阶段 0/1（3–4 天），若提示无效再决定是否投入 SFT。
2. **是否加另一个 VLM 做对比？**
   它回答“是否 SmolVLM2 特有”，不解决提升；建议放到报告后期，选同量级模型 4-bit 跑同一套诊断。
3. **是否把比较器模块作为主线？**
   若目标是“机制清晰 + 可复现”，模块更可控；若目标是“模型本身变强”，SFT 更合适。
   当前倾向：SFT 主线，模块作为回退与对照。

---

## 8. 协作约定

- 结果以 `results/*.json` 为准，summary 只做索引；新实验先写脚本再写结论。
- 所有 probe 必须按 image id 分组交叉验证；所有提升必须报告位置残差化与翻转一致性。
- 提交信息用“动词 + 对象”，一次提交只做一件事。
- 大文件（`data/raw`、`data/fixed/images`、`models/`、`results/readout/*.pt`）不入库。

---

## 9. 变更记录

| 日期 | 变更 |
|---|---|
| 2026-11-17 | 建立本文档；记录诊断结论、修正 0.81 的解读、确定三阶段计划 |
| 2026-11-17 | 从 ModelScope 下载 GQA train 图（74,256 张，9.8GB），SFT 数据就绪 |
| 2026-11-18 | Stage 0/1：侧别 probe、按组逐 token probe、直接询问方位、给方位后的关系；结论改为“方位读出”瓶颈，SFT 目标调整为先教方位报告 |
| 2026-11-18 | SFT 首轮 + 2×2 对照：side-only 迁移到关系（ΔAUC +0.117），relation-only 不提升方位；直接关系监督最强（+0.177） |
