# CS224N 精读进度台账

## 当前 goal

完成 15 讲 slides、现有 notes 与各讲扩展阅读的中文精读，并写入 Obsidian。当前阶段优先复核 Lecture 01–02 与 ass1 学习闭环。

## 已核对材料规模

- Slides：15 份，共 847 页。
- Notes：7 份，共 95 页；Lecture 06 与 Lecture 07 共用同一份 notes。
- External readings：`external-readings/manifest.json` 登记 69 项；其中网页、notebook、论文及不可访问条目按来源状态分别记录。

## 阶段状态

| 阶段 | 状态 | 证据/下一步 |
|---|---|---|
| Lecture 01 slides 逐页清单 | 已有底稿，待图示复核 | Obsidian `lec1 & lec2.md` 的 L1-P01–P36 |
| Lecture 01 notes 逐页复核 | 本批完成文字层核对 | Obsidian 的 L1N-P01–P13 |
| Lecture 02 slides 逐页清单 | 已有底稿，待图示复核 | Obsidian 的 L2-P01–P46 |
| Lecture 02 notes 逐页复核 | 本批完成文字层核对 | Obsidian 的 L2N-P01–P13 |
| A1 知识闭环 | Q1.1–Q2.8 已在兼容环境完成一次实际运行并写入 lec1/lec2 | Q2.9 仍是知识/方案入口，未运行去偏干预；不等同于提交 notebook |
| Lecture 03 | slides p.01–p.66 图像复核完成；独立分讲笔记已同步逐页底稿；toy corpus、NumPy 与 A1 `TruncatedSVD` 已复算 | Q1.4/Q1.5 图已写入 Vault；继续做全课程缺口清点 |
| Lecture 04 | slides p.01–p.96 与 notes p.01–p.18 已核读并写入 | 8 项扩展阅读中 7 项已完成；仅 Karpathy 原文缺口 |
| Lecture 05 | slides p.01–p.53、notes p.01–p.05 已完成文字/图像核读；独立笔记已建立；扩展阅读 6/7 已完成 | 仅剩 Springer 全文缺口；后续继续做全文缺口清点 |
| Lecture 06 | slides p.01–p.54、notes p.01–p.14 已完成文字/图像核读并建立分讲笔记；扩展阅读 4/4 已完成 | 已验收；进入 Lecture 07 |
| Lecture 07 | slides p.01–p.57、共享 notes p.01–p.14 已完成；扩展阅读 4/4 已完成 | Bengio et al. (1994) 已通过公开作者存档取得完整 35 页并核读；未运行原论文实验 |
| Lecture 08 | slides p.01–p.44、扩展阅读 Chapter 11 已完成并写入 | 已验收；下一项为 Lecture 09 |
| Lecture 09 | slides p.01–p.47、notes p.01–p.18、扩展阅读 7/7 已完成 | 已验收；下一项为 Lecture 10 |
| Lecture 10 | slides p.01–p.61；BERT、contextual、Illustrated BERT 与官方 2025 归档 Masked Language Models 已完成并写入 | 清单写 Chapter 11、官方 2025 归档实际为 Chapter 10；当前 `11.pdf` 是 RAG/IR，版本差异已登记；Colab 仅作入口 |
| Lecture 11 | slides p.01–p.61 与 5 项扩展阅读已完成并写入；正文/图表/实质性附录均已核读 | 未运行论文代码或 benchmark；本讲已满足材料验收 |
| Lecture 12 | slides p.01–p.60、扩展阅读 5/5 已完成 | 已验收；下一项进入 Lecture 13 |
| Lecture 13 | slides p.01–p.59、扩展阅读 4/4 已完成 | 已验收；下一项进入 Lecture 14 |
| Lecture 14 | slides p.01–p.53、扩展阅读 6/6 已完成 | 已验收；下一项进入 Lecture 15 |
| Lecture 15 | slides p.01–p.54、扩展阅读 6/6 已完成 | 已验收；全课程材料复核与缺口清点继续 |

## 每批恢复规则

1. 读取本台账、目标 Obsidian 文件和材料 manifest。
2. 从最早未完成页或论文小节继续；写入前保留用户新增内容。
3. 写入后回读并更新本台账和目标笔记的执行记录。
4. 只把可由原文或实际运行支持的内容标为已完成；不可访问来源保留缺口。

## 最近检查点

- 2026-09-12 06:54（Asia/Shanghai）：在夜间窗口结束前的收尾核验中，发现 Bengio–Simard–Frasconi (1994) 的 PDF 已在 `external-readings/07-advanced-variants-of-rnns-attention/`，但目标 Vault 缺少对应精读页；已补写 `reading - Bengio Simard Frasconi 1994 Long Term Dependencies.md`，包含 35 页页码清单、Jacobian 连乘、稳定性/可训练性 trade-off、算法比较、Lecture 06–07 回链和未复现实验边界，并回读 Lecture 07 链接与附件索引。当前时间已超过 06:00，停止新的长操作；Karpathy 原文与 Springer 章节缺口保留给下一次夜间批次。
- 2026-09-12 06:55（Asia/Shanghai）：对 Vault `/Users/xiaohufeng/Documents/model_eval` 下 14 个 `lec*.md`（Lecture 01–02 为合并文件）执行 Obsidian wikilink/图片目标审计；去除 heading-only 链接后共检查到的目标均可解析，未发现断链。`cs224n/assets/...` 路径按实际 Vault 根目录 `/Users/xiaohufeng/Documents/model_eval` 解析正确。
- 2026-09-12 06:57（Asia/Shanghai）：用 `pdfinfo` 重算本地课程材料：15 份 slides 共 847 页、7 份 notes 共 95 页，与总索引一致；将 `external-readings/manifest.json` 中 Bengio–Simard–Frasconi (1994) 从 `skipped/paywalled` 修正为 `downloaded`，登记公开作者存档 URL、本地 PDF 路径与 321,695 bytes，并通过 `jq` 验证 JSON 有效。当前 manifest 计数为 exists 80、downloaded 3、failed 2、skipped 2。
- 2026-09-12 06:59（Asia/Shanghai）：逐讲解析 Vault 页码标记并与 PDF 页数比对：Lecture 01–03、05–15 的 slides/notes 标记均覆盖从 1 到实际总页数且无越界；Lecture 04 slides 覆盖 96/96，notes 使用 `notes p.01`–`notes p.18` 格式，另行核验为 18/18。未发现漏页；此前使用过窄 heading 正则得到的 Lecture 04 notes 假阴性已确认是标记格式差异，不是内容缺失。
- 2026-09-12 07:01（Asia/Shanghai）：为两个仍不可取得全文的直接阅读建立 Vault 缺口页：`reading - Karpathy Yes You Should Understand Backprop - Access Gap.md` 与 `reading - Kubler McDonald Nivre Dependency Parsing - Access Gap.md`，分别记录原始来源、HTTP/付费访问边界、不得用替代材料冒充精读的规则和下一次恢复动作；Lecture 04/05 已回链到对应缺口页。
- 2026-09-12 07:03（Asia/Shanghai）：在课程总索引新增“当前完成边界”区，明确 847/847 slides、95/95 notes、两个真实全文缺口和 Masked LM 版本差异；同时修正 lec11/lec12 两个旧 wikilink 文件名，索引目标审计结果为 `missing=[]`。
- 2026-09-12 07:04（Asia/Shanghai）：扫描 Vault 全部 Markdown 的未勾选事项并分类：剩余 13 项均是“未运行论文代码/benchmark”或明确的实验边界，不是逐页阅读缺口；已将 Lecture 06 三项历史待办、Lecture 05 的 Springer 访问确认、Andor 的旧“需继续”记录和 Chapter 11 版本错配说明修正为当前状态。修复 Andor→Chen–Manning 断链后，全 Vault wikilink 审计为 `missing=0`。
- 2026-09-12 07:05（Asia/Shanghai）：再次检索“保持未完成/需继续”等旧状态；确认 Lecture 04 的 7/8、Lecture 05 的 6/7 是真实扩展阅读缺口，Lecture 10 的 RAG/IR 文件是版本边界记录，其余未勾选项均明确写的是未运行实验。未发现新的逐页或逐节阅读缺失。

- 日期：2026-09-12。
- 已复核：Lecture 01 notes 与 Lecture 02 notes 的文字层、已有页码入口和 A1 对照条目。
- 已复核：Lecture 01/02 slides 的关键图示与公式（分布式语义、共现矩阵、SVD、GloVe、类比、多义词）；未发现与现有页码条目的矛盾。
- 已开始：Lecture 03 Python Review，PDF p.46–p.54 的 NumPy/shape/广播图示已复核，并已建立独立入口 `cs224n/lec03 Python Review.md`。
- 已复算：Lecture 03 p.53–p.54 三种广播结果与 A1 行归一化 shape；使用工作区依赖运行时，未将缺少 NumPy 的系统 Python 当作验证环境。
- 已复算：A1 notebook 自带 toy corpus 的 Q1.1–Q1.2 sanity check；排序词表、`word2ind`、`M.shape=(10,10)`、对称计数和关键行与期望矩阵逐项一致，并补测重复词形导致非零对角线的边界语义。
- 已复核：Lecture 03 PDF p.25–p.43 的 Python 基础/数据结构图示；记录 PY-P38 索引输出疑似笔误，并以实际 Python 0-based 语义为准；补充 Python 3.8+ 字典插入顺序与显式 `sorted` 词表的区别。
- 已复核：Lecture 03 slides p.01–p.66 页面图像总览；重点代码/图示页 p.25–p.43、p.46–p.54 做高分辨率检查，未发现除 PY-P38 外的内容矛盾。
- 已核验：用 NumPy full SVD 复核 $U_k$、$\Sigma_k$、$V_k^\top$ 与 $U_k\Sigma_k$ 的 shape；随后在 `/Users/xiaohufeng/miniconda3/envs/cs224n`（scikit-learn 1.9.0）实际运行 A1 的 `TruncatedSVD(n_components=2,n_iters=10)`，确认 IMDB 前 150 条评论的 $M:(5880,5880)$、输出为 $(5880,2)$，并生成 Q1.5 图。
- 已完成：A1 Q2.1–Q2.8 实际运行核验；加载 `glove-wiki-gigaword-200`（400,000×200），记录 Q2.2 `leaves`、Q2.3 `hot/warm/cold`、Q2.4 grandfather 类比、Q2.5 France–Paris 类比、Q2.6 fish–water 失败类比，以及 Q2.7/Q2.8 偏见查询结果；图和具体输出已写入目标 lec1/lec2 笔记。Q2.9 未执行去偏干预。
- 已生成并回读：`/Users/xiaohufeng/Documents/model_eval/cs224n/assets/ass1-q1-5-run-2026-09-12.png` 与 `ass1-q2-1-run-2026-09-12.png`；标注、坐标轴和二维投影局限已核对。
- 已开始：Lecture 04，已建立独立笔记 `cs224n/lec04 Backpropagation and Neural Network Basics.md`，登记 slides 96 页、notes 18 页和 8 项扩展阅读；slides p.01–p.30 已完成首批文字/图示核读。
- 已复核：Lecture 04 slides p.31–p.58 的 Jacobian、链式法则、上游梯度、外积与 shape convention；图像与文字层一致，公式已写入独立笔记。
- 已核验：对 $s=u^T\tanh(Wx+b)$ 做解析梯度与中心差分 toy 检查，最大绝对误差约 $7.5\times10^{-11}$；仅作为公式核验，不计作 A2 实验。
- 已复核：Lecture 04 slides p.59–p.96 的计算图、单节点反向传播、共享分支梯度相加、自动微分 API、数值梯度检查和总结页；动画阶段差异已写入独立笔记。
- 已完成：Lecture 04 notes p.01–p.18 逐页补充，覆盖 max-margin、逐元素/向量化反传、gradient check、L2/dropout、激活函数、预处理、Xavier、学习率、momentum、AdaGrad、RMSProp 和 Adam；notes 与 slides 的目标函数/参数差异已标出。
- 已完成：Lecture 04 扩展阅读《Derivatives, Backpropagation, and Vectorization》7 页逐页精读，并建立独立 Obsidian 笔记；重点补充张量 Jacobian、巨大 Jacobian 的内存问题和 vector-Jacobian product。
- 已完成：Lecture 04 扩展阅读 CS231n《Backpropagation, Intuitions》按网页小节精读，并建立独立 Obsidian 笔记；重点补充基本门、链式法则、sigmoid 模块化、分叉梯度累加和线性层向量化反传。
- 已完成：Lecture 04 扩展阅读 CS231n《Neural Network Architectures》按网页小节精读，并建立独立 Obsidian 笔记；重点补充单神经元、激活函数、全连接层的矩阵形状、表达能力与容量/正则化关系。
- 已完成：Lecture 04 扩展阅读 Rumelhart、Hinton、Williams (1986)《Learning Representations by Back-propagating Errors》4 页逐页精读，并建立独立 Obsidian 笔记；扫描版文字层不可用，已通过页面图像核读公式、图注、权重更新和图表边界，明确 p.4 右半的无关论文不属于本文。
- 已完成：Lecture 04 扩展阅读 Collobert 等人 (2011)《Natural Language Processing (Almost) from Scratch》共 45 页；正文、实验、讨论和实质性附录 p.1–p.40 已完成逐页文字/图示核读，p.41–p.45 参考文献已登记为检索入口，独立笔记覆盖四个 benchmark、统一网络、embedding、ranking LM、MTL、任务工程、SENNA、句法特征和逐层梯度。
- 已完成：Lecture 04 官方 matrix calculus notes，PDF 7 页；逐页核读并写入 `cs224n/reading - Clark CS224N Gradient Notes.md`。
- 已完成：Lecture 04 官方 Review of differential calculus，PDF 10 页；逐页核读并写入 `cs224n/reading - Genthial Differential Calculus Review.md`。
- 已登记缺口：Lecture 04 仅剩 Karpathy《Yes you should understand backprop》原文，原站 403、存档请求受限；未用二手摘要替代全文。
- 已尝试：Lecture 04 的 Karpathy backprop 原文再次通过 Medium 官方 URL 获取，返回 Cloudflare challenge 页面而无文章正文；继续保留全文缺口，不以二手摘要冒充精读。
- 已核验：本地 Vault、课程材料目录与 Git 历史中没有额外的 Springer *Dependency Parsing* 或清单同名 Chapter 11 副本；Bengio (1994) 的旧缺口已由公开作者存档补齐，Masked LM 则由官方 `old_aug25/10.pdf` 补齐。当前仍需保留的全文缺口是 Karpathy backprop 原文与 Springer *Dependency Parsing*。
- 已回读：Lecture 04 主笔记与 7 篇已完成扩展阅读笔记，确认逐页/逐节标题、Obsidian 链接和最近状态可保存；未发现残留 `+` 标记，工作区台账通过 `git diff --check`。
- 已开始：Lecture 05 Dependency Parsing，建立独立笔记 `cs224n/lec05 Dependency Parsing.md`；slides p.01–p.32 已完成文字/图像复核，封面“Lecture 4”与目录“Lecture 05”的编号差异已记录。
- 已完成：Lecture 05 notes p.01–p.05 逐页核对，补入 learning/parsing 子问题、$c=(\sigma,\beta,A)$、Shift/Left-Arc/Right-Arc 前置条件、嵌入矩阵维度与 18/18/12 特征计数。
- 已完成：Lecture 05 slides p.01–p.53 与 notes p.01–p.05 逐页核读并写入 `cs224n/lec05 Dependency Parsing.md`；覆盖附着歧义、依存树/项目性、treebank、transition/graph-based parser、UAS/LAS、神经特征与 biaffine 预告。
- 已完成：Lecture 05 扩展阅读 Chen & Manning (2014) 论文 p.1–p.9 正文/图表精读，p.10–p.11 参考文献边界登记；建立独立笔记 `cs224n/reading - Chen Manning 2014 Neural Dependency Parser.md`，明确论文 92.2/92.0 等不同实验口径和 cube/ReLU 差异。
- 已完成：Lecture 05 扩展阅读 Nivre《Incrementality in Deterministic Dependency Parsing》8 页正文/图表精读；建立独立笔记，明确 strict incrementality 不可能、arc-eager 最优性及 68.9%/87.1% 分母差异。
- 已完成：Lecture 05 扩展阅读 Andor et al.《Globally Normalized Transition-Based Neural Networks》12 页核读；建立独立笔记，覆盖 local/global CRF、label bias 严格包含证明、lookahead 局限、beam/early update、full backprop 与 POS/解析/压缩实验；实验数字明确为论文报告，未复现。
- 已完成：Lecture 05 扩展阅读 de Marneffe et al.《Universal Dependencies: A cross-linguistic typology》8 页逐页精读；建立独立笔记，覆盖 UD taxonomy、lexicalism、case/preposition 平行性、跨语言 scheme mapping 与 basic/enhanced/parsing representation。
- 已完成：Lecture 05 扩展阅读 Universal Dependencies 本地主页快照精读；建立独立笔记，记录 UD v2、release/treebank 元信息、UPOS/FEATS/DEPREL/CoNLL-U 入口，并明确动态子页面未冒充已读。
- 已完成：Lecture 05 扩展阅读 Jurafsky & Martin Chapter 19 共 26 页逐页精读；建立独立笔记，覆盖 CFG/CNF、treebank、结构歧义、CKY、神经 span parser、PARSEVAL 和 head-finding；实现与 evalb 未运行。
- 已验收：Lecture 05 页面笔记 53/53、notes 5/5；扩展阅读笔记页数 Andor 12/12、UD 8/8、Jurafsky–Martin 26/26，Nivre 8/8，Chen–Manning 正文/图表 p.1–p.9 且参考文献边界登记；本讲 wikilinks 和关键公式已回读。
- 已更新：Springer *Dependency Parsing* 全文仍是 Lecture 05 的唯一缺口；Lecture 04 当前仅保留 Karpathy 原文这一项不可访问来源。
- 已开始：Lecture 06 Language Models and RNNs，确认 slides 54 页、notes 14 页；建立 `cs224n/lec06 Language Models and RNNs.md`。
- 已完成：Lecture 06 slides P01–P16 逐页文字/图像核读；覆盖 LM 定义/链式法则、next-word 能力示例、n-gram、Markov 假设、计数估计、smoothing/backoff 和存储问题。
- 已完成：Lecture 06 slides P17–P24 逐页文字/图像核读；覆盖 n-gram autoregressive sampling、perplexity、fixed-window neural LM、embedding 拼接、隐藏层与词表 softmax 维度。
- 已完成：Lecture 06 slides P25–P32 逐页文字/图像核读；覆盖 fixed-window 局限、RNN 权重共享、hidden state、RNN LM 公式、逐时间步交叉熵与展开训练图。
- 已完成：Lecture 06 slides P33–P40 逐页文字/图像核读；覆盖 teacher forcing、按句/批次 SGD、共享参数梯度求和、多变量链式法则、BPTT 截断与自回归生成 rollout。
- 已完成：Lecture 06 slides P41–P54 逐页文字/图像核读；覆盖 vanishing/exploding gradients 的时间链式法则、线性情形 $W_h^\ell$ 与特征值证明草图、长距离依赖、norm clipping、LSTM/attention/residual 预告和 recap。物理 PDF 页码与课件页脚在该段跳号，已在分讲笔记中明确。
- 已完成：Lecture 06 notes N01–N14 逐页文字/图像核读；补充 n-gram 的 MLE/PPL/smoothing、RNN 维度与 Jacobian 范数界、双向/深层 RNN、encoder–decoder、GRU/LSTM 方程与直觉，并标明共享 notes 的 Lecture 07 先修边界。
- 已完成：Lecture 06 扩展阅读 4/4：Jurafsky–Martin Chapter 3（26 页）、Karpathy RNN article、Goodfellow Chapter 10、Norvig “Two Cultures”；均建立独立 Obsidian 笔记，未将未复现实验写成事实。
- 已验收：Lecture 06 slides 54/54、notes 14/14、扩展阅读 4/4；下一项为 Lecture 07 Advanced Variants of RNNs and Attention。
- 已完成：Lecture 07 slides P01–P57 逐页文字/图像核读；P40–P51 attention 动画重复页逐页登记，页脚编号跳变与课件内部 Lecture 6 编号差异已记录。
- 已完成：Lecture 07 复用 Lecture 06 notes N01–N14；LSTM、双向/多层 RNN、seq2seq、attention 的共享先修入口已建立。
- 已完成：Lecture 07 扩展阅读 4/4：Pascanu et al. (2013) 12 页、Vaswani et al. (2017) 15 页、Stanford vanishing-gradient notebook 静态 HTML、Bengio–Simard–Frasconi (1994) 公开作者存档 35 页；均建立独立 Obsidian 笔记，未将未复现实验写成事实。
- 已更正：Bengio et al. (1994) 的 IEEE 版本仍可能受版权访问限制，但已取得并核读公开作者存档；原先“未取得公开版本”的记录由本条和 2026-09-12 检查点 supersede。
- 已验收：Lecture 07 主笔记、4 份扩展阅读笔记和台账回读；下一项为 Lecture 08。
- 已完成：Lecture 08 slides P01–P44 逐页文字/图像核读；项目选择、proposal/milestone/report、数据来源、数据划分、训练策略与结束页均登记。
- 已完成：Lecture 08 扩展阅读 Goodfellow Chapter 11《Practical Methodology》§11.1–§11.6；建立独立笔记，覆盖指标、baseline、数据量、超参搜索、debugging、gradient check 与案例。
- 已验收：Lecture 08 主笔记、扩展阅读笔记和台账回读；下一项为 Lecture 09 Transformers。
- 已完成：Lecture 09 slides P01–P47 与 notes N01–N18 逐页文字/图像核读；P45“是否需要消除二次成本”及 notes 参考文献边界均单独登记。
- 已建立：Lecture 09 主笔记覆盖 self-attention、位置表示/RoPE、FFN、causal mask、多头、scaled dot-product、残差、LayerNorm、encoder/decoder/cross-attention 与复杂度边界。
- 已完成：Lecture 09 扩展阅读 7/7：Vaswani 论文共享笔记、Illustrated Transformer、Google AI blog、LayerNorm、Image Transformer、Music Transformer、Jurafsky–Martin Chapter 9；均建立独立 Obsidian 入口，未将未复现实验写成事实。
- 已验收：Lecture 09 主笔记、notes、7 项扩展阅读和台账回读；下一项为 Lecture 10 Pretraining。
- 已完成：Lecture 10 slides P01–P61 逐页登记与文字/图像总览；固定词表/BPE、contextual representation、pretrain/finetune、MLM、BERT、T5、GPT、in-context learning 与 scaling 已写入主笔记。
- 已完成：Lecture 10 扩展阅读 2/4：Devlin et al. (2019) BERT（16 页正文/实验/附录）与 Contextual Word Representations（15 页）；均建立独立 Obsidian 笔记，未将论文结果写成复现实验。
- 已完成：Lecture 10 Illustrated BERT/ELMo 网页按原文小节精读并写入 `reading - Illustrated BERT ELMo and co.md`。
- 已核验：清单文件名为 `jurafsky-and-martin-chapter-11-masked-language-models.pdf`，但当前 PDF 内部为 2026-08-19 草稿的 Chapter 11 *Information Retrieval and Retrieval-Augmented Generation*（25 页）；该版本实际内容已写入 `reading - Jurafsky Martin Chapter 11 Retrieval-based Models.md`。同时通过官方 `old_aug25/10.pdf` 取得并精读目标 Masked Language Models 章节（20 页），写入 `reading - Jurafsky Martin Chapter 10 Masked Language Models 2025.md`；章节号/版本差异保留在两份笔记中。
- 已验收：Lecture 10 直接扩展阅读 4/4；L10 的版本错配已解释，不再把 Masked LM 标为缺口。
- 已完成：Lecture 11 slides p.01–p.61 逐页文字/图像核读；重复 Lecture Plan、动画连续页、DPO 推导、reward hacking 和标注偏差页面均已登记。
- 已写入：`cs224n/lec11 Post-training RLHF SFT DPO.md`，覆盖 SFT、Bradley–Terry RM、REINFORCE、KL-regularized RLHF、DPO 公式与边界；未运行 PPO/DPO 实验。
- 下一项：Lecture 11 五项扩展阅读逐篇精读；完成后进入 Lecture 12。
- 已完成：Lecture 11 扩展阅读 Rafailov et al. (2023) DPO，PDF 27 页；正文、实验设置、理论附录 A–C 已核读，参考文献列表仅登记边界。
- 已写入：`cs224n/reading - Rafailov 2023 DPO.md`，补充 KL 约束闭式解、reward equivalence class、DPO gradient、实验 proxy 与未复现边界。
- 下一项：继续 Lecture 11 剩余 4 项扩展阅读；本讲暂不完全验收。
- 已完成：Lecture 11 扩展阅读 Chung et al. (2022) FLAN，PDF 54 页；正文、图表、讨论与实质性附录已核读，参考文献列表仅登记边界。
- 已写入：`cs224n/reading - Chung 2022 Scaling Instruction-Finetuned Language Models.md`，补充 task/model scaling、CoT 混合、zero-shot reasoning、开放式人评与 Responsible AI 边界。
- 下一项：继续 Lecture 11 剩余 3 项扩展阅读；本讲暂不完全验收。
- 已完成：Lecture 11 扩展阅读 Dubois et al. (2023) AlpacaFarm，PDF 31 页；正文、实验、讨论和实质性附录已核读，参考文献列表仅登记边界。
- 已写入：`cs224n/reading - Dubois 2023 AlpacaFarm.md`，补充 simulator 设计、annotator noise、方法 ranking、reward over-optimization 与 faithfulness–performance trade-off。
- 下一项：继续 Lecture 11 剩余 2 项扩展阅读；本讲暂不完全验收。
- 已完成：Lecture 11 扩展阅读 Wang et al. (2023) *How Far Can Camels Go?*，PDF 23 页；正文、主要图表、结论和实质性附录已核读，参考文献列表仅登记边界。
- 已写入：`cs224n/reading - Wang 2023 How Far Can Camels Go.md`，补充公开 instruction 数据集、assistant-only loss、多维评估、T ÜLU、长度偏差与安全评估。
- 下一项：完成 Lecture 11 最后一项 OpenAI *Aligning language models to follow instructions*；本讲暂不完全验收。
- 已完成：Lecture 11 扩展阅读 Ouyang et al. (2022) InstructGPT，PDF 68 页；正文、实验、讨论与实质性附录已核读，参考文献列表仅登记边界。
- 已写入：`cs224n/reading - Ouyang 2022 InstructGPT.md`，补充 SFT/RM/PPO/PPO-ptx、alignment tax、truthfulness/toxicity/bias 评估和“对齐谁”的限制。
- 已验收：Lecture 11 slides p.01–p.61、直接清单 5/5 扩展阅读、独立笔记和台账均已回读；下一项进入 Lecture 12。
- 已开始：Lecture 12 Efficient Adaptation，确认 slides 物理页数 60；课件内部标题/页脚称 Lecture 11，已在独立笔记中登记编号差异。
- 已完成：Lecture 12 slides p.01–p.60 逐页文字核读并建立 `cs224n/lec12 Efficient Adaptation Prompting PEFT.md`；覆盖 prompting、CoT、pruning、Lottery Ticket、LoRA、QLoRA、prefix/prompt tuning、adapters 与其他 PEFT。
- 已完成：Lecture 12 slides p.01–p.60 页面缩略图/图示核查；公式页、LoRA 表图、prompt tuning scaling、adapter 结构和重复 Overview 页与文字层一致。
- 已完成：Lecture 12 扩展阅读 Hu et al. (2021) LoRA，PDF 26 页；正文、实验、rank/矩阵分析和实质性附录已核读，参考文献列表仅登记边界。
- 已写入：`cs224n/reading - Hu 2021 LoRA.md`，补充低秩参数化、初始化/缩放、参数量、矩阵选择、rank 分析和合并推理边界。
- 已完成：Lecture 12 扩展阅读 Houlsby et al. (2019) Adapter，PDF 13 页（含补充材料）；正文、结构图、GLUE/额外任务实验和补充材料已核读，参考文献列表仅登记边界。
- 已写入：`cs224n/reading - Houlsby 2019 Parameter-Efficient Transfer Learning.md`，补充 bottleneck adapter、near-identity、参数/性能 trade-off、顺序任务与 LoRA 对照。
- 已完成：Lecture 12 扩展阅读 Wei et al. (2022) Chain-of-Thought，PDF 43 页；正文、FAQ、实验全表、扩展相关工作、错误分析和完整 prompt 附录已核读。
- 已写入：`cs224n/reading - Wei 2022 Chain-of-Thought Prompting.md`，补充 CoT 形式化、算术/常识/符号实验、规模涌现、prompt 鲁棒性、错误链与复现边界。
- 已完成：Lecture 12 扩展阅读 Frankle & Carbin (2019) Lottery Ticket，PDF 42 页；正文、主要图表、限制、相关工作和 Appendix B–I 实质性材料已核读。
- 已写入：`cs224n/reading - Frankle Carbin 2019 Lottery Ticket.md`，补充 mask/初始化形式化、迭代剪枝、随机重初始化、深层网络 warmup 与 PEFT 边界。
- 已完成：Lecture 12 扩展阅读 Brown et al. (2020) GPT-3 Few-Shot，PDF 75 页；正文、实验图表、benchmark contamination、limitations、broader impacts 与 Appendix A–H 实质性材料已核读。
- 已写入：`cs224n/reading - Brown 2020 GPT-3 Few-Shot Learners.md`，补充 zero/one/few-shot、模型/数据/评测、任务结果、污染分析、社会影响和与 CoT/PEFT 边界。
- 已验收：Lecture 12 slides p.01–p.60、5/5 直接扩展阅读、独立笔记、总索引和台账均已回读；下一项进入 Lecture 13。
- 已开始：Lecture 13 Benchmarking and Evaluation，确认 slides 物理页数 59；课件内部标题称 Lecture 12，已在独立笔记中登记编号差异。
- 已完成：Lecture 13 slides p.01–p.59 逐页文字核读和页面缩略图核查；覆盖 closed/open-ended benchmark、shortcut、diagnostic/adversarial、生成指标、人评、LM judge、arena、HELM、code/factuality/agents/opinion、consistency 与 contamination。
- 已完成：Lecture 13 扩展阅读 Hendrycks et al. (2021) MMLU，PDF 27 页；正文、实验图表、讨论和 Appendix B 任务样例已核读。
- 已写入：`cs224n/reading - Hendrycks 2021 MMLU.md`，补充 57 任务、数据划分、few-shot prompt、跨学科结果、校准、污染与 benchmark 边界。
- 已完成：Lecture 13 扩展阅读 Ruder《Challenges and Opportunities in NLP Benchmarking》网页快照，原文 8 个小节已核读。
- 已写入：`cs224n/reading - Ruder NLP Benchmarking.md`，补充 benchmark 定义、指标/下游约束、细粒度/长尾、统计显著性和动态版本化。
- 已完成：Lecture 13 扩展阅读 AlpacaEval 官方网页快照，About、添加模型/评测集和限制已核读。
- 已写入：`cs224n/reading - AlpacaEval.md`，补充 pairwise LM judge、reference、length-controlled win rate、judge bias 与安全缺口。
- 已完成：Lecture 13 扩展阅读 Liang et al. (2022/2023) HELM，PDF 162 页；摘要、正文、核心/targeted 场景、指标、实验发现、缺失项、限制、结论和实质性附录已核读。
- 已写入：`cs224n/reading - Liang 2022 HELM.md`，补充 scenario–metric taxonomy、7 类指标、标准化 prompting、30 模型/42 场景、污染与聚合边界。
- 已验收：Lecture 13 slides p.01–p.59、4/4 直接扩展阅读、独立笔记、总索引和台账均已回读；下一项进入 Lecture 14。
- 已开始：Lecture 14 Question Answering and Knowledge，确认 slides 物理页数 53；课件内部标题称 Lecture 13，已在独立笔记中登记编号差异。
- 已完成：Lecture 14 slides p.01–p.53 逐页文字核读和页面缩略图核查；覆盖 closed-book/parametric knowledge、reading comprehension、open-domain QA、retriever-reader、BM25/DPR、RAG、long-context、web search 与 citation correctness。
- 已完成：Lecture 14 扩展阅读 Rajpurkar et al. (2016) SQuAD，PDF 10 页；正文、数据收集/分析、方法、实验和图表已核读。
- 已写入：`cs224n/reading - Rajpurkar 2016 SQuAD.md`，补充 span QA、标注流程、answer/reasoning 类型、dependency divergence、EM/F1 和 baseline。
- 已完成：Lecture 14 扩展阅读 Seo et al. (2017) BiDAF，PDF 13 页；正文、attention 公式、SQuAD/CNN-DailyMail 实验、消融、可视化和错误分析已核读。
- 已写入：`cs224n/reading - Seo 2017 BiDAF.md`，补充 C2Q/Q2C、similarity matrix、start/end span loss、主要 ablation 和长上下文边界。
- 已完成：Lecture 14 扩展阅读 Karpukhin et al. (2020) DPR，PDF 13 页；正文、双编码器公式、in-batch negatives、BM25 hard negatives、检索/端到端指标、消融与效率讨论已核读。
- 已写入：`cs224n/reading - Karpukhin 2020 DPR.md`，补充 dense retrieval、FAISS/离线段落编码、top-k retrieval 与 reader EM 的区分，以及 SQuAD 中 BM25 可能占优的边界。
- 已完成：Lecture 14 扩展阅读 Chen et al. (2017) Reading Wikipedia to Answer Open-Domain Questions，PDF 10 页；正文、DrQA retriever/reader、span 公式、distant supervision、multitask、实验与误差分析已核读。
- 已写入：`cs224n/reading - Chen 2017 Reading Wikipedia Open Domain QA.md`，补充 TF-IDF/bigram hashing、SQuAD 与 open-domain 的评测差异、检索—阅读误差分解和与 DPR/RAG 的演进联系。
- 已完成：Lecture 14 扩展阅读 Guu et al. (2020) REALM，PDF 12 页；正文、潜变量检索公式、MIPS/异步刷新、梯度推导、salient masking、null/trivial retrieval、实验和知识更新附录已核读。
- 已写入：`cs224n/reading - Guu 2020 REALM.md`，补充 implicit-to-explicit knowledge、retriever/encoder 联合预训练、retrieval utility 及与 DrQA/DPR/RAG 的演进关系。
- 已完成：Lecture 14 扩展阅读 Liu et al. (2023) Lost in the Middle，PDF 18 页（含实质性附录）；多文档 QA、key-value retrieval、U 形位置曲线、模型/训练长度分析、RAG case study 和附录控制实验已核读。
- 已写入：`cs224n/reading - Liu 2023 Lost in the Middle.md`，补充位置敏感性、retriever recall 与最终 QA 脱钩、长上下文评测协议和 citation/evidence use 边界。
- 已验收：Lecture 14 slides p.01–p.53、6/6 直接扩展阅读、独立笔记、总索引和台账均已回读；下一项进入 Lecture 15。
- 已开始：Lecture 15 Model Analysis and Interpretability，确认 slides 物理页数 54；逐页登记已写入独立分讲笔记。
- 已完成：Lecture 15 slides p.01–p.54 文字层逐页核对，覆盖 stress testing、probing、feature attribution、causal intervention、解释性评估与 auto-interpretability case study。
- 已完成：Lecture 15 扩展阅读 Jia & Liang (2017) Adversarial Examples for Evaluating Reading Comprehension Systems，PDF 11 页；adversarial evaluation、ADDSENT/ADDONESENT/ADDANY、16 模型结果、误差分析、迁移性与 adversarial training 已核读。
- 已写入：`cs224n/reading - Jia Liang 2017 Adversarial Reading Comprehension.md`，补充 stress testing、兼容性约束、overstability 与测试集外推边界。
- 已完成：Lecture 15 扩展阅读 Tenney et al. (2019) BERT Rediscovers the Classical NLP Pipeline，PDF 9 页；edge probing、scalar mixing、cumulative scoring、BERT-base/large、逐例动态修正与附录已核读。
- 已写入：`cs224n/reading - Tenney 2019 BERT NLP Pipeline.md`，补充 probing 的“可解码不等于使用”、层级指标公式与 causal intervention 边界。
- 已完成：Lecture 15 扩展阅读 Sundararajan et al. (2017) Axiomatic Attribution for Deep Networks，PDF 9 页；Sensitivity、Implementation Invariance、Integrated Gradients、Completeness、path methods、baseline、数值近似、应用与附录已核读。
- 已写入：`cs224n/reading - Sundararajan 2017 Integrated Gradients.md`，补充 IG 公式、baseline/step-size 验收、应用案例和“函数归因不等于因果机制”边界。
- 已完成：Lecture 15 扩展阅读 Vig et al. (2020) Investigating Gender Bias in Language Models Using Causal Mediation Analysis，PDF 14 页；TE/NDE/NIE、neuron/head intervention、GPT-2 结果、稀疏性、数据集与伦理局限已核读。
- 已写入：`cs224n/reading - Vig 2020 Causal Mediation Gender Bias.md`，补充从 probing 到 causal intervention 的桥接、效应定义和“模型计算机制不等于社会因果”的边界。
- 已完成：Lecture 15 直接清单中的 Jing Huang 研究主页 HTML，页面正文与研究方向已核读；该项目按“网页背景入口”记录，不冒充论文全文。
- 已写入：`cs224n/reading - Jing Huang Interpretability Page.md`，补充 memorization、causal abstraction、generalization 与 interpretability evaluation 的课程联系。
- 已完成：Lecture 15 直接清单中的 Faithful, Interpretable Model Explanations via Causal Abstraction Stanford AI Lab Blog HTML，正文、动画 alt text、IIT、interchange intervention accuracy 和附录方法分类已核读。
- 已写入：`cs224n/reading - Geiger 2022 Causal Abstraction Web.md`，补充高层/低层因果模型、alignment、IIT 与机制验收边界。
- 已验收：Lecture 15 slides p.01–p.54、6/6 直接扩展阅读、独立笔记、总索引和台账均已回读；全课程材料缺口清点继续。
- 已更新：A1 Q1.1–Q2.8 已在兼容的 `cs224n` Conda 环境实际运行一次；运行参数、模型版本、图像和输出已回写 lec1/lec2，原先“待运行”的状态已改为已核验。Q2.9 仍未做去偏实验。
- 已确认：Stanford `slp3/11.pdf` 当前仍返回 25 页、标题为 *Information Retrieval and Retrieval-Augmented Generation* 的 2026-08-19 草稿；通过官方 `old_aug25/10.pdf` 取得并精读了清单目标的 20 页 Masked Language Models 章节，清单“Chapter 11”与官方版本号差异已写入笔记。Bengio–Simard–Frasconi (1994) 已通过 Internet Archive 的完整作者存档取得并核读 35 页；Karpathy Medium 返回 403、Springer 书页仍为目录/付费访问。
