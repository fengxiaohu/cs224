# A3 Written Questions — Explanation and Answer Draft

这份草稿先用中文拆解题意和计算，再给出可整理进作业的英文答案。Q2(d) 需要引用你自己的 beam-search 诊断记录；当前目录没有这些 JSON，不能替你编造模型输出，所以该小节留有待填模板。

## 中文讲解

### Q1(g)：padding mask 怎样影响 attention

一个 batch 里的源句长度可能不同。为了组成矩形张量，较短的句子会在末尾补上 `<pad>`，但这些位置不是输入句子的一部分。代码里的 `enc_masks` 用 1 标记 padding、用 0 标记真实 token；decoder 在算完每个源位置的 attention score `e_t` 后，把 mask 为 1 的分数设成 `-inf`，然后才做 softmax。

softmax 对 `-inf` 的指数权重为 0，所以 padding 位置的 attention probability 是 0。随后 attention context 是 encoder hidden states 的加权和，padding 位置因此不会进入 context，也不会影响后续 combined output。必须 mask 是因为 padding 只是 batch 对齐用的占位符，不含源句语义；不 mask 时模型可能把注意力分给这些无意义位置，污染 decoder 的上下文。

### Q1(i)：三种 attention 的比较

- **Dot-product**：`e = sᵀh`。优点是不用额外学习投影矩阵，参数较少、计算通常更快；缺点是 encoder 和 decoder 表示通常要维度相同，匹配方式也受限，表达能力比带可学习投影的 multiplicative 弱。
- **Multiplicative**：`e = sᵀWh`。这是当前实现；`W` 可以把 encoder 表示投影到 decoder 表示的维度，因此能学习如何匹配两侧特征。相对 dot-product，它要多算一个投影并学习额外参数。
- **Additive**：`e = vᵀ tanh(W₁h + W₂s)`。独立投影加 `tanh` 能建模非线性匹配，也能自然处理两侧原始维度不同；代价是更多参数和投影/非线性运算，通常计算开销更高。

题目只要求说明 dot-product、additive 各自相对 multiplicative 的一个优点和一个缺点，回答时不必把所有实现细节都写上。

### Q2(a)：embedding 后的一维卷积有什么用

SentencePiece 的一个 token 可能是单个汉字，也可能由多个汉字构成；相邻汉字的组合也会产生新的词义。题目举的例子中，`电` 是“电”，`脑` 是“脑”，但 `电脑` 的整体意思是“计算机”。embedding 后的一维卷积沿 token 序列滑动，能在进入 BiLSTM 前学习相邻 token 的局部组合特征，帮助模型表示词素、复合词和短语；多个卷积核或窗口宽度可以捕获不同局部模式。通常应保留序列长度，使 encoder 的位置和 attention 对齐不被卷积改变。

### Q2(b)：四个标注错误

作业强调只分析下划线处，因此不要把未标注的其他措辞差异混进诊断。

1. **`the culprits were` / `the culprit was`**：标注处把复数名词和复数系动词变成了单数，造成 number/agreement 错误。中文复数标记或其对应的源端线索可能没有被模型充分利用，也可能是英语形态变化较稀疏、单数表达更常见。一种修复方向是加入更多带清晰复数一致关系的平行训练例句，或使用显式的 agreement/number 约束。
2. **重复 `resources have been exhausted`**：模型重复了前面已经生成的分句，同时漏译“几乎没有地方容纳这些人”的内容。这常见于模型没有跟踪源句覆盖情况，decoder 再次关注已翻译区域，或重复沿用一个高概率短语。可以加入 coverage 向量/coverage penalty，降低重复关注同一源端位置的倾向。
3. **`a national mourning today` / `today's day`**：`today's day` 不自然，也丢失了“国殇日”作为“national mourning day”这一整体概念的含义。多字固定表达可能在数据里较少，模型把它拆成普通词逐个组合。一种修复办法是补充相关领域的平行语料，或在解码中加入术语/短语约束。
4. **`act not, err not` / `it's not wrong`**：模型只保留了“没有错”的含义，漏掉“不做就不会犯错”的前半部分及其平行结构；这句还是粤语俗语，训练语料若以普通话新闻为主，可能存在方言和固定表达的数据分布差异。可以用粤英谚语/粤语平行例句做领域微调，或把整条俗语作为固定短语建模。

### Q2(c)：BLEU 手算

权重是 `λ₁=λ₂=0.5`，忽略 3-gram 和 4-gram，因此：

`BLEU = BP × exp(0.5 ln p₁ + 0.5 ln p₂) = BP × sqrt(p₁p₂)`.

两个 reference 的长度分别为：`r₁=11`，`r₂=6`。候选长度：`c₁=9`，`c₂=6`。题面把 `sufficient` 中的 `ff` 排成了 Unicode ligature `ﬀ`；以下按其意图将它规范化为普通单词 `sufficient` 再匹配。

**两个 references 都使用：**

- `c₁ = there is a need for adequate and predictable resources`：9 个 unigram 中有 4 个可匹配（`adequate`, `and`, `predictable`, `resources`），所以 `p₁=4/9`。8 个 bigram 中有 3 个可匹配（`adequate and`, `and predictable`, `predictable resources`），所以 `p₂=3/8`。最近的 reference 长度是 11，故 `BP=exp(1−11/9)=0.801`，`BLEU=0.801×sqrt((4/9)(3/8))=0.327`。
- `c₂ = resources be sufficient and predictable to`：6 个 unigram 都能在至少一个 reference 中匹配，所以 `p₁=6/6=1`。5 个 bigram 中有 3 个匹配（`be sufficient`, `sufficient and`, `and predictable`），所以 `p₂=3/5=0.600`。最近的 reference 长度是 6，故 `BP=1`，`BLEU=sqrt(1×0.600)=0.775`。

按 BLEU，`c₂` 更好（`0.775 > 0.327`），但人工更可能认为完整、自然的 `c₁` 更好；`c₂` 的语法不完整，末尾的 `to` 悬空。两个 references 的最大计数规则让 `c₂` 能从不同 reference 获得多个表面 n-gram 匹配。

**只使用 `r₂`：**

- `c₁`：`p₁=4/9`、`p₂=3/8`、`len(c)=9`、`len(r)=6`、`BP=1`，所以 `BLEU=sqrt((4/9)(3/8))=0.408`。
- `c₂`：只有 `resources`, `and`, `predictable` 三个 unigram 匹配，`p₁=3/6=0.500`；只有 `and predictable` 这个 bigram 匹配，`p₂=1/5=0.200`。`len(c)=len(r)=6`，`BP=1`，所以 `BLEU=sqrt(0.500×0.200)=0.316`。

此时 `c₁` 得分更高（`0.408 > 0.316`），与人工判断更一致。单一 reference 会把没有出现在那条参考表达里的正确改写计成不匹配；增加 references 能覆盖更多措辞并改变长度惩罚所选的 reference，但 BLEU 仍主要看表面 n-gram，不能直接判断意思是否完整或语法是否自然。

相对人工评价，BLEU 的优点是便宜快速、可在大语料上重复计算，也提供方便比较实验的统一数值。缺点是依赖参考措辞、分词和 n-gram 表面重合，容易惩罚正确的同义改写；它也无法可靠评估语义、流畅度、事实准确性，且单个分数不说明错在哪里。

### Q2(d)：Beam search 需要真实输出

Beam search 每一步保留累计模型分数最高的前 `K` 个部分译文（本题 `K=10`），再把它们分别扩展到下一步；这比每步只选一个最高概率 token 的 greedy decoding 多保留一些候选路径。题目要求比较 iteration 200、3000 和最后一轮的第一个 hypothesis，并展示最后一轮的另外三个 hypothesis，因此必须读取自己训练生成的诊断 JSON。当前目录没有 `student/outputs/beam_search_diagnostics/*.json`，现有 metrics 记录到 `train_iter=400`，所以无法从当前文件核实所需的 iteration 3000/last 输出，也不能可靠判断实际翻译质量是否随训练改善。

题目中记录的示例源句与参考译文可用于确认是不是同一个例子：

- Source: `我还澄清了在该会议上提出的若干事项。`
- Reference: `i was able to provide clarification on some of the matters which were raised at that meeting .`

拿到诊断 JSON 后，抄录 `hypotheses[0].hypothesis` 在 200、3000 和最后一轮的文本，并对比是否逐渐更完整、更接近 reference；再抄最后一轮 `hypotheses[1]` 到 `[3]`，比较 beam 候选在词序、用词、长度或完整性上的差别。不能只凭训练轮次假设质量单调上升，必须以这些实际 hypothesis 为依据。

## English answer draft

### Q1(g)

`generate_sent_masks()` marks padding positions with 1 and real source positions with 0. Before softmax, `step()` replaces every attention score at a masked position with `−∞`. Softmax therefore assigns zero attention probability to padding, so those positions contribute nothing to the attention-weighted context or the combined decoder output.

Padding is only a batching placeholder and carries no source meaning; masking prevents the decoder from assigning attention to meaningless encoder positions and corrupting its context.

### Q1(i)

Dot-product attention is simpler and usually faster than multiplicative attention because it requires no learned projection matrix. Its disadvantage is that the encoder and decoder representations generally need compatible dimensions, and the fixed dot product is less flexible than a learned bilinear score.

Additive attention can model a nonlinear match with separate projections, so it can compare representations with different dimensions. Its disadvantage is that the projections and `tanh` usually require more parameters and computation than multiplicative attention.

### Q2(a)

A 1D convolution over the embeddings can learn local combinations of neighboring Chinese pieces before the BiLSTM processes the sequence. For example, it can combine the features for `电` (“electricity”) and `脑` (“brain”) into a feature for `电脑` (“computer”), and different filters can capture other local morpheme or word patterns. This gives the encoder useful local lexical features and reduces the burden on the recurrent layers to discover every such pattern themselves.

### Q2(b)

1. **Number agreement:** The underlined phrase changes plural `the culprits were` into singular `the culprit was`. The model may have missed the source-side plurality cue or favored the more frequent singular form. One possible fix is to train with more examples that make English number agreement explicit, or to add a number-agreement constraint.
2. **Repeated phrase:** The model repeats `resources have been exhausted` instead of translating the distinct source content about there being almost no space for the people. The decoder may be attending again to a source region it has already translated. A coverage mechanism or coverage penalty could discourage repeated attention and output.
3. **Mistranslated expression:** `today's day` is unnatural and fails to convey `a national mourning today` as a national mourning day. The model may have treated a rare multi-character expression compositionally rather than recognizing it as a phrase. Adding domain examples or a terminology/phrase constraint could help.
4. **Lost proverb structure:** `it's not wrong` omits the “act not” part and loses the proverb's parallel conditional meaning. The model may not have learned this Cantonese expression from predominantly Mandarin or news-domain data. Fine-tuning on Cantonese-English proverb examples or modeling the proverb as a phrase could help.

### Q2(c)

I treat the printed `suﬀicient` as the intended word `sufficient`, with the `ﬀ` ligature normalized. With the two references, `len(r₁)=11` and `len(r₂)=6`.

| Candidate | `p₁` | `p₂` | `len(c)` | selected `len(r)` | BP | BLEU |
|---|---:|---:|---:|---:|---:|---:|
| `c₁` | `4/9 = 0.444` | `3/8 = 0.375` | 9 | 11 | `exp(1−11/9)=0.801` | `0.327` |
| `c₂` | `6/6 = 1.000` | `3/5 = 0.600` | 6 | 6 | 1 | `0.775` |

`BLEU(c₁)=0.801×sqrt((4/9)(3/8))=0.327`, while `BLEU(c₂)=sqrt(1×0.600)=0.775`. BLEU therefore ranks `c₂` higher, but I do not agree that it is the better translation: `c₂` is incomplete and ends with a stranded `to`, while `c₁` is grammatical and conveys the meaning naturally.

Using only `r₂`, `c₁` has `p₁=4/9`, `p₂=3/8`, `len(c)=9`, `len(r)=6`, and `BP=1`, giving `BLEU(c₁)=sqrt((4/9)(3/8))=0.408`. `c₂` has `p₁=3/6=0.500`, `p₂=1/5=0.200`, `len(c)=len(r)=6`, and `BP=1`, giving `BLEU(c₂)=sqrt(0.500×0.200)=0.316`. With one reference, `c₁` receives the higher score, which agrees better with human judgment.

Using only one reference is problematic because a valid paraphrase may have low n-gram overlap with that particular wording and be penalized, even when its meaning is correct. With multiple references, a candidate n-gram can match the maximum count found in any reference, and the closest reference length is used for the brevity penalty; one reference provides fewer valid wording and length alternatives. Multiple references reduce but do not remove BLEU's dependence on surface n-gram overlap.

Compared with human evaluation, BLEU is inexpensive, fast, reproducible, and easy to apply to large test sets. However, it can penalize valid paraphrases because it relies on reference n-gram overlap, and it does not reliably judge meaning, grammaticality, fluency, or factual correctness; its score also does not explain the source of an error.

### Q2(d) — fill in from the recorded diagnostics

The required beam-search JSON files are not present in the current workspace, so this section cannot be completed accurately from the available evidence. Fill in the bracketed text from your own TensorBoard text records or `outputs/beam_search_diagnostics/` JSON files:

> At iteration 200, the first hypothesis is: **[paste hypotheses[0].hypothesis]**. At iteration 3000, it is: **[paste hypotheses[0].hypothesis]**. At the last recorded iteration **[iteration]**, it is: **[paste hypotheses[0].hypothesis]**. Comparing these outputs, translation quality **[improved / did not improve consistently]** because **[describe concrete changes in completeness, meaning, and fluency relative to the reference]**.
>
> Three other hypotheses at the last iteration are: **[hypotheses[1].hypothesis]**, **[hypotheses[2].hypothesis]**, and **[hypotheses[3].hypothesis]**. They differ in **[specific lexical, word-order, or length differences]**; **[say whether those differences preserve the meaning and which hypothesis reads more naturally]**.

