# Mission: 理解 A3 神经机器翻译

## Why
已有 A3 实现，希望结合课程课件理解模型原理，能够解释现有代码为什么这样组织，而不只是运行代码。

## Success looks like
- 能解释 LSTM hidden state 与 cell state 的区别及更新过程。
- 能标注 encoder、decoder、attention 的张量各轴含义，并追踪形状变化。
- 能把课件公式对应到已有 A3 实现，解释训练与生成的不同。

## Constraints
- 用户尚未学习 LSTM hidden/cell state 与 PyTorch 张量维度操作。
- 中文教学，以指定课件为主，小步练习后根据回答调整进度。

## Out of scope
- 当前不改作业实现，不启动训练。
