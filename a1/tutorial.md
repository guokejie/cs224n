# Lecture 01 Introduction and Word Vectors

---
## Word2vec introduction
我们为每个单词构建一个**密集**的向量，使其与出现在相似上下文中的单词向量相似

词向量 **word vectors** 有时被称为词嵌入 **word embeddings** 或词表示 **word representations**

它们是分布式表示 **distributed representation**

## Word2vec objective function
对于每个位置 $t = 1, \ldots, T$，在大小为 $m$ 的固定窗口内预测上下文单词，给定中心词 $w_t$。

$$
\text{Likelihood} = L(\theta)
= \prod_{t=1}^{T}
\prod_{\substack{-m \leq j \leq m \\ j \neq 0}}
P(w_{t+j} \mid w_t; \theta)
$$