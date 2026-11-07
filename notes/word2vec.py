import torch
import torch.nn as nn
from torch.nn import functional


# 目标: 迭代出词向量字典 embeddings
class CBOW(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super(CBOW, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.linear = nn.Linear(embedding_dim, vocab_size)

    def forward(self, context_idxs):
        # context_idxs: 是待预测中心词周围的上下文词的编号
        # 查询到这些词语的词向量
        embeds = self.embeddings(context_idxs)
        # 平均
        embeds = embeds.mean(dim=0, keepdim=True)
        out = self.linear(embeds)
        return out


class SkipGram(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super(SkipGram, self).__init__()
        # 将目标词转换为词向量
        self.in_embeddings = nn.Embedding(vocab_size, embedding_dim)
        # 表示所有上下文的词向量
        self.out_embeddings = nn.Embedding(vocab_size, embedding_dim)

    def forward(self, target):
        in_vec = self.in_embeddings(target)
        out_vec = self.out_embeddings.weight
        scores = torch.matmul(in_vec, out_vec.t())
        return functional.log_softmax(scores, dim=1)


# ---------- 1. 准备语料和词表 ----------
torch.manual_seed(43)

# 已分词的句子；构建上下文时不跨句子边界
sentences = [
    ["我", "喜欢", "喝", "咖啡"],
    ["我", "喜欢", "喝", "牛奶"],
    ["你", "每天", "喝", "咖啡"],
]

vocab = sorted({word for sentence in sentences for word in sentence})
word_to_idx = {word: idx for idx, word in enumerate(vocab)}

# ---------- 2. 构建训练数据: 上下文 -> 中心词 ----------
window_size = 2
data = []

for sentence in sentences:
    ids = [word_to_idx[word] for word in sentence]
    for i, center_idx in enumerate(ids):
        context = [ids[j] for j in range(max(0, i - window_size), min(len(ids), i + window_size + 1), ) if j != i]
        data.append((torch.tensor(context, dtype=torch.long), torch.tensor([center_idx], dtype=torch.long),))

# ---------- 3. 训练 ----------
model = CBOW(len(vocab), embedding_dim=8)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
loss_fn = nn.CrossEntropyLoss()

for epoch in range(1000):
    total_loss = 0.0

    for context, target in data:
        optimizer.zero_grad()
        scores = model(context)
        loss = loss_fn(scores, target)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    if (epoch + 1) % 20 == 0:
        print(f"第 {epoch + 1} 轮，平均损失：{total_loss / len(data):.4f}")

# ---------- 4. 提取词向量字典 ----------
embeddings = {
    word: model.embeddings.weight[idx].detach().clone()
    for word, idx in word_to_idx.items()
}

print("“咖啡”的词向量：", embeddings["咖啡"])
