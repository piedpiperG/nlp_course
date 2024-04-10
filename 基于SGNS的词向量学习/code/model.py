"""
定义模型
在PyTorch中定义一个简单的模型，包含词嵌入层和负采样逻辑。

定义嵌入层：使用torch.nn.Embedding为中心词和上下文词创建两个嵌入层。
负采样：利用PyTorch的torch.nn.functional.nll_loss（负对数似然损失）实现负采样。这需要正确地选择负样本。
"""
import torch.nn as nn
import torch.nn.functional as F
import torch


class SGNSModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super(SGNSModel, self).__init__()
        # 中心词嵌入层
        self.center_embeddings = nn.Embedding(vocab_size, embedding_dim)
        # 上下文词嵌入层
        self.context_embeddings = nn.Embedding(vocab_size, embedding_dim)
        # 词嵌入维度
        self.embedding_dim = embedding_dim

        # 初始化权重
        self.init_weights()

    def forward(self, center_word_indices, context_word_indices, negative_word_indices):
        # 查找中心词和上下文词的嵌入
        center_embeds = self.center_embeddings(center_word_indices)
        context_embeds = self.context_embeddings(context_word_indices)

        # 计算正样本对的得分
        positive_scores = torch.sum(center_embeds * context_embeds, dim=1)
        # 使用logsigmoid处理正样本得分
        log_target = F.logsigmoid(positive_scores)

        # 查找负样本的嵌入并计算得分
        negative_embeds = self.context_embeddings(negative_word_indices)
        # 负样本得分的计算稍有不同，需要额外的维度操作
        negative_scores = torch.bmm(negative_embeds, center_embeds.unsqueeze(2)).squeeze(2)
        # 使用logsigmoid处理负样本得分
        sum_log_sampled = F.logsigmoid(-1 * negative_scores)

        # 损失是正样本和负样本得分的logsigmoid之和的负数
        loss = -1 * (log_target.sum() + sum_log_sampled.sum()) / (len(positive_scores) + len(negative_scores))

        return loss

    def init_weights(self):
        # Xavier初始化的一种常见选择是使用均匀分布
        initrange = 0.5 / self.embedding_dim
        self.center_embeddings.weight.data.uniform_(-initrange, initrange)
        self.context_embeddings.weight.data.uniform_(-initrange, initrange)
