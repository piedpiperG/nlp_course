import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import pickle

"""
定义模型
在PyTorch中定义一个简单的模型，包含词嵌入层和负采样逻辑。

定义嵌入层：使用torch.nn.Embedding为中心词和上下文词创建两个嵌入层。
负采样：利用PyTorch的torch.nn.functional.nll_loss（负对数似然损失）实现负采样。这需要正确地选择负样本。
"""


class SGNSModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super(SGNSModel, self).__init__()
        # 中心词嵌入层
        self.center_embeddings = nn.Embedding(vocab_size, embedding_dim)
        # 上下文词嵌入层
        self.context_embeddings = nn.Embedding(vocab_size, embedding_dim)

        # 非线性映射层
        self.fc1 = nn.Linear(embedding_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, embedding_dim)

        # 激活函数
        self.relu = nn.ReLU()

        # Dropout层
        self.dropout = nn.Dropout(0.5)

    def forward(self, center_word_indices, context_word_indices, negative_word_indices):
        # 查找中心词和上下文词的嵌入
        center_embeds = self.center_embeddings(center_word_indices)
        context_embeds = self.context_embeddings(context_word_indices)

        # 应用非线性映射
        center_embeds = self.dropout(self.relu(self.fc1(center_embeds)))
        center_embeds = self.fc2(center_embeds)

        # 计算正样本对的得分
        positive_scores = torch.sum(center_embeds * context_embeds, dim=1)

        # 查找负样本的嵌入并计算得分
        negative_embeds = self.context_embeddings(negative_word_indices)
        negative_scores = torch.bmm(negative_embeds, center_embeds.unsqueeze(2)).squeeze(2)

        # 计算损失
        positive_loss = F.binary_cross_entropy_with_logits(positive_scores, torch.ones_like(positive_scores))
        negative_loss = F.binary_cross_entropy_with_logits(negative_scores, torch.zeros_like(negative_scores))

        # 最终损失是正负样本损失的和
        loss = positive_loss + negative_loss.mean()  # 对负样本损失取平均

        return loss


class Word2VecDataset(Dataset):
    def __init__(self, positive_pairs, negative_samples):
        self.positive_pairs = positive_pairs
        self.negative_samples = negative_samples

    def __len__(self):
        return len(self.positive_pairs)

    def __getitem__(self, idx):
        center, context = self.positive_pairs[idx]
        negatives = self.negative_samples[idx]
        # 将列表转换为张量
        return torch.tensor(center, dtype=torch.long), torch.tensor(context, dtype=torch.long), torch.tensor(negatives,
                                                                                                             dtype=torch.long)


class Data_prepare:

    def __init__(self, stopwords_path, train_path, window_size):
        self.stopwords_path = stopwords_path
        self.train_path = train_path
        self.window_size = window_size
        self.processed_sentences = None
        self.vocab = None

    # 分词
    def preprocess_file(self):
        """从文件中读取文本并预处理：去除停用词"""
        with open(self.stopwords_path, 'r', encoding='utf-8') as f:
            stopwords = set([line.strip() for line in f])
        processed_sentences = []
        with open(self.train_path, 'r', encoding='utf-8') as f:
            for line in f:
                words = line.strip().split()
                filtered_words = [word for word in words if word not in stopwords and word.strip() != '']
                processed_sentences.append(" ".join(filtered_words))
        self.processed_sentences = processed_sentences
        return processed_sentences

    # 构建词汇表
    def build_vocab(self):
        vocab = {}
        for sentence in self.processed_sentences:
            for word in sentence.split():
                if word not in vocab:
                    vocab[word] = len(vocab)
        return vocab

    def generate_training_data(self, num_negative_samples=5):
        positive_pairs = []
        negative_samples = []

        # 生成所有可能的正样本对
        for sentence in self.processed_sentences:
            words = sentence.split()
            for i, center_word in enumerate(words):
                center_word_index = self.vocab[center_word]
                for j in range(max(0, i - self.window_size), min(i + self.window_size + 1, len(words))):
                    if i != j:
                        context_word = words[j]
                        context_word_index = self.vocab[context_word]
                        positive_pairs.append((center_word_index, context_word_index))

        # 生成负样本
        vocab_indices = list(self.vocab.values())
        for _ in range(len(positive_pairs)):
            negatives = []
            while len(negatives) < num_negative_samples:
                negative = random.choice(vocab_indices)
                if negative not in negatives:  # 确保负样本是唯一的
                    negatives.append(negative)
            negative_samples.append(negatives)

        return positive_pairs, negative_samples

    def pre_data(self):
        self.processed_sentences()
        self.build_vocab()
        positive_pairs, negative_samples = self.generate_training_data()

        # 输出样本数量进行检查
        print(f"词汇储量为：{len(self.vocab)}")
        print(f"总共有{len(positive_pairs)}个正样本对")
        print(f"每个正样本对对应{len(negative_samples[0])}个负样本")

        return positive_pairs, negative_samples, len(self.vocab), self.vocab


# 定义训练过程
def train(model, data_loader, optimizer, epochs=5):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for center, context, negatives in data_loader:
            center = center.to(device)
            context = context.to(device)
            negatives = negatives.to(device)

            optimizer.zero_grad()
            loss = model(center, context, negatives)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch + 1}, Loss: {total_loss / len(data_loader)}")


# 计算余弦相似度
def cosine_similarity(vec1, vec2):
    dot_product = torch.dot(vec1, vec2)
    norm_a = torch.norm(vec1)
    norm_b = torch.norm(vec2)
    return dot_product / (norm_a * norm_b)


# 进行预测
def predict_similarity(test_path, vocab, word_vectors):
    similarities_sgns = []
    with open(test_path, 'r', encoding='utf-8') as f:
        for line in f:
            word1, word2 = line.strip().split()
            if word1 in vocab and word2 in vocab:
                index1 = vocab[word1]
                index2 = vocab[word2]
                vec1 = word_vectors[index1]
                vec2 = word_vectors[index2]
                similarity = cosine_similarity(vec1, vec2).item()
                similarities_sgns.append((word1, word2, similarity))
                print(f"{word1} 和 {word2} 的余弦相似度是：{similarity}")
            else:
                similarities_sgns.append((word1, word2, 0))
                print(f"{word1} 和 {word2} 的余弦相似度是：0（至少一个词不在词汇表中）")
    return similarities_sgns


def test(test_path):
    # 加载词向量
    word_vectors = torch.load('word_vectors.pth')
    # 加载词汇表
    with open('vocab.pkl', 'rb') as f:
        vocab = pickle.load(f)
    len_vocab = len(vocab)

    # 实例化模型
    embedding_dim = 200  # 嵌入向量的维度
    hidden_dim = 400
    vocab_size = len_vocab  # 假定的词汇表大小
    model = SGNSModel(vocab_size, embedding_dim, hidden_dim)
    model.load_state_dict(torch.load('sgns_model.pth'))

    # 进行预测
    similarities_sgns = predict_similarity(test_path, vocab, word_vectors)

    # 读取原始文件
    with open('../result/similarties.txt', 'r', encoding='utf-8') as file:
        lines = file.readlines()
    # 修改每一行，追加SGNS相似度数据
    modified_lines = []
    for line, (_, _, sim_sgns) in zip(lines, similarities_sgns):
        modified_line = line.strip() + f"\t{sim_sgns}\n"  # 追加SGNS数据
        modified_lines.append(modified_line)

    # 将修改后的内容写回到新文件或覆盖原文件
    with open('../result/similarties.txt', 'w', encoding='utf-8') as file:
        file.writelines(modified_lines)


if __name__ == '__main__':
    stopwords_path = '../data/stopwords.txt'
    train_path = '../data/training.txt'
    test_path = '../data/pku_sim_test.txt'
    window_size = 5

    test(test_path)

    # # 预处理获取训练数据
    # data_prepare = Data_prepare(stopwords_path, train_path, window_size)
    # positive_pairs, negative_samples, len_vocab, vocab = data_prepare.pre_data()
    # dataset = Word2VecDataset(positive_pairs, negative_samples)
    # data_loader = DataLoader(dataset, batch_size=1024000, shuffle=True)
    #
    # # 实例化模型
    # embedding_dim = 200  # 嵌入向量的维度
    # hidden_dim = 400
    # vocab_size = len_vocab  # 假定的词汇表大小
    # model = SGNSModel(vocab_size, embedding_dim, hidden_dim)
    #
    # # 开始训练
    # # 定义设备
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # model.to(device)
    # # 定义优化器
    # optimizer = optim.Adam(model.parameters(), lr=0.003)
    #
    # # 训练
    # train(model, data_loader, optimizer)
    #
    # # 保存模型和词向量
    # # 保存模型参数
    # torch.save(model.state_dict(), 'sgns_model.pth')
    # # 保存词向量
    # word_vectors = model.center_embeddings.weight.data
    # torch.save(word_vectors, 'word_vectors.pth')
    # # 保存词汇表
    # with open('vocab.pkl', 'wb') as f:
    #     pickle.dump(vocab, f)
