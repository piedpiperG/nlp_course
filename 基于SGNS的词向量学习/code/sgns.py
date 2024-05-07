import torch
from matplotlib import pyplot as plt
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import pickle
import time
import os

from model import SGNSModel
from data_pre import Data_prepare


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


# 定义训练过程
def train_process(model, data_loader, optimizer, device, epochs=1):
    model.train()
    loss_values = []  # 用于存储每个epoch的平均损失
    for epoch in range(epochs):
        batch_loss_values = []  # 用于存储每200个batch的损失
        start_time = time.time()  # 记录每个epoch开始的时间
        total_loss = 0
        for batch_idx, (center, context, negatives) in enumerate(data_loader):
            center = center.to(device)
            context = context.to(device)
            negatives = negatives.to(device)

            optimizer.zero_grad()
            loss = model(center, context, negatives)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            # 特定批次打印时间和损失信息
            if (batch_idx + 1) % 1 == 0:  # 假设每200个batch打印一次
                elapsed = time.time() - start_time
                print(
                    f"Epoch {epoch + 1}, Batch {batch_idx + 1}, Current Loss: {loss.item()}, Time elapsed: {elapsed:.2f} seconds")
                batch_loss_values.append(loss.item())  # 将当前损失加入列表
            if (batch_idx + 1) % 100 == 0:  # 假设每500个batch打印一次
                # 保存模型和词向量
                # 保存模型参数
                torch.save(model.state_dict(), 'sgns_model.pth')
                # 保存词向量
                word_vectors = model.center_embeddings.weight.data
                torch.save(word_vectors, 'word_vectors.pth')

                # test(test_path, embedding_dim)

        # 绘制每个epoch的损失变化
        plt.figure(figsize=(10, 5))
        plt.plot(batch_loss_values, label='Batch Loss')
        plt.xlabel('Batch (every 200th)')
        plt.ylabel('Loss')
        plt.title('Loss During Training (per 200 batches)')
        plt.legend()
        plt.grid(True)
        plt.savefig(f'../result/batch_loss_epoch{epoch + 1}.png')  # 保存损失图像到本地
        plt.show()  # 显示图像

        average_loss = total_loss / len(data_loader)
        print(f"Epoch {epoch + 1}, Loss: {average_loss}")
        loss_values.append(average_loss)

    # 保存模型和词向量
    # 保存模型参数
    torch.save(model.state_dict(), 'sgns_model.pth')
    # 保存词向量
    word_vectors = model.center_embeddings.weight.data
    torch.save(word_vectors, 'word_vectors.pth')

    test(test_path, embedding_dim)

    # 绘制损失图
    plt.figure(figsize=(10, 5))
    plt.plot(loss_values, label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss During Training')
    plt.legend()
    plt.grid(True)
    plt.savefig('training_loss.png')  # 保存损失图像到本地
    plt.show()  # 显示图像


# 计算余弦相似度
def cosine_similarity(vec1, vec2):
    dot_product = torch.dot(vec1, vec2)
    norm_a = torch.norm(vec1)
    norm_b = torch.norm(vec2)
    return dot_product / (norm_a * norm_b)


# 进行预测
def predict_similarity(test_path, vocab, word_vectors):
    similarities_sgns = []
    total_similarity = 0  # 用于累计所有的相似度值
    valid_pairs = 0  # 有效的词对数量

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
                total_similarity += similarity  # 累加相似度值
                valid_pairs += 1  # 有效词对数量加一
                # print(f"{word1} 和 {word2} 的余弦相似度是：{similarity}")
            else:
                similarities_sgns.append((word1, word2, 0))
                # print(f"{word1} 和 {word2} 的余弦相似度是：0（至少一个词不在词汇表中）")
    if valid_pairs > 0:
        average_similarity = total_similarity / valid_pairs  # 计算平均相似度
        print(f"平均余弦相似度是：{average_similarity}")
        return similarities_sgns, average_similarity

    else:
        print("没有有效的词对来计算平均余弦相似度。")
        return similarities_sgns, -1


def test(test_path, embedding_dim):
    # 加载词向量
    word_vectors = torch.load('word_vectors.pth')
    # 加载词汇表
    with open('vocab.pkl', 'rb') as f:
        vocab = pickle.load(f)
    len_vocab = len(vocab)

    # 实例化模型
    vocab_size = len_vocab  # 假定的词汇表大小
    model = SGNSModel(vocab_size, embedding_dim)
    model.load_state_dict(torch.load('sgns_model.pth'))

    # 进行预测
    similarities_sgns, average_similarity = predict_similarity(test_path, vocab, word_vectors)

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


# 加载预处理数据
def load_preprocessed_data(filepath):
    with open(filepath, 'rb') as f:
        data = pickle.load(f)
    print(f"Preprocessed data loaded from {filepath}")
    return data['positive_pairs'], data['negative_samples'], data['vocab']


def train(stopwords_path, train_path, window_size, embedding_dim, num_negative_samples):
    # 首先检查预处理文件是否存在
    filepath = 'preprocessed_data.pkl'
    if os.path.exists(filepath):
        positive_pairs, negative_samples, vocab = load_preprocessed_data(filepath)
        len_vocab = len(vocab)
        # 输出样本数量进行检查
        print(f"词汇储量为：{len(vocab)}")
        print(f"总共有{len(positive_pairs)}个正样本对")
        print(f"每个正样本对对应{len(negative_samples[0])}个负样本")
    # 如果不存在，预处理获取训练数据
    else:
        data_prepare = Data_prepare(stopwords_path, train_path, window_size, num_negative_samples)
        positive_pairs, negative_samples, len_vocab, vocab = data_prepare.pre_data()

    dataset = Word2VecDataset(positive_pairs, negative_samples)
    data_loader = DataLoader(dataset, batch_size=5096, shuffle=True)

    # 保存词汇表
    with open('vocab.pkl', 'wb') as f:
        pickle.dump(vocab, f)

    # 实例化模型
    vocab_size = len_vocab  # 假定的词汇表大小
    model = SGNSModel(vocab_size, embedding_dim)

    # 开始训练
    # 定义设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    # 定义优化器，并应用L2正则化
    optimizer = optim.Adam(model.parameters(), lr=0.09, weight_decay=1e-5)

    # 训练
    train_process(model, data_loader, optimizer, device)


if __name__ == '__main__':
    stopwords_path = '../data/stopwords.txt'
    train_path = '../data/training.txt'
    test_path = '../data/pku_sim_test.txt'
    window_size = 2
    embedding_dim = 200
    num_negative_samples = 20

    train(stopwords_path, train_path, window_size, embedding_dim, num_negative_samples)
    # test(test_path, embedding_dim)
