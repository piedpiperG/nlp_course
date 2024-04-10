import torch
import numpy as np


class SvdDecPyTorch:

    def __init__(self, stopwords_path, train_path, test_path, window_size=5, k=5, device='cuda'):
        self.stopwords_path = stopwords_path
        self.train_path = train_path
        self.test_path = test_path
        self.window_size = window_size
        self.k = k
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        print(f"使用设备: {self.device}")

    def preprocess_file(self):
        """从文件中读取文本并预处理：去除停用词"""
        print("开始预处理文件...")
        with open(self.stopwords_path, 'r', encoding='utf-8') as f:
            stopwords = set(line.strip() for line in f)
        processed_sentences = []
        with open(self.train_path, 'r', encoding='utf-8') as f:
            for line in f:
                words = line.strip().split()
                filtered_words = [word for word in words if word not in stopwords and word.strip() != '']
                processed_sentences.append(" ".join(filtered_words))
        print("文件预处理完成。")
        return processed_sentences

    def build_cooccurrence_matrix(self, sentences):
        print("构建共现矩阵...")
        vocab = set(word for sentence in sentences for word in sentence.split())
        vocab_index = {word: i for i, word in enumerate(vocab)}
        cooccurrence_matrix = torch.zeros((len(vocab), len(vocab)), dtype=torch.float32, device=self.device)

        for sentence in sentences:
            words = sentence.split()
            for i, word in enumerate(words):
                if word in vocab_index:
                    target_word_index = vocab_index[word]
                    start = max(0, i - self.window_size)
                    end = min(len(words), i + self.window_size + 1)
                    for j in range(start, end):
                        if i != j and words[j] in vocab_index:
                            context_word_index = vocab_index[words[j]]
                            cooccurrence_matrix[target_word_index, context_word_index] += 1
        print("共现矩阵构建完成。")
        return cooccurrence_matrix, vocab_index

    def apply_svd(self, cooccurrence_matrix):
        print("应用SVD分解...")
        # 将数据转换为半精度浮点数 (float16) 以减少内存使用
        with torch.no_grad():
            cooccurrence_matrix_half = cooccurrence_matrix.to(dtype=torch.float16)
            u, s, v = torch.linalg.svd(cooccurrence_matrix_half, full_matrices=False)
            u_k = u[:, :self.k].to(self.device, dtype=torch.float32)  # 转换回全精度用于后续计算

        total_non_zero_singular_values = torch.sum(s > 0).item()
        selected_singular_values_sum = torch.sum(s[:self.k]).item()
        total_singular_values_sum = torch.sum(s).item()
        ratio = selected_singular_values_sum / total_singular_values_sum

        print(f"总共有 {total_non_zero_singular_values} 个非零奇异值")
        print(f"选取的奇异值之和: {selected_singular_values_sum}")
        print(f"全部奇异值之和: {total_singular_values_sum}")
        print(f"二者的比例: {ratio}")

        # 删除不再需要的大变量并清理内存
        del cooccurrence_matrix_half, u, s, v
        gc.collect()  # 手动触发垃圾收集
        if self.device == 'cuda':
            torch.cuda.empty_cache()  # 清空CUDA缓存

        print("SVD分解应用完成。")
        return u_k

    def calculate_similarity_pairs(self, reduced_matrix, vocab_index):
        print("计算词对相似度...")
        similarities = []
        with open(self.test_path, 'r', encoding='utf-8') as f:
            for line in f:
                word1, word2 = line.strip().split()
                if word1 in vocab_index and word2 in vocab_index:
                    vec1 = reduced_matrix[vocab_index[word1]]
                    vec2 = reduced_matrix[vocab_index[word2]]
                    similarity = self.cosine_similarity(vec1, vec2).item()
                else:
                    similarity = 0
                similarities.append((word1, word2, similarity))
        print("词对相似度计算完成。")
        return similarities


if __name__ == '__main__':
    stopwords_path = '../data/stopwords.txt'
    train_path = '../data/training.txt'
    test_path = '../data/pku_sim_test.txt'
    svd_dec = SvdDecPyTorch(stopwords_path, train_path, test_path, device='cuda')

    # 对training.txt文件进行预处理，包括分词、去除停用词（如果需要）等操作，以获得文本的基本单位（如词或子词）。
    processed_sentences = svd_dec.preprocess_file()
    # 构建共现矩阵：基于预处理后的语料，构建一个共现矩阵。在这个矩阵中，行和列分别代表语料库中的唯一词汇，矩阵中的每个元素代表对应行词和列词共同出现在一定窗口大小内的次数。
    coocurrence_matrix, vocab_index = svd_dec.build_cooccurrence_matrix(processed_sentences)
    # 应用SVD分解：对共现矩阵应用奇异值分解（SVD），以减少特征空间的维度。这里可以选择降维后的维数
    reduced_matrix = svd_dec.apply_svd(coocurrence_matrix)
    # 计算词向量相似度：使用SVD分解的结果获取每个词的向量表示。然后，基于这些向量，计算pku_sim_test.txt中每一行两个词的余弦相似度。如果某个词在training.txt中没有出现，将这对词的相似度设为0。
    similarities = svd_dec.calculate_similarity_pairs(reduced_matrix, vocab_index)

    similarities_svd = []
    # 打印结果
    for word1, word2, similarity in similarities:
        # print(f"{word1} - {word2}: {similarity}")
        similarities_svd.append((word1, word2, similarity))
    print(similarities_svd)
