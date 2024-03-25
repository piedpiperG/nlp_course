import numpy as np
from scipy.linalg import svd


class Svd_dec:

    def __init__(self, stopwords_path, train_path, test_path, window_size=2, k=5):
        self.stopwords_path = stopwords_path
        self.train_path = train_path
        self.test_path = test_path
        self.window_size = window_size
        self.k = k

    def preprocess_file(self):
        """从文件中读取文本并预处理：去除停用词"""
        with open(self.stopwords_path, 'r', encoding='utf-8') as f:
            stopwords = set([line.strip() for line in f])
        processed_sentences = []
        with open(self.train_path, 'r', encoding='gbk') as f:
            for line in f:
                words = line.strip().split()
                filtered_words = [word for word in words if word not in stopwords and word.strip() != '']
                processed_sentences.append(" ".join(filtered_words))
        return processed_sentences

    def build_cooccurrence_matrix(self, sentences):
        """构建共现矩阵"""
        # 统计所有唯一词汇并创建词汇索引
        vocab = set(word for sentence in sentences for word in sentence.split())
        vocab_index = {word: i for i, word in enumerate(vocab)}

        # 初始化共现矩阵
        cooccurrence_matrix = np.zeros((len(vocab), len(vocab)), dtype=np.int32)

        for sentence in sentences:
            words = sentence.split()
            for i, word in enumerate(words):
                target_word_index = vocab_index[word]
                # 为每个词汇定义窗口内的词汇
                start = max(0, i - self.window_size)
                end = min(len(words), i + self.window_size + 1)
                for j in range(start, end):
                    if i != j:  # 避免自身计数
                        context_word_index = vocab_index[words[j]]
                        cooccurrence_matrix[target_word_index, context_word_index] += 1

        return cooccurrence_matrix, vocab_index

    def apply_svd(self, cooccurrence_matrix):
        """对共现矩阵应用SVD分解，并保留前k个奇异值"""
        k = self.k
        U, Sigma, VT = svd(cooccurrence_matrix, full_matrices=False)
        # 仅保留前k个特征向量
        U_k = U[:, :k]
        return U_k

    def cosine_similarity(self, vec1, vec2):
        """计算两个向量之间的余弦相似度"""
        dot_product = np.dot(vec1, vec2)
        norm_vec1 = np.linalg.norm(vec1)
        norm_vec2 = np.linalg.norm(vec2)
        similarity = dot_product / (norm_vec1 * norm_vec2)
        return similarity

    def calculate_similarity_pairs(self, reduced_matrix, vocab_index):
        """计算pku_sim_test.txt中每一行两个词的余弦相似度"""
        similarities = []
        with open(self.test_path, 'r', encoding='utf-8') as f:  # 确保使用正确的编码
            for line in f:
                word1, word2 = line.strip().split()
                # 检查词汇是否存在
                if word1 in vocab_index and word2 in vocab_index:
                    vec1 = reduced_matrix[vocab_index[word1]]
                    vec2 = reduced_matrix[vocab_index[word2]]
                    similarity = self.cosine_similarity(vec1, vec2)
                else:
                    similarity = 0  # 如果任一词汇不存在，相似度为0
                similarities.append((word1, word2, similarity))
        return similarities


if __name__ == '__main__':
    stopwords_path = '../data/stopwords.txt'
    train_path = '../data/training.txt'
    test_path = '../data/pku_sim_test.txt'
    svd_dec = Svd_dec(stopwords_path, train_path, test_path)

    # 对training.txt文件进行预处理，包括分词、去除停用词（如果需要）等操作，以获得文本的基本单位（如词或子词）。
    processed_sentences = svd_dec.preprocess_file()
    # 构建共现矩阵：基于预处理后的语料，构建一个共现矩阵。在这个矩阵中，行和列分别代表语料库中的唯一词汇，矩阵中的每个元素代表对应行词和列词共同出现在一定窗口大小内的次数。
    cooccurrence_matrix, vocab_index = svd_dec.build_cooccurrence_matrix(processed_sentences)
    # 应用SVD分解：对共现矩阵应用奇异值分解（SVD），以减少特征空间的维度。这里可以选择降维后的维数，但要保持K=5。
    reduced_matrix = svd_dec.apply_svd(cooccurrence_matrix)
    # 计算词向量相似度：使用SVD分解的结果获取每个词的向量表示。然后，基于这些向量，计算pku_sim_test.txt中每一行两个词的余弦相似度。如果某个词在training.txt中没有出现，将这对词的相似度设为0。
    similarities = svd_dec.calculate_similarity_pairs(reduced_matrix, vocab_index)

    # 打印结果
    for word1, word2, similarity in similarities:
        print(f"{word1} - {word2}: {similarity}")
