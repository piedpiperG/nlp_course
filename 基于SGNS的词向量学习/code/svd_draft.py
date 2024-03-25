from collections import defaultdict
import numpy as np
from scipy.linalg import svd


# 对training.txt文件进行预处理，包括分词、去除停用词（如果需要）等操作，以获得文本的基本单位（如词或子词）。

def load_stopwords(path):
    """加载停用词表"""
    with open(path, 'r', encoding='utf-8') as f:
        stopwords = set([line.strip() for line in f])
    return stopwords


def preprocess_file(text_path, stopwords):
    """从文件中读取文本并预处理：去除停用词"""
    processed_sentences = []
    with open(text_path, 'r', encoding='gbk') as f:
        for line in f:
            words = line.strip().split()
            filtered_words = [word for word in words if word not in stopwords and word.strip() != '']
            processed_sentences.append(" ".join(filtered_words))
    return processed_sentences


def pre_precessing():
    # 加载停用词
    stopwords_path = '../data/stopwords.txt'
    stopwords = load_stopwords(stopwords_path)

    # 指定文本文件路径
    text_path = '../data/pretest.txt'

    # 执行预处理
    preprocessed_sentences = preprocess_file(text_path, stopwords)

    # 打印预处理后的结果
    # for sentence in preprocessed_sentences:
    #     print(sentence)

    return preprocessed_sentences


# 构建共现矩阵：基于预处理后的语料，构建一个共现矩阵。在这个矩阵中，行和列分别代表语料库中的唯一词汇，矩阵中的每个元素代表对应行词和列词共同出现在一定窗口大小内的次数。

def build_cooccurrence_matrix(sentences, window_size=2):
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
            start = max(0, i - window_size)
            end = min(len(words), i + window_size + 1)
            for j in range(start, end):
                if i != j:  # 避免自身计数
                    context_word_index = vocab_index[words[j]]
                    cooccurrence_matrix[target_word_index, context_word_index] += 1

    return cooccurrence_matrix, vocab_index


def act_build_matrix():
    # 使用示例
    preprocessed_sentences = pre_precessing()
    window_size = 2  # 可根据需要调整窗口大小
    cooccurrence_matrix, vocab_index = build_cooccurrence_matrix(preprocessed_sentences, window_size=window_size)

    # 打印共现矩阵和词汇索引（可选，可能输出很大）
    print(cooccurrence_matrix)
    print(vocab_index)

    return cooccurrence_matrix, vocab_index


# 应用SVD分解：对共现矩阵应用奇异值分解（SVD），以减少特征空间的维度。这里可以选择降维后的维数，但要保持K=5。

def apply_svd(cooccurrence_matrix, k=5):
    """对共现矩阵应用SVD分解，并保留前k个奇异值"""
    U, Sigma, VT = svd(cooccurrence_matrix, full_matrices=False)
    # 仅保留前k个特征向量
    U_k = U[:, :k]
    return U_k


def act_build_matrix_and_apply_svd():
    preprocessed_sentences = pre_precessing()
    window_size = 2  # 可根据需要调整窗口大小
    cooccurrence_matrix, vocab_index = build_cooccurrence_matrix(preprocessed_sentences, window_size=window_size)

    # 应用SVD并保留前K=5个特征
    reduced_matrix = apply_svd(cooccurrence_matrix, k=5)

    # 打印降维后的矩阵（可选，可能输出很大）
    print(reduced_matrix)

    return reduced_matrix, vocab_index


# 计算词向量相似度：使用SVD分解的结果获取每个词的向量表示。然后，基于这些向量，计算pku_sim_test.txt中每一行两个词的余弦相似度。如果某个词在training.txt中没有出现，将这对词的相似度设为0。
def cosine_similarity(vec1, vec2):
    """计算两个向量之间的余弦相似度"""
    dot_product = np.dot(vec1, vec2)
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)
    similarity = dot_product / (norm_vec1 * norm_vec2)
    return similarity


def calculate_similarity_pairs(reduced_matrix, vocab_index, test_path):
    """计算pku_sim_test.txt中每一行两个词的余弦相似度"""
    similarities = []
    with open(test_path, 'r', encoding='utf-8') as f:  # 确保使用正确的编码
        for line in f:
            word1, word2 = line.strip().split()
            # 检查词汇是否存在
            if word1 in vocab_index and word2 in vocab_index:
                vec1 = reduced_matrix[vocab_index[word1]]
                vec2 = reduced_matrix[vocab_index[word2]]
                similarity = cosine_similarity(vec1, vec2)
            else:
                similarity = 0  # 如果任一词汇不存在，相似度为0
            similarities.append((word1, word2, similarity))
    return similarities


# 示例使用
reduced_matrix, vocab_index = act_build_matrix_and_apply_svd()
test_path = '../data/pku_sim_test.txt'  # 调整为实际的文件路径
similarities = calculate_similarity_pairs(reduced_matrix, vocab_index, test_path)

# 打印结果
for word1, word2, similarity in similarities:
    print(f"{word1} - {word2}: {similarity}")
