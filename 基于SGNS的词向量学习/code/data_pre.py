import pickle
import random


class Data_prepare:

    def __init__(self, stopwords_path, train_path, window_size, num_negative_samples):
        self.stopwords_path = stopwords_path
        self.train_path = train_path
        self.window_size = window_size
        self.processed_sentences = None
        self.vocab = None
        self.num_negative_samples = num_negative_samples

    # 分词
    def preprocess_file(self):
        """从文件中读取文本并预处理：去除停用词"""
        processed_sentences = []
        with open(self.train_path, 'r', encoding='utf-8') as f:
            for line in f:
                words = line.strip().split()
                filtered_words = [word for word in words if word.strip() != '']
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
        self.vocab = vocab
        return vocab

    def generate_training_data(self):
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
            while len(negatives) < self.num_negative_samples:
                negative = random.choice(vocab_indices)
                if negative not in negatives:  # 确保负样本是唯一的
                    negatives.append(negative)
            negative_samples.append(negatives)

        return positive_pairs, negative_samples

    # 保存预处理数据
    def save_preprocessed_data(self, positive_pairs, negative_samples, vocab, filepath):
        with open(filepath, 'wb') as f:
            pickle.dump({
                'positive_pairs': positive_pairs,
                'negative_samples': negative_samples,
                'vocab': vocab
            }, f)
        print(f"Preprocessed data saved to {filepath}")

    def pre_data(self):
        self.preprocess_file()
        self.build_vocab()
        positive_pairs, negative_samples = self.generate_training_data()

        # 输出样本数量进行检查
        print(f"词汇储量为：{len(self.vocab)}")
        print(f"总共有{len(positive_pairs)}个正样本对")
        print(f"每个正样本对对应{len(negative_samples[0])}个负样本")

        self.save_preprocessed_data(positive_pairs, negative_samples, self.vocab, 'preprocessed_data.pkl')
        return positive_pairs, negative_samples, len(self.vocab), self.vocab
