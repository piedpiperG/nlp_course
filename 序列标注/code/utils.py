import matplotlib.pyplot as plt


def read_tags(file_path):
    unique_tags = set()
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            tags = line.strip().split()
            unique_tags.update(tags)
    return unique_tags


def read_data(text_file, tag_file):
    texts = []
    tags = []

    with open(text_file, 'r', encoding='utf-8') as file:
        text_lines = [line.strip().split() for line in file.readlines()]

    with open(tag_file, 'r', encoding='utf-8') as file:
        tag_lines = [line.strip().split() for line in file.readlines()]

    # 对每一行文本和标签进行处理
    for text_line, tag_line in zip(text_lines, tag_lines):
        sentence = []
        sentence_tags = []
        for word, tag in zip(text_line, tag_line):
            sentence.append(word)
            sentence_tags.append(tag)
            # 检查是否为句子的结尾（逗号或句号）
            if word in {',', '。', '、', ';', ':', '('}:
                texts.append(sentence)
                tags.append(sentence_tags)
                sentence = []
                sentence_tags = []

        # 如果一行结束后还有剩余词汇没有被加入，将它们作为一个单独的句子
        if sentence:
            texts.append(sentence)
            tags.append(sentence_tags)

    return texts, tags


def read_test_data(text_file):
    texts = []

    with open(text_file, 'r', encoding='utf-8') as file:
        text_lines = [line.strip().split() for line in file.readlines()]

    # 对每一行文本和标签进行处理
    for text_line in text_lines:
        sentence = []

        for word in text_line:
            sentence.append(word)

            # 检查是否为句子的结尾（逗号或句号）
            if word in {',', '。', '、', ';', ':', '('}:
                texts.append(sentence)

                sentence = []

        # 如果一行结束后还有剩余词汇没有被加入，将它们作为一个单独的句子
        if sentence:
            texts.append(sentence)

    return texts


def create_tag_to_ix(tags):
    tag_set = set()
    for tag_list in tags:
        tag_set.update(tag_list)
    return {tag: i for i, tag in enumerate(tag_set)}


def plot_losses(losses, file_path):
    plt.figure(figsize=(10, 5))
    plt.plot(losses, label='Training Loss')
    plt.title('Loss During Training')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(file_path)  # Save the figure
    plt.show()  # Optionally display the figure


texts, labels = read_data('../data/train.txt', '../data/train_TAG.txt')
tag_to_ix = {'I_LOC': 0, 'B_LOC': 1, 'I_T': 2, 'O': 3, 'I_ORG': 4, 'B_T': 5, 'I_PER': 6, 'B_PER': 7, 'B_ORG': 8}

if __name__ == "__main__":
    print(read_tags('../data/train_TAG.txt'))
    texts, tags = read_data('../data/dev.txt', '../data/dev_TAG.txt')
    # for i in range(0, len(texts)):
    # print(f'texts{i}:{len(texts[i])}')
    # print(len(tags[i]))
    # if len(texts[i]) > 128:
    #     print(f'texts{i}:{len(texts[i])}')
    #     print(texts[i])
    # print(texts[i])

    # print(f'texts{3}:{len(texts[3])}')
    # print(len(tags[3]))
    # print(texts[3])
    # print(tags[3])
    # import os
    #
    # os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'
    # from dataset import NERDataset
    # from transformers import BertTokenizer
    #
    # # 假设已经有tokenizer, texts, labels等数据准备好
    # tokenizer = BertTokenizer.from_pretrained('../bert_model/bert-base-chinese')
    # tag_to_ix = create_tag_to_ix(tags)
    # print(len(tag_to_ix))
    # # 将标签文本转换为索引
    # labels = [[tag_to_ix[tag] for tag in label] for label in tags]
    # dataset = NERDataset(texts, labels, tokenizer)
    # # 选择一个索引，比如第一个样本
    # i = 5
    # dataset.__getitem__(i)
    # print(f'texts:{len(texts[i])}')
    # print(texts[i])

    test_texts = read_test_data('../data/dev.txt')
    # for i in range(0, len(test_texts)):
    #     if len(test_texts[i]) > 128:
    #         print(f'texts{i}:{len(test_texts[i])}')
    #         print(test_texts[i])



    # texts = read_test_data('../data/dev.txt')
    # # 创建与 texts 形状相同但元素全为 1 的 labels 数组
    # labels = [[1] * len(sentence) for sentence in texts]
    # for i in range(0, 10):
    #     print(len(texts[i]))
    #     print(texts[i])
    #     print(len(labels[i]))
    #     print(labels[i])
