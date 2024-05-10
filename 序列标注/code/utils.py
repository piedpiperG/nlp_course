def read_tags(file_path):
    unique_tags = set()
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            tags = line.strip().split()
            unique_tags.update(tags)
    return unique_tags


def read_data(text_file, tag_file):
    with open(text_file, 'r', encoding='utf-8') as file:
        # 读取每一行，分割每行的字符
        texts = [line.strip().split() for line in file.readlines()]

    with open(tag_file, 'r', encoding='utf-8') as file:
        # 读取每一行，分割每行的标签
        tags = [line.strip().split() for line in file.readlines()]

    return texts, tags


def create_tag_to_ix(tags):
    tag_set = set()
    for tag_list in tags:
        tag_set.update(tag_list)
    return {tag: i for i, tag in enumerate(tag_set)}


if __name__ == "__main__":
    print(read_tags('../data/train_TAG.txt'))
    texts, tags = read_data('../data/dev.txt', '../data/dev_TAG.txt')
    # for i in range(0, len(texts)):
    #     print(f'texts:{len(texts[i])}')

    import os

    os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'
    from dataset import NERDataset
    from transformers import BertTokenizer

    # 假设已经有tokenizer, texts, labels等数据准备好
    tokenizer = BertTokenizer.from_pretrained('../bert_model/bert-base-chinese')
    tag_to_ix = create_tag_to_ix(tags)
    # 将标签文本转换为索引
    labels = [[tag_to_ix[tag] for tag in label] for label in tags]
    dataset = NERDataset(texts, labels, tokenizer)
    # 选择一个索引，比如第一个样本
    i = 5
    dataset.__getitem__(i)
    print(f'texts:{len(texts[i])}')
    print(texts[i])
