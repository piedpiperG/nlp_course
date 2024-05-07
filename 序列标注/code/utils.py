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

