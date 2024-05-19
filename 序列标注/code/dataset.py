import torch
from torch.utils.data import Dataset


class NERDataset(Dataset):
    """
    用于命名实体识别任务的自定义数据集类。

    这个类继承自 PyTorch 的 Dataset 类，主要用于处理文本和对应标签，以及将文本通过tokenizer转换为模型可处理的格式。
    """

    def __init__(self, texts, labels, tokenizer, max_length=128):
        """
        初始化数据集。

        参数:
            texts (list): 包含所有句子的列表。
            labels (list): 每个句子对应的标签列表，每个标签列表与句子中的词汇一一对应。
            tokenizer: 用于文本编码的分词器。
            max_length (int): 每个句子编码后的最大长度。
        """
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        """
        返回数据集中的样本总数。
        """
        return len(self.texts)

    def __getitem__(self, index):
        """
        根据给定的索引获取一个样本。

        参数:
            index (int): 要获取的样本的索引值。

        返回:
            tuple: 包含输入ID、注意力掩码和标签的元组。
        """
        sentence = self.texts[index]  # 获取对应索引的句子
        label = self.labels[index]  # 获取对应索引的标签列表

        # 使用tokenizer编码句子
        encoded = self.tokenizer.encode_plus(
            sentence,
            max_length=self.max_length,  # 设定最大长度
            padding='max_length',  # 进行填充到最大长度
            truncation=True,  # 若句子超长，则进行截断
            return_tensors='pt'  # 返回PyTorch张量
        )

        input_ids = encoded['input_ids'].squeeze(0)  # 获取编码后的输入ID张量并移除批量维度
        attention_mask = encoded['attention_mask'].squeeze(0)  # 获取注意力掩码张量并移除批量维度

        # 处理标签，使其长度与输入ID一致
        if isinstance(label, list):
            # 对于标签是列表的情况，将标签转换为ID，并在两端添加忽略标记（-100）
            label_ids = [-100] + label[:self.max_length - 2] + [-100]
        else:  # 如果标签不是列表，即标签数据异常
            # 创建一个全为忽略标记的标签列表
            label_ids = [-100] * self.max_length

        # 确保标签长度与max_length一致
        if len(label_ids) < self.max_length:
            label_ids += [-100] * (self.max_length - len(label_ids))  # 使用忽略标记进行填充

        return input_ids, attention_mask, torch.tensor(label_ids, dtype=torch.long)
