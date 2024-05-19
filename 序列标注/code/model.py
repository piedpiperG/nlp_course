import torch.nn as nn
from transformers import AutoModel


class BertForNER(nn.Module):
    """
    用于命名实体识别（NER）的BERT模型。

    这个模型使用了预训练的BERT模型作为基础，然后在顶部添加了一个线性层来进行实体识别的分类任务。
    """

    def __init__(self, bert_model, num_labels, hidden_size=768, dropout_prob=0.01):
        """
        初始化BertForNER模型。

        参数:
            bert_model (str): 预训练BERT模型的名称或路径，用于加载模型。
            num_labels (int): 标签的数量，即模型需要分类的实体类别数。
            hidden_size (int): BERT模型的隐藏层大小。默认为768，这是BERT Base模型的隐藏层大小。
            dropout_prob (float): Dropout层的概率，用于减少过拟合。
        """
        super(BertForNER, self).__init__()  # 调用父类的构造函数
        self.bert = AutoModel.from_pretrained(bert_model)  # 从预训练模型加载BERT
        self.dropout = nn.Dropout(dropout_prob)  # 初始化Dropout层
        self.classifier = nn.Linear(hidden_size, num_labels)  # 初始化一个线性层，从隐藏状态到标签的映射
        self.num_labels = num_labels  # 存储标签的数量

    def forward(self, input_ids, attention_mask=None):
        """
        定义模型的前向传播路径。

        参数:
            input_ids (torch.Tensor): 输入数据的ID张量，通常是编码后的文本数据。
            attention_mask (torch.Tensor, optional): 用于指示哪些部分是真实数据，哪些部分是填充数据的掩码张量。

        返回:
            torch.Tensor: 经过BERT模型和线性层后的输出，这些输出是对每个标签的未规范化分数（logits）。
        """
        outputs = self.bert(input_ids, attention_mask=attention_mask)  # 通过BERT模型获取输出
        sequence_output = outputs.last_hidden_state  # 获取BERT输出的最后一个隐藏状态
        sequence_output = self.dropout(sequence_output)  # 对隐藏状态应用Dropout
        logits = self.classifier(sequence_output)  # 通过线性层获取最终的logits

        return logits  # 返回logits
