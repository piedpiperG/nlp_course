import torch
import torch.nn as nn
from transformers import AutoModel


class BertForNER(nn.Module):
    def __init__(self, bert_model, num_labels, hidden_size=768):
        super(BertForNER, self).__init__()
        self.bert = AutoModel.from_pretrained(bert_model)
        self.classifier = nn.Linear(hidden_size, num_labels)
        self.num_labels = num_labels  # 确保这一行存在

    def forward(self, input_ids, attention_mask=None):
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state
        logits = self.classifier(sequence_output)
        return logits
