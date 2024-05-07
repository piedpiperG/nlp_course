import torch
from torch.utils.data import Dataset


class NERDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, index):
        sentence = self.texts[index]
        label = self.labels[index]
        encoded = self.tokenizer.encode_plus(
            sentence,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        input_ids = encoded['input_ids'].squeeze(0)
        attention_mask = encoded['attention_mask'].squeeze(0)

        # 调整标签以匹配特殊令牌
        label_ids = [-100] + label[:self.max_length - 2] + [-100]  # 确保不超过 max_length
        label_ids += [-100] * (self.max_length - len(label_ids))  # 填充到 max_length

        return input_ids, attention_mask, torch.tensor(label_ids, dtype=torch.long)
