import os
import random

os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'

from torch import nn
from utils import read_data, create_tag_to_ix
from transformers import BertTokenizer
import torch.utils.data as Data
from torch.optim import AdamW
import torch
import matplotlib.pyplot as plt

from dataset import NERDataset
from model import BertForNER


def predict(model, data_loader, ix_to_tag, device):
    model.eval()
    predictions = []
    all_labels = []
    with torch.no_grad():
        for input_ids, attention_mask, labels in data_loader:
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)

            outputs = model(input_ids, attention_mask=attention_mask)
            _, preds = torch.max(outputs, dim=2)

            # 处理每个预测序列
            for i, input_id in enumerate(input_ids):
                input_mask = attention_mask[i]
                length = input_mask.sum()  # 实际长度
                pred = preds[i, :length]  # 截取到实际长度
                label = labels[i, :length]

                # 转换索引回标签
                pred_tags = [ix_to_tag[ix] for ix in pred.cpu().numpy() if ix != -100]
                true_tags = [ix_to_tag[ix] for ix in label.cpu().numpy() if ix != -100]

                predictions.extend(pred_tags)
                all_labels.extend(true_tags)

    return predictions, all_labels


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    tokenizer = BertTokenizer.from_pretrained('../bert_model/bert-base-chinese')
    texts, labels = read_data('../data/train.txt', '../data/train_TAG.txt')

    tag_to_ix = create_tag_to_ix(labels)

    # 将标签文本转换为索引
    labels = [[tag_to_ix[tag] for tag in label] for label in labels]

    model = BertForNER('../bert_model/bert-base-chinese', num_labels=len(labels)).to(device)
    model.load_state_dict(torch.load('model_epoch_1_batch_3000.pth'))
    model.eval()  # 将模型设置为评估模式

    # 加载验证数据
    dev_texts, dev_labels = read_data('../data/dev.txt', '../data/dev_TAG.txt')
    dev_labels = [[tag_to_ix[tag] for tag in label] for label in dev_labels]
    dev_dataset = NERDataset(dev_texts, dev_labels, tokenizer)
    dev_loader = Data.DataLoader(dataset=dev_dataset, batch_size=32, shuffle=False)

    # 获取预测结果
    predictions, all_labels = predict(model, dev_loader, {v: k for k, v in tag_to_ix.items()}, device)

    # 输出到文件
    with open('predicted_results.txt', 'w') as f:
        for pred in predictions:
            f.write(f"{pred}\n")

    # 计算准确率
    correct = sum(p == t for p, t in zip(predictions, all_labels))
    accuracy = correct / len(all_labels)
    print(f"Accuracy on test data: {accuracy:.4f}")
