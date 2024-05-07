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


def train(model, train_loader, dev_loader, optimizer, epochs=3, patience=2):
    model.train()
    total_batches = len(train_loader)
    epoch_losses = []
    best_accuracy = 0
    patience_counter = 0

    for epoch in range(epochs):
        total_loss = 0
        for batch_idx, (input_ids, attention_mask, labels) in enumerate(train_loader):
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask=attention_mask)
            loss = nn.CrossEntropyLoss()(outputs.view(-1, model.num_labels), labels.view(-1))
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            percentage = (batch_idx + 1) / total_batches * 100

            # 每10个batch检测一次
            if batch_idx % 10 == 0:
                accuracy = compute_accuracy(model, dev_loader)
                print(
                    f"Epoch {epoch + 1}, Batch {batch_idx}, Loss: {loss.item()}, Completed: {percentage:.2f}%, Validation Accuracy: {accuracy:.4f}")

                # 早停逻辑
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        print("Stopping early due to no improvement in validation accuracy.")
                        return epoch_losses

        epoch_losses.append(total_loss / total_batches)
        print(f"Epoch {epoch + 1}, Average Loss: {total_loss / total_batches}")

    return epoch_losses


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


# 评估函数
def evaluate(model, dev_loader):
    model.eval()
    total_eval_accuracy = 0
    total_eval_loss = 0
    eval_steps = 0

    with torch.no_grad():
        for input_ids, attention_mask, labels in dev_loader:
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)

            outputs = model(input_ids, attention_mask=attention_mask)
            loss = nn.CrossEntropyLoss()(outputs.view(-1, model.num_labels), labels.view(-1))

            total_eval_loss += loss.item()
            eval_steps += 1

    avg_loss = total_eval_loss / eval_steps
    print(f"Validation Loss: {avg_loss}")


def compute_accuracy(model, data_loader, sample_ratio=0.1):
    model.eval()
    correct_predictions = 0
    total_predictions = 0

    # 随机选择部分数据进行评估
    data_list = list(data_loader)
    sample_size = max(1, int(len(data_list) * sample_ratio))
    sampled_data = random.sample(data_list, sample_size)

    with torch.no_grad():
        for input_ids, attention_mask, labels in sampled_data:
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)

            outputs = model(input_ids, attention_mask=attention_mask)
            _, preds = torch.max(outputs, dim=2)  # 获取最可能的类别标签

            # 展平预测和标签以计算准确率
            preds = preds.view(-1)
            labels = labels.view(-1)

            # 忽略特定的标签值（如-100，通常用于NER任务中的非实体标记）
            active_accuracy = labels != -100
            active_preds = preds[active_accuracy]
            active_labels = labels[active_accuracy]

            correct_predictions += torch.sum(active_preds == active_labels)
            total_predictions += active_labels.size(0)

    accuracy = correct_predictions.double() / total_predictions if total_predictions > 0 else 0
    print(f"Computed accuracy over {sample_size} batches: {accuracy:.4f}")
    return accuracy


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

tokenizer = BertTokenizer.from_pretrained('../bert_model/bert-base-chinese')
texts, labels = read_data('../data/train.txt', '../data/train_TAG.txt')

tag_to_ix = create_tag_to_ix(labels)

# 将标签文本转换为索引
labels = [[tag_to_ix[tag] for tag in label] for label in labels]

train_dataset = NERDataset(texts, labels, tokenizer)
train_loader = Data.DataLoader(dataset=train_dataset, batch_size=32, shuffle=True)

model = BertForNER('../bert_model/bert-base-chinese', num_labels=len(labels)).to(device)
model.train()

optimizer = AdamW(model.parameters(), lr=5e-5)

# 加载验证数据
dev_texts, dev_labels = read_data('../data/dev.txt', '../data/dev_TAG.txt')
dev_labels = [[tag_to_ix[tag] for tag in label] for label in dev_labels]
dev_dataset = NERDataset(dev_texts, dev_labels, tokenizer)
dev_loader = Data.DataLoader(dataset=dev_dataset, batch_size=8, shuffle=False)

# 训练模型并绘制损失
losses = train(model, train_loader, dev_loader, optimizer)
# 设置文件保存路径
file_path = 'training_loss_plot.png'
plot_losses(losses, file_path)
# 开始评估
evaluate(model, dev_loader)
