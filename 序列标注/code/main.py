import os
import random

os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'

from torch import nn
from transformers import BertTokenizer
import torch.utils.data as Data
from torch.optim import AdamW
import torch
import matplotlib.pyplot as plt
import time

from utils import read_data, create_tag_to_ix
from dataset import NERDataset
from model import BertForNER


def train(model, train_loader, dev_loader, optimizer, epochs=3):
    model.train()
    total_batches = len(train_loader)
    print(total_batches)

    epoch_losses = []

    for epoch in range(epochs):
        epoch_start_time = time.time()  # 开始记录一个周期的时间
        total_loss = 0
        for batch_idx, (input_ids, attention_mask, labels) in enumerate(train_loader):
            batch_start_time = time.time()  # 开始记录一个批次的时间
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
            batch_end_time = time.time()  # 结束一个批次的时间记录

            # 每10个batch检测一次
            if batch_idx % 10 == 0:
                print(
                    f"Epoch {epoch + 1}, Batch {batch_idx}, Loss: {loss.item()}, Completed: {percentage:.2f}%, "
                    f"Batch Time: {batch_end_time - epoch_start_time:.2f} sec")

            if batch_idx % 6000 == 0 and batch_idx != 0:  # 每20个batch执行
                predictions, all_labels = predict(model, dev_loader, {v: k for k, v in tag_to_ix.items()})
                correct = sum(p == t for p, t in zip(predictions, all_labels))
                accuracy = correct / len(all_labels)
                print(f"Accuracy on dev data: {accuracy:.4f}")

                # 保存模型和向量
                torch.save(model.state_dict(), 'model_epoch_{}_batch_{}.pth'.format(epoch + 1, batch_idx))
                torch.save(model.bert.state_dict(), 'bert_vectors_epoch_{}_batch_{}.pth'.format(epoch + 1, batch_idx))

                # 输出预测结果到文件
                with open('predictions_epoch_{}_batch_{}.txt'.format(epoch + 1, batch_idx), 'w') as f:
                    for pred in predictions:
                        f.write(f"{pred}\n")

                print(f"Saved model and predictions at batch {batch_idx} of epoch {epoch + 1}")

        epoch_end_time = time.time()  # 结束一个周期的时间记
        epoch_losses.append(total_loss / total_batches)
        print(f"Epoch {epoch + 1}, Average Loss: {total_loss / total_batches}, "
              f"Epoch Time: {epoch_end_time - epoch_start_time:.2f} sec")

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


def predict(model, data_loader, ix_to_tag):
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

            preds = preds.view(-1)
            labels = labels.view(-1)

            # 转换索引回标签
            pred_tags = [ix_to_tag[ix] for ix in preds.cpu().numpy() if ix != -100]
            true_tags = [ix_to_tag[ix] for ix in labels.cpu().numpy() if ix != -100]

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

    batch_size = 32

    train_dataset = NERDataset(texts, labels, tokenizer)
    train_loader = Data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

    model = BertForNER('../bert_model/bert-base-chinese', num_labels=len(labels)).to(device)
    model.train()

    optimizer = AdamW(model.parameters(), lr=5e-5)

    # 加载验证数据
    dev_texts, dev_labels = read_data('../data/dev.txt', '../data/dev_TAG.txt')
    dev_labels = [[tag_to_ix[tag] for tag in label] for label in dev_labels]
    dev_dataset = NERDataset(dev_texts, dev_labels, tokenizer)
    dev_loader = Data.DataLoader(dataset=dev_dataset, batch_size=batch_size, shuffle=False)

    # 训练模型并绘制损失
    losses = train(model, train_loader, dev_loader, optimizer)
    # 设置文件保存路径
    file_path = 'training_loss_plot.png'
    plot_losses(losses, file_path)

    # 加载模型
    model.load_state_dict(torch.load('best_model.pth'))

    # 获取预测结果
    predictions, all_labels = predict(model, dev_loader, {v: k for k, v in tag_to_ix.items()})

    # 输出到文件
    with open('predicted_results.txt', 'w') as f:
        for pred in predictions:
            f.write(f"{pred}\n")

    # 计算准确率
    correct = sum(p == t for p, t in zip(predictions, all_labels))
    accuracy = correct / len(all_labels)
    print(f"Accuracy on test data: {accuracy:.4f}")
