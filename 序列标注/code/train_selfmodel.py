import os

# 设置 Protocol Buffers 的 Python 实现方式为 Python 原生，而不是默认的 C++ 实现，以解决兼容性问题
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'

import torch
import torch.nn as nn
import torch.utils.data as Data
from torch.optim import AdamW
from tqdm import tqdm  # 用于显示进度条
from torch.nn import DataParallel  # 允许模型在多个GPU上并行运行
from transformers import BertTokenizerFast, BertTokenizer

# 导入自定义的模块，确保正确引入了自定义的BERT模型和其他工具函数
from self_bert_model import selfBertForNER
from utils import read_data, create_tag_to_ix, plot_losses_and_accuracy
from dataset import NERDataset


def train(model, train_loader, dev_loader, optimizer, epochs=20):
    """训练模型的函数。

    参数:
        model (nn.Module): 要训练的模型。
        train_loader (Data.DataLoader): 训练数据的加载器。
        dev_loader (Data.DataLoader): 验证数据的加载器。
        optimizer (torch.optim.Optimizer): 优化器。
        epochs (int): 训练的轮数。
    返回:
        list: 每个epoch的平均损失列表。
    """
    model.train()  # 将模型设置为训练模式
    total_items = len(train_loader.dataset)  # 总训练样本数
    epoch_losses = []  # 存储每个epoch的损失
    accuracy_list = []
    # if_first = True

    for epoch in range(epochs):
        total_loss = 0
        processed_items = 0

        # 初始化进度条
        train_progress_bar = tqdm(train_loader, total=total_items, desc=f"Epoch {epoch + 1}/{epochs}", leave=True)

        for (input_ids, attention_mask, labels) in train_progress_bar:
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()  # 清空之前的梯度
            outputs = model(input_ids, attention_mask=attention_mask)  # 计算模型输出
            loss = nn.CrossEntropyLoss()(outputs.view(-1, model.module.num_labels), labels.view(-1))  # 计算损失
            loss.backward()  # 反向传播
            optimizer.step()  # 更新模型参数

            # if if_first:
            #     epoch_losses.append(loss.item())
            #     if_first = False

            total_loss += loss.item()  # 累加损失
            processed_items += input_ids.size(0)  # 更新处理过的样本数
            train_progress_bar.update(input_ids.size(0))  # 更新进度条
            train_progress_bar.set_description(
                f"Epoch {epoch + 1}/{epochs}, Processed {processed_items}/{total_items}, Loss: {loss.item():.4f}")

        # 保存每个epoch的模型
        torch.save(model.state_dict(), f'model_epoch_{epoch + 1}.pth')
        # 评估当前epoch的模型
        accuracy = evaluate(model, dev_loader, epoch_number=epoch, results_dir='predictions')
        print(f"Epoch {epoch + 1}, Average Loss: {total_loss / len(train_loader):.4f}, Accuracy: {accuracy:.4f}")

        epoch_losses.append(total_loss / len(train_loader))
        accuracy_list.append(accuracy)

    return epoch_losses, accuracy_list


def evaluate(model, dev_loader, epoch_number, results_dir='predictions'):
    """评估模型的函数。

    参数:
        model (nn.Module): 要评估的模型。
        dev_loader (Data.DataLoader): 验证数据加载器。
        epoch_number (int): 当前的epoch编号。
        results_dir (str): 保存预测结果文件的目录。
    返回:
        accuracy(float): 准确率。
    """
    model.eval()  # 将模型设置为评估模式
    total_items = len(dev_loader.dataset)  # 总验证样本数
    correct, total = 0, 0  # 初始化正确和总数计数器
    eval_progress_bar = tqdm(dev_loader, total=total_items, desc="Evaluating", leave=False)  # 初始化进度条

    os.makedirs(results_dir, exist_ok=True)  # 创建结果目录
    results_filepath = os.path.join(results_dir, f'epoch_{epoch_number}_predictions.txt')  # 定义结果文件路径

    with torch.no_grad(), open(results_filepath, 'w') as f:
        for input_ids, attention_mask, labels in eval_progress_bar:
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)

            outputs = model(input_ids, attention_mask=attention_mask)  # 计算模型输出
            _, predicted = torch.max(outputs, dim=2)  # 获取预测结果

            active_positions = labels != -100  # 获取有效标签位置
            true_labels = labels[active_positions]
            predicted_labels = predicted[active_positions]

            correct += (predicted_labels == true_labels).sum().item()  # 计算正确预测的数量
            total += true_labels.shape[0]  # 更新总数

            # 将预测结果的索引转换为标签名称
            ix_to_tag = {ix: tag for tag, ix in tag_to_ix.items()}
            predicted_tags = [ix_to_tag[ix] for ix in predicted_labels.cpu().numpy()]
            for tag in predicted_tags:
                f.write(f"{tag}\n")

            eval_progress_bar.update(input_ids.size(0))

    accuracy = correct / total if total > 0 else 0
    tqdm.write(f"Evaluation Accuracy: {accuracy:.4f}")  # 打印准确率
    return accuracy


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # 检查是否可以使用GPU

    tokenizer = BertTokenizer.from_pretrained('../bert_model/bert-base-chinese')  # 加载预训练的BERT分词器

    # 读取训练和验证数据
    texts, labels = read_data('../data/train.txt', '../data/train_TAG.txt')
    tag_to_ix = create_tag_to_ix(labels)  # 创建标签索引
    labels = [[tag_to_ix[tag] for tag in label] for label in labels]  # 转换标签为索引
    batch_size = 4096  # 可能需要根据你的硬件调整批量大小

    # 创建数据集和数据加载器
    train_dataset = NERDataset(texts, labels, tokenizer)
    train_loader = Data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    dev_texts, dev_labels = read_data('../data/dev.txt', '../data/dev_TAG.txt')
    dev_labels = [[tag_to_ix[tag] for tag in label] for label in dev_labels]
    dev_dataset = NERDataset(dev_texts, dev_labels, tokenizer)
    dev_loader = Data.DataLoader(dataset=dev_dataset, batch_size=batch_size, shuffle=False)

    # 初始化模型、并行处理和优化器
    model = selfBertForNER(2, 96, 256, 6, 128, tokenizer.vocab_size, tokenizer.pad_token_id, 0.1, len(tag_to_ix)).to(
        device)
    model = DataParallel(model)
    optimizer = AdamW(model.parameters(), lr=5e-5)  # 可能需要调整学习率

    # 开始训练模型，并绘制损失图
    losses, accuracy = train(model, train_loader, dev_loader, optimizer)
    file_path = 'training_loss_plot.png'
    plot_losses_and_accuracy(losses, accuracy, file_path)  # 绘制损失曲线
