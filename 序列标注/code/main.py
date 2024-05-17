import os

os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'

from torch import nn
from transformers import BertTokenizer
import torch.utils.data as Data
from torch.optim import AdamW
import torch
from tqdm import tqdm

from utils import read_data, create_tag_to_ix, plot_losses
from dataset import NERDataset
from model import BertForNER


def train(model, train_loader, dev_loader, optimizer, epochs=3):
    model.train()
    total_items = len(train_loader.dataset)  # 获取总数据项数
    epoch_losses = []

    for epoch in range(epochs):
        total_loss = 0
        processed_items = 0  # Track number of processed items

        # 初始化tqdm进度条
        train_progress_bar = tqdm(train_loader, total=total_items, desc=f"Epoch {epoch + 1}/{epochs}", leave=True)

        for (input_ids, attention_mask, labels) in train_progress_bar:
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask=attention_mask)
            loss = nn.CrossEntropyLoss()(outputs.view(-1, model.num_labels), labels.view(-1))
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            processed_items += input_ids.size(0)  # Increment by the number of items in the batch

            # 更新tqdm进度条描述信息
            train_progress_bar.update(input_ids.size(0))
            train_progress_bar.set_description(
                f"Epoch {epoch + 1}/{epochs}, Processed {processed_items}/{total_items}, Loss: {loss.item():.4f}")

        # Save model and vectors, evaluate model
        torch.save(model.state_dict(), f'model_epoch_{epoch + 1}.pth')
        torch.save(model.bert.state_dict(), f'bert_vectors_epoch_{epoch + 1}.pth')
        accuracy = evaluate(model, dev_loader, epoch_number=epoch, results_dir='predictions')
        print(f"Epoch {epoch + 1}, Average Loss: {total_loss / len(train_loader):.4f}, Accuracy: {accuracy:.4f}")

        epoch_losses.append(total_loss / len(train_loader))

    return epoch_losses


def evaluate(model, dev_loader, epoch_number, results_dir='predictions'):
    model.eval()
    total_items = len(dev_loader.dataset)  # 获取总数据项数
    correct, total = 0, 0
    eval_progress_bar = tqdm(dev_loader, total=total_items, desc="Evaluating", leave=False)

    # 确保结果目录存在
    os.makedirs(results_dir, exist_ok=True)

    # 为这个训练轮次创建一个新的文件
    results_filepath = os.path.join(results_dir, f'epoch_{epoch_number}_predictions.txt')

    with torch.no_grad():
        with open(results_filepath, 'w') as f:
            for input_ids, attention_mask, labels in eval_progress_bar:
                input_ids = input_ids.to(device)
                attention_mask = attention_mask.to(device)
                labels = labels.to(device)

                outputs = model(input_ids, attention_mask=attention_mask)
                _, predicted = torch.max(outputs, dim=2)

                # 获取非[PAD]部分的标签
                active_positions = labels != -100
                true_labels = labels[active_positions]
                predicted_labels = predicted[active_positions]

                correct += (predicted_labels == true_labels).sum().item()
                total += true_labels.shape[0]

                # 将预测结果的索引转换为标签名称
                ix_to_tag = {ix: tag for tag, ix in tag_to_ix.items()}
                predicted_tags = [ix_to_tag[ix] for ix in predicted_labels.cpu().numpy()]

                # 写入转换后的标签名称到文件
                for tag in predicted_tags:
                    f.write(f"{tag}\n")

    accuracy = correct / total if total > 0 else 0
    tqdm.write(f"Evaluation Accuracy: {accuracy:.4f}")
    return accuracy


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    tokenizer = BertTokenizer.from_pretrained('../bert_model/bert-base-chinese')
    texts, labels = read_data('../data/train.txt', '../data/train_TAG.txt')
    tag_to_ix = create_tag_to_ix(labels)
    # 将标签文本转换为索引
    labels = [[tag_to_ix[tag] for tag in label] for label in labels]
    batch_size = 256

    # 加载训练数据
    train_dataset = NERDataset(texts, labels, tokenizer)
    train_loader = Data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    # 加载验证数据
    dev_texts, dev_labels = read_data('../data/dev.txt', '../data/dev_TAG.txt')
    dev_dataset = NERDataset(dev_texts, dev_labels, tokenizer)
    dev_loader = Data.DataLoader(dataset=dev_dataset, batch_size=batch_size, shuffle=False)

    # 加载模型，并设置为训练模式
    model = BertForNER('../bert_model/bert-base-chinese', num_labels=len(tag_to_ix)).to(device)
    model.train()
    optimizer = AdamW(model.parameters(), lr=5e-5)

    # 训练模型并绘制损失
    losses = train(model, train_loader, dev_loader, optimizer)
    # 设置文件保存路径
    file_path = 'training_loss_plot.png'
    plot_losses(losses, file_path)
