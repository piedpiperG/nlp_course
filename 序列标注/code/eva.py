import os

os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'

from torch import nn
from transformers import BertTokenizer
import torch.utils.data as Data
from torch.optim import AdamW
import torch
from tqdm import tqdm
from torch.nn import DataParallel

from utils import read_data, create_tag_to_ix, plot_losses, read_test_data
from dataset import NERDataset
from model import BertForNER

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = BertTokenizer.from_pretrained('../bert_model/bert-base-chinese')
    batch_size = 256
    texts = read_test_data('../data/test.txt')
    labels = [[1] * len(sentence) for sentence in texts]

    tag_to_ix = {'I_LOC': 0, 'B_LOC': 1, 'I_T': 2, 'O': 3, 'I_ORG': 4, 'B_T': 5, 'I_PER': 6, 'B_PER': 7, 'B_ORG': 8}

    dataset = NERDataset(texts, labels, tokenizer)
    data_loader = Data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False)

    model = BertForNER('../bert_model/bert-base-chinese', num_labels=9).to(device)
    model = DataParallel(model)  # 包装你的模型以使用DataParallel
    model.load_state_dict(torch.load('model_epoch_1.pth'))
    model.eval()

    eval_progress_bar = tqdm(data_loader, total=len(data_loader.dataset), desc='Evaluating', leave=False)

    with torch.no_grad():
        with open('predictions.txt', 'w') as f:
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

                # 将预测结果的索引转换为标签名称
                ix_to_tag = {ix: tag for tag, ix in tag_to_ix.items()}
                predicted_tags = [ix_to_tag[ix] for ix in predicted_labels.cpu().numpy()]

                # 写入转换后的标签名称到文件
                for tag in predicted_tags:
                    f.write(f"{tag}\n")

                eval_progress_bar.update(input_ids.size(0))
