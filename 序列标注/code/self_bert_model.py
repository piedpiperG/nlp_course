import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEmbedding(nn.Module):

    def __init__(self, d_model, max_len=512):
        super().__init__()

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)]


class MultiHeadAttention(nn.Module):
    def __init__(self, heads, d_model, dropout=0.1):
        super().__init__()

        self.d_model = d_model
        self.d_k = d_model // heads
        self.h = heads

        self.q_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)

        self.dropout1 = nn.Dropout(dropout)
        self.proj = nn.Linear(d_model, d_model)
        self.dropout2 = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, q, k, v, mask=None):
        # q:[b_size, len_q, d_model] -> [b_size, len_q, head x d_k] -> [b_size, len_q, head, d_k] -> [b_size, head, len_q, d_k]

        bs = q.size(0)
        residual = q

        # q:[bs x head x len_q x d_k]
        # k:[bs x head x len_k x d_k]
        # v:[bs x head x len_k x d_k]
        k = self.k_linear(k).view(bs, -1, self.h, self.d_k).transpose(1, 2)
        q = self.q_linear(q).view(bs, -1, self.h, self.d_k).transpose(1, 2)
        v = self.v_linear(v).view(bs, -1, self.h, self.d_k).transpose(1, 2)

        # 计算attention
        # scores:[bs x head x len_q x len_k]
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        # mask:[bs x len_q x len_k]
        if mask is not None:
            # 将 mask 的形状调整为 [bs, 1, 1, len_k]
            mask = mask.unsqueeze(1).unsqueeze(2)

            # print("Scores shape:", scores.shape)
            # print("Mask shape:", mask.shape)

            scores = scores.masked_fill(mask == 0, -1e9)

        # attn:[bs x head x len_q x len_k]
        attn = self.dropout1(F.softmax(scores, dim=-1))

        # out:[bs x len_q x d_model]
        out = torch.matmul(attn, v).transpose(1, 2).contiguous().view(bs, -1, self.d_model)
        out = self.dropout2(self.proj(out))
        out = self.layer_norm(residual + out)
        return out, attn


class FeedForwardLayer(nn.Module):
    def __init__(self, d_model, d_ff=2048, dropout=0.1):
        super().__init__()

        # We set d_ff as a default to 2048
        self.linear_1 = nn.Linear(d_model, d_ff)
        self.dropout1 = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model)
        self.dropout2 = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, x):
        residual = x
        out = self.dropout1(F.relu(self.linear_1(x)))
        out = self.dropout2(F.relu(self.linear_2(out)))
        out = self.layer_norm(residual + out)
        return out


class BertModel(nn.Module):
    def __init__(self, n_layers, d_model, d_ff, n_heads,
                 max_seq_len, vocab_size, pad_id, dropout=0.1):
        super(BertModel, self).__init__()
        self.d_model = d_model
        self.pad_id = pad_id
        self.word_emb = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        self.pos_emb = PositionalEmbedding(d_model, max_seq_len)
        self.segment_emb = nn.Embedding(2, d_model, padding_idx=0)
        self.dropout_emb = nn.Dropout(dropout)
        self.layers = nn.ModuleList(
            [EncoderLayer(d_model, d_ff, n_heads, dropout) for _ in range(n_layers)])

    def forward(self, input_ids, token_type_ids, mask, return_attn=False):
        # input_ids:[b_size, len]
        out = self.word_emb(input_ids) + self.pos_emb(input_ids) + self.segment_emb(token_type_ids)
        out = self.dropout_emb(out)
        attn_list = []
        for layer in self.layers:
            out, attn = layer(out, mask)
            if return_attn:
                attn_list.append(attn)

        out = (out, attn_list) if return_attn else out
        return out


class EncoderLayer(nn.Module):
    def __init__(self, d_model, d_ff, n_heads, dropout):
        super(EncoderLayer, self).__init__()
        self.attention = MultiHeadAttention(n_heads, d_model, dropout)
        self.feed_forward = FeedForwardLayer(d_model, d_ff, dropout)

    def forward(self, inputs, mask):
        out, attn = self.attention(inputs, inputs, inputs, mask)
        out = self.feed_forward(out)

        return out, attn


class selfBertForNER(nn.Module):
    def __init__(self, n_layers, d_model, d_ff, n_heads, max_seq_len, vocab_size, pad_id, dropout=0.1, num_labels=9):
        super(selfBertForNER, self).__init__()
        self.d_model = d_model
        self.pad_id = pad_id
        self.num_labels = num_labels
        self.word_emb = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        self.pos_emb = PositionalEmbedding(d_model, max_seq_len)
        self.segment_emb = nn.Embedding(2, d_model, padding_idx=0)
        self.dropout_emb = nn.Dropout(dropout)
        self.layers = nn.ModuleList([EncoderLayer(d_model, d_ff, n_heads, dropout) for _ in range(n_layers)])
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(d_model, num_labels)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, return_attn=False):
        out = self.word_emb(input_ids) + self.pos_emb(input_ids)
        if token_type_ids is not None:
            out += self.segment_emb(token_type_ids)
        out = self.dropout_emb(out)

        attn_list = []
        for layer in self.layers:
            if attention_mask is not None:
                out, attn = layer(out, attention_mask)
            else:
                out, attn = layer(out)
            if return_attn:
                attn_list.append(attn)

        out = self.dropout(out)

        # 打印输出形状
        # print("BERT output shape:", out.shape)

        # 调整形状: [batch_size, seq_length, d_model] -> [batch_size * seq_length, d_model]
        batch_size, seq_length, d_model = out.shape
        out = out.view(batch_size * seq_length, d_model)

        # 打印调整后的 out 形状
        # print("Sequence output shape after view:", out.shape)

        logits = self.classifier(out)

        # 将形状恢复: [batch_size * seq_length, num_labels] -> [batch_size, seq_length, num_labels]
        logits = logits.view(batch_size, seq_length, -1)

        return (logits, attn_list) if return_attn else logits
