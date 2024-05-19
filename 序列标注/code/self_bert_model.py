import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class TokenPositionalEncoding(nn.Module):
    """基于Transformer模型的词位编码模块。

    该模块为序列中的词元生成正弦波形的位置编码，
    帮助模型理解词元之间的相对位置。
    """

    def __init__(self, embedding_dim, sequence_length=512):
        """初始化位置编码模块。

        参数:
            embedding_dim (int): 模型嵌入的维度。
            sequence_length (int): 序列的最大长度。
        """
        super(TokenPositionalEncoding, self).__init__()
        # 创建一个不需要梯度的位置编码矩阵。
        position_encoding = torch.zeros(sequence_length, embedding_dim).float()
        position_encoding.requires_grad = False

        # 计算位置索引和缩放除数。
        positions = torch.arange(0, sequence_length).unsqueeze(1).float()
        scale_factor = (torch.arange(0, embedding_dim, 2).float() * -(math.log(10000.0) / embedding_dim)).exp()

        # 填充位置编码矩阵。
        position_encoding[:, 0::2] = torch.sin(positions * scale_factor)
        position_encoding[:, 1::2] = torch.cos(positions * scale_factor)

        # 增加一个额外维度以适应批处理大小。
        position_encoding = position_encoding.unsqueeze(0)
        self.register_buffer('position_encoding', position_encoding)

    def forward(self, input_tensor):
        """根据输入批次提取位置编码。

        参数:
            input_tensor (Tensor): 需要提取位置编码的输入张量。

        返回:
            Tensor: 对应于输入张量批次大小和序列长度的位置编码。
        """
        # 返回位置编码，直到输入张量的序列长度。
        return self.position_encoding[:, :input_tensor.size(1)]


class EnhancedMultiHeadAttention(nn.Module):
    """增强多头注意力模块。

    这个模块使用多个注意力头来同时处理信息的不同表示子空间。
    """

    def __init__(self, num_heads, model_dim, dropout_rate=0.1):
        """初始化多头注意力模块。

        参数:
            num_heads (int): 注意力头的数量。
            model_dim (int): 模型的维度。
            dropout_rate (float): Dropout层的概率。
        """
        super(EnhancedMultiHeadAttention, self).__init__()
        self.model_dim = model_dim
        self.key_dim = model_dim // num_heads
        self.num_heads = num_heads

        # 定义用于Q、K、V的线性变换层
        self.query_linear = nn.Linear(model_dim, model_dim)
        self.value_linear = nn.Linear(model_dim, model_dim)
        self.key_linear = nn.Linear(model_dim, model_dim)

        # 定义dropout层和输出的线性变换层
        self.attention_dropout = nn.Dropout(dropout_rate)
        self.output_linear = nn.Linear(model_dim, model_dim)
        self.output_dropout = nn.Dropout(dropout_rate)
        self.normalization = nn.LayerNorm(model_dim)

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
        residual = query

        # 线性变换并重排为多头格式
        key_transformed = self.key_linear(key).view(batch_size, -1, self.num_heads, self.key_dim).transpose(1, 2)
        query_transformed = self.query_linear(query).view(batch_size, -1, self.num_heads, self.key_dim).transpose(1, 2)
        value_transformed = self.value_linear(value).view(batch_size, -1, self.num_heads, self.key_dim).transpose(1, 2)

        # 计算点积注意力得分，并通过key_dim的平方根进行缩放
        attention_scores = torch.matmul(query_transformed, key_transformed.transpose(-2, -1)) / math.sqrt(self.key_dim)
        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(2)  # 调整mask维度
            attention_scores = attention_scores.masked_fill(mask == 0, -float('inf'))

        # 应用softmax获取注意力权重，后接dropout
        attention_weights = self.attention_dropout(F.softmax(attention_scores, dim=-1))

        # 计算输出并通过输出层和层归一化
        attention_output = torch.matmul(attention_weights, value_transformed).transpose(1, 2).contiguous().view(batch_size, -1, self.model_dim)
        output = self.output_dropout(self.output_linear(attention_output))
        output = self.normalization(residual + output)
        return output, attention_weights


class FeedForwardLayer(nn.Module):
    def __init__(self, model_dim, ff_dim=2048, dropout_rate=0.1):
        """初始化前馈层模块。

        参数:
            model_dim (int): 模型的维度。
            ff_dim (int): 前馈层的内部维度。
            dropout_rate (float): Dropout层的概率。
        """
        super(FeedForwardLayer, self).__init__()
        # 设定ff_dim默认值为2048
        self.first_linear = nn.Linear(model_dim, ff_dim)
        self.first_dropout = nn.Dropout(dropout_rate)
        self.second_linear = nn.Linear(ff_dim, model_dim)
        self.second_dropout = nn.Dropout(dropout_rate)
        self.normalization = nn.LayerNorm(model_dim)

    def forward(self, input_tensor):
        """前馈层的前向传播。

        参数:
            input_tensor (Tensor): 输入张量。

        返回:
            Tensor: 经过前馈层处理后的张量。
        """
        residual = input_tensor
        # 执行第一次线性变换后接ReLU激活和dropout
        output = self.first_dropout(F.relu(self.first_linear(input_tensor)))
        # 执行第二次线性变换后接ReLU激活和dropout
        output = self.second_dropout(F.relu(self.second_linear(output)))
        # 添加残差连接和层归一化后返回结果
        output = self.normalization(residual + output)
        return output


class BertModel(nn.Module):
    """基于BERT的语言模型，适用于多种NLP任务。

    包括词嵌入层、位置嵌入层、片段嵌入层和多个编码器层。
    """
    def __init__(self, n_layers, d_model, d_ff, n_heads,
                 max_seq_len, vocab_size, pad_id, dropout=0.1):
        super(BertModel, self).__init__()
        self.model_dim = d_model
        self.pad_token_id = pad_id
        self.token_embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        self.positional_encoding = TokenPositionalEncoding(d_model, max_seq_len)
        self.segment_embedding = nn.Embedding(2, d_model, padding_idx=0)
        self.embedding_dropout = nn.Dropout(dropout)
        self.encoder_layers = nn.ModuleList(
            [TransformerEncoderLayer(d_model, d_ff, n_heads, dropout) for _ in range(n_layers)])

    def forward(self, input_ids, token_type_ids, mask, return_attention=False):
        """模型前向传播，支持返回注意力权重。

        参数:
            input_ids (Tensor): 输入的词ID。
            token_type_ids (Tensor): 词的类型ID。
            mask (Tensor): 输入的掩码。
            return_attention (bool): 是否返回注意力权重。
        """
        output = self.token_embedding(input_ids) + self.positional_encoding(input_ids) + self.segment_embedding(token_type_ids)
        output = self.embedding_dropout(output)
        attention_weights = []
        for layer in self.encoder_layers:
            output, attention = layer(output, mask)
            if return_attention:
                attention_weights.append(attention)

        output = (output, attention_weights) if return_attention else output
        return output


class TransformerEncoderLayer(nn.Module):
    """单个Transformer编码器层，包括多头注意力和前馈网络。
    """

    def __init__(self, d_model, d_ff, n_heads, dropout):
        super(TransformerEncoderLayer, self).__init__()
        self.multi_head_attention = EnhancedMultiHeadAttention(n_heads, d_model, dropout)
        self.feed_forward_network = FeedForwardLayer(d_model, d_ff, dropout)

    def forward(self, inputs, mask):
        """编码器层的前向传播。

        参数:
            inputs (Tensor): 来自前一层的输入。
            mask (Tensor): 输入的掩码。
        """
        output, attention = self.multi_head_attention(inputs, inputs, inputs, mask)
        output = self.feed_forward_network(output)

        return output, attention


class selfBertForNER(nn.Module):
    """
    基于BERT模型的命名实体识别（NER）模块。

    此模型集成了BERT架构，专门为命名实体识别任务进行了优化。包括嵌入层、位置编码、编码器层和分类器。
    """

    def __init__(self, n_layers, d_model, d_ff, n_heads, max_seq_len, vocab_size, pad_id, dropout=0.1, num_labels=9):
        """
        初始化命名实体识别模型。

        参数:
            n_layers (int): 编码器层数。
            d_model (int): 嵌入和模型的维度。
            d_ff (int): 编码器层的前馈网络维度。
            n_heads (int): 多头注意力机制中的头数。
            max_seq_len (int): 输入序列的最大长度。
            vocab_size (int): 词汇表大小。
            pad_id (int): 填充词汇的索引。
            dropout (float): Dropout层的概率。
            num_labels (int): 预测标签的数量，对应于不同的实体类型。
        """
        super(selfBertForNER, self).__init__()
        self.d_model = d_model
        self.pad_id = pad_id
        self.num_labels = num_labels

        # 初始化词嵌入层，位置嵌入层和片段嵌入层。
        self.word_emb = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        self.pos_emb = TokenPositionalEncoding(d_model, max_seq_len)
        self.segment_emb = nn.Embedding(2, d_model, padding_idx=0)

        # 嵌入后应用的Dropout。
        self.dropout_emb = nn.Dropout(dropout)

        # 多个Transformer编码器层，构成模型的主体。
        self.layers = nn.ModuleList([TransformerEncoderLayer(d_model, d_ff, n_heads, dropout) for _ in range(n_layers)])

        # 输出前的额外Dropout。
        self.dropout = nn.Dropout(dropout)

        # 分类器，用于将编码器输出的每个令牌映射到标签空间。
        self.classifier = nn.Linear(d_model, num_labels)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, return_attn=False):
        """
        模型的前向传播。

        参数:
            input_ids (Tensor): 输入的词ID张量。
            token_type_ids (Tensor, 可选): 每个词元的段ID。
            attention_mask (Tensor, 可选): 用于掩蔽无关词元的掩码。
            return_attn (bool, 可选): 是否返回注意力权重。

        返回:
            如果 return_attn 为 True，则返回 (logits, attn_list)，否则只返回 logits。
        """
        # 应用词嵌入、位置嵌入、（如果提供）段嵌入，然后应用Dropout。
        out = self.word_emb(input_ids) + self.pos_emb(input_ids)
        if token_type_ids is not None:
            out += self.segment_emb(token_type_ids)
        out = self.dropout_emb(out)

        # 遍历所有编码器层，应用注意力机制和前馈网络。
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

        # 应用分类器层，并调整输出形状以适配序列长度和批量大小。
        # 调整形状: [batch_size, seq_length, d_model] -> [batch_size * seq_length, d_model]
        batch_size, seq_length, d_model = out.shape
        out = out.view(batch_size * seq_length, d_model)

        # 打印调整后的 out 形状
        # print("Sequence output shape after view:", out.shape)

        logits = self.classifier(out)

        # 将形状恢复: [batch_size * seq_length, num_labels] -> [batch_size, seq_length, num_labels]
        logits = logits.view(batch_size, seq_length, -1)

        return (logits, attn_list) if return_attn else logits
