"""
Transformer模型实现 - 作业要求：实现multi-head self-attention, position-wise FFN,
残差+LayerNorm, 位置编码等核心模块
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class ResidualConnection(nn.Module):
    """残差连接 + LayerNorm - 作业要求模块"""

    def __init__(self, d_model, dropout=0.1):
        super(ResidualConnection, self).__init__()
        self.layer_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.layer_norm(x)))


class PositionwiseFFN(nn.Module):
    """位置前馈网络 - 作业要求模块：两层MLP独立应用于每个token"""

    def __init__(self, d_model, d_ff, dropout=0.1, activation="relu"):
        super(PositionwiseFFN, self).__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

        if activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "gelu":
            self.activation = nn.GELU()
        else:
            raise ValueError(f"不支持的激活函数: {activation}")

    def forward(self, x):
        x = self.linear1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x


class MultiHeadAttention(nn.Module):
    """多头自注意力 - 作业要求模块：实现scaled dot-product attention"""

    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0, "d_model必须能被num_heads整除"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        # 线性变换矩阵
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

    def forward(self, x, mask=None):
        batch_size, seq_len, d_model = x.size()

        # 线性变换得到Q, K, V
        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x)

        # 重塑为多头形式
        Q = Q.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)

        # 计算注意力分数：Attention(Q,K,V)=softmax(QK^T/sqrt(d_k))V
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)

        # 应用mask（padding mask）
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        # Softmax得到注意力权重
        attn_weights = torch.softmax(scores, dim=-1)

        # 应用注意力权重到V
        context = torch.matmul(attn_weights, V)

        # 重塑回原始形状
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)
        output = self.W_o(context)

        return output, attn_weights


class MaskedMultiHeadAttention(nn.Module):
    """带因果掩码的多头注意力 - 用于Decoder（防止看到未来信息）"""

    def __init__(self, d_model, num_heads):
        super(MaskedMultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

    def forward(self, x, mask=None):
        batch_size, seq_len, d_model = x.size()

        # 线性变换
        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x)

        # 重塑为多头
        Q = Q.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)

        # 计算注意力分数
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)

        # 创建因果掩码（future mask）- 只允许关注当前及之前的位置
        causal_mask = torch.tril(torch.ones(seq_len, seq_len, device=x.device))
        causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)
        scores = scores.masked_fill(causal_mask == 0, -1e9)

        # 应用额外mask
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        # Softmax和加权平均
        attn_weights = torch.softmax(scores, dim=-1)
        context = torch.matmul(attn_weights, V)

        # 重塑回原始形状
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)
        output = self.W_o(context)

        return output, attn_weights


class CrossAttention(nn.Module):
    """Encoder-Decoder交叉注意力 - 连接编码器和解码器"""

    def __init__(self, d_model, num_heads):
        super(CrossAttention, self).__init__()
        assert d_model % num_heads == 0

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        # Query来自Decoder，Key和Value来自Encoder
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

    def forward(self, decoder_x, encoder_output, mask=None):
        batch_size, tgt_seq_len, d_model = decoder_x.size()
        src_seq_len = encoder_output.size(1)

        # Query来自Decoder
        Q = self.W_q(decoder_x)
        # Key和Value来自Encoder
        K = self.W_k(encoder_output)
        V = self.W_v(encoder_output)

        # 重塑为多头
        Q = Q.view(batch_size, tgt_seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, src_seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, src_seq_len, self.num_heads, self.d_k).transpose(1, 2)

        # 计算注意力分数
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)

        # 应用mask
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        # Softmax和加权平均
        attn_weights = torch.softmax(scores, dim=-1)
        context = torch.matmul(attn_weights, V)

        # 重塑和输出
        context = context.transpose(1, 2).contiguous().view(batch_size, tgt_seq_len, d_model)
        output = self.W_o(context)

        return output, attn_weights


class SinusoidalPositionalEncoding(nn.Module):
    """正弦位置编码 - 作业要求模块：提供序列位置信息"""

    def __init__(self, d_model, max_len=5000):
        super(SinusoidalPositionalEncoding, self).__init__()

        # 创建位置编码矩阵
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)

        # 频率项计算
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() *
            (-math.log(10000.0) / d_model)
        )

        # 正弦余弦交替编码
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class TransformerEncoderLayer(nn.Module):
    """Transformer编码器层 - 包含自注意力和前馈网络"""

    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super(TransformerEncoderLayer, self).__init__()

        # 自注意力子层
        self.self_attention = MultiHeadAttention(d_model, num_heads)
        self.attention_residual = ResidualConnection(d_model, dropout)

        # 前馈网络子层
        self.ffn = PositionwiseFFN(d_model, d_ff, dropout, activation="relu")
        self.ffn_residual = ResidualConnection(d_model, dropout)

    def forward(self, x, mask=None):
        # 自注意力 + 残差连接
        x = self.attention_residual(x, lambda x: self.self_attention(x, mask)[0])
        # 前馈网络 + 残差连接
        x = self.ffn_residual(x, self.ffn)
        return x


class TransformerEncoder(nn.Module):
    """完整的Transformer编码器 - 作业基础要求"""

    def __init__(self, vocab_size, d_model, num_heads, d_ff, num_layers,
                 max_len=5000, dropout=0.1):
        super(TransformerEncoder, self).__init__()

        # 词嵌入层
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        # 位置编码
        self.positional_encoding = SinusoidalPositionalEncoding(d_model, max_len)
        self.dropout = nn.Dropout(dropout)

        # 编码器层堆叠
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])

        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, x, mask=None):
        # 词嵌入 + 位置编码
        x = self.token_embedding(x)
        x = self.positional_encoding(x)
        x = self.dropout(x)

        # 通过多个编码器层
        for layer in self.layers:
            x = layer(x, mask)

        x = self.layer_norm(x)
        return x


class TransformerDecoderLayer(nn.Module):
    """Transformer解码器层 - 包含掩码自注意力、交叉注意力和前馈网络"""

    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super(TransformerDecoderLayer, self).__init__()

        # 掩码自注意力
        self.masked_self_attention = MaskedMultiHeadAttention(d_model, num_heads)
        self.self_attention_residual = ResidualConnection(d_model, dropout)

        # Encoder-Decoder交叉注意力
        self.cross_attention = CrossAttention(d_model, num_heads)
        self.cross_attention_residual = ResidualConnection(d_model, dropout)

        # 前馈网络
        self.ffn = PositionwiseFFN(d_model, d_ff, dropout, activation="relu")
        self.ffn_residual = ResidualConnection(d_model, dropout)

    def forward(self, x, encoder_output, self_mask=None, cross_mask=None):
        # 掩码自注意力
        x = self.self_attention_residual(
            x, lambda x: self.masked_self_attention(x, self_mask)[0])
        # 交叉注意力
        x = self.cross_attention_residual(
            x, lambda x: self.cross_attention(x, encoder_output, cross_mask)[0])
        # 前馈网络
        x = self.ffn_residual(x, self.ffn)
        return x


class TransformerDecoder(nn.Module):
    """完整的Transformer解码器 - 作业进阶要求"""

    def __init__(self, vocab_size, d_model, num_heads, d_ff, num_layers,
                 max_len=5000, dropout=0.1):
        super(TransformerDecoder, self).__init__()

        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = SinusoidalPositionalEncoding(d_model, max_len)
        self.dropout = nn.Dropout(dropout)

        # 解码器层堆叠
        self.layers = nn.ModuleList([
            TransformerDecoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])

        self.layer_norm = nn.LayerNorm(d_model)
        # 输出层映射回词汇表
        self.output_layer = nn.Linear(d_model, vocab_size)

    def forward(self, x, encoder_output, self_mask=None, cross_mask=None):
        # 词嵌入 + 位置编码
        x = self.token_embedding(x)
        x = self.positional_encoding(x)
        x = self.dropout(x)

        # 通过多个解码器层
        for layer in self.layers:
            x = layer(x, encoder_output, self_mask, cross_mask)

        x = self.layer_norm(x)
        # 输出logits
        logits = self.output_layer(x)
        return logits


class Transformer(nn.Module):
    """完整的Encoder-Decoder Transformer - 作业最高分要求"""

    def __init__(self, src_vocab_size, tgt_vocab_size, d_model, num_heads, d_ff,
                 num_encoder_layers, num_decoder_layers, max_len=5000, dropout=0.1):
        super(Transformer, self).__init__()

        # 编码器
        self.encoder = TransformerEncoder(
            src_vocab_size, d_model, num_heads, d_ff, num_encoder_layers, max_len, dropout
        )

        # 解码器
        self.decoder = TransformerDecoder(
            tgt_vocab_size, d_model, num_heads, d_ff, num_decoder_layers, max_len, dropout
        )

    def forward(self, src, tgt, src_mask=None, tgt_mask=None, memory_mask=None):
        # 编码器处理源序列
        encoder_output = self.encoder(src, src_mask)
        # 解码器生成目标序列
        decoder_output = self.decoder(tgt, encoder_output, tgt_mask, memory_mask)
        return decoder_output

    def get_num_params(self):
        """返回模型参数数量 - 用于统计"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)