#from datasets import load_dataset
#dataset = load_dataset("iwslt2017", "iwslt2017-de-en", cache_dir="D:/pycode/transformer/data", trust_remote_code=True)

"""
Transformer模型（可用于消融实验）
支持禁用位置编码与残差连接，用于对比实验。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# ======== 基础模块 ========

class ResidualConnection(nn.Module):
    """残差连接 + LayerNorm"""
    def __init__(self, d_model, dropout=0.1):
        super().__init__()
        self.layer_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.layer_norm(x)))


class PositionwiseFFN(nn.Module):
    """位置前馈网络"""
    def __init__(self, d_model, d_ff, dropout=0.1, activation="relu"):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ReLU() if activation == "relu" else nn.GELU()

    def forward(self, x):
        return self.linear2(self.dropout(self.activation(self.linear1(x))))


class MultiHeadAttention(nn.Module):
    """多头自注意力"""
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

    def forward(self, x, mask=None):
        B, L, _ = x.size()
        Q = self.W_q(x).view(B, L, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(x).view(B, L, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(x).view(B, L, self.num_heads, self.d_k).transpose(1, 2)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        attn = torch.softmax(scores, dim=-1)
        context = torch.matmul(attn, V).transpose(1, 2).contiguous().view(B, L, self.d_model)
        return self.W_o(context), attn


class MaskedMultiHeadAttention(nn.Module):
    """带因果掩码的多头注意力（用于Decoder）"""
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.attn = MultiHeadAttention(d_model, num_heads)

    def forward(self, x, mask=None):
        seq_len = x.size(1)
        causal_mask = torch.tril(torch.ones(seq_len, seq_len, device=x.device)).unsqueeze(0).unsqueeze(0)
        if mask is not None:
            mask = mask * causal_mask
        else:
            mask = causal_mask
        return self.attn(x, mask)


class CrossAttention(nn.Module):
    """交叉注意力（Decoder → Encoder输出）"""
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

    def forward(self, decoder_x, encoder_output, mask=None):
        B, L_tgt, _ = decoder_x.size()
        L_src = encoder_output.size(1)
        Q = self.W_q(decoder_x).view(B, L_tgt, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(encoder_output).view(B, L_src, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(encoder_output).view(B, L_src, self.num_heads, self.d_k).transpose(1, 2)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        attn = torch.softmax(scores, dim=-1)
        context = torch.matmul(attn, V).transpose(1, 2).contiguous().view(B, L_tgt, self.d_model)
        return self.W_o(context), attn


class SinusoidalPositionalEncoding(nn.Module):
    """正弦位置编码"""
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


# ======== 编码器与解码器 ========

class TransformerEncoderLayer(nn.Module):
    """编码器层（支持禁用残差连接）"""
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1, disable_residual=False):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.ffn = PositionwiseFFN(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.disable_residual = disable_residual

    def forward(self, x, mask=None):
        attn_out, _ = self.self_attn(x, mask)
        if not self.disable_residual:
            x = x + self.dropout(attn_out)
        else:
            x = self.dropout(attn_out)
        x = self.norm1(x)
        ffn_out = self.ffn(x)
        if not self.disable_residual:
            x = x + self.dropout(ffn_out)
        else:
            x = self.dropout(ffn_out)
        x = self.norm2(x)
        return x


class TransformerEncoder(nn.Module):
    """完整编码器（支持禁用位置编码）"""
    def __init__(self, vocab_size, d_model, num_heads, d_ff, num_layers,
                 max_len=5000, dropout=0.1, disable_pos_encoding=False, disable_residual=False):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = SinusoidalPositionalEncoding(d_model, max_len)
        self.dropout = nn.Dropout(dropout)
        self.disable_pos_encoding = disable_pos_encoding

        self.layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, num_heads, d_ff, dropout, disable_residual)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, mask=None):
        x = self.token_embedding(x)
        if not self.disable_pos_encoding:
            x = self.positional_encoding(x)
        x = self.dropout(x)
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1, disable_residual=False):
        super().__init__()
        self.self_attn = MaskedMultiHeadAttention(d_model, num_heads)
        self.cross_attn = CrossAttention(d_model, num_heads)
        self.ffn = PositionwiseFFN(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.disable_residual = disable_residual

    def forward(self, x, encoder_output, self_mask=None, cross_mask=None):
        attn_out, _ = self.self_attn(x, self_mask)
        if not self.disable_residual:
            x = x + self.dropout(attn_out)
        else:
            x = self.dropout(attn_out)
        x = self.norm1(x)
        cross_out, _ = self.cross_attn(x, encoder_output, cross_mask)
        if not self.disable_residual:
            x = x + self.dropout(cross_out)
        else:
            x = self.dropout(cross_out)
        x = self.norm2(x)
        ffn_out = self.ffn(x)
        if not self.disable_residual:
            x = x + self.dropout(ffn_out)
        else:
            x = self.dropout(ffn_out)
        x = self.norm3(x)
        return x


class TransformerDecoder(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, d_ff, num_layers,
                 max_len=5000, dropout=0.1, disable_pos_encoding=False, disable_residual=False):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = SinusoidalPositionalEncoding(d_model, max_len)
        self.dropout = nn.Dropout(dropout)
        self.disable_pos_encoding = disable_pos_encoding
        self.layers = nn.ModuleList([
            TransformerDecoderLayer(d_model, num_heads, d_ff, dropout, disable_residual)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.output_layer = nn.Linear(d_model, vocab_size)

    def forward(self, x, encoder_output, self_mask=None, cross_mask=None):
        x = self.token_embedding(x)
        if not self.disable_pos_encoding:
            x = self.positional_encoding(x)
        x = self.dropout(x)
        for layer in self.layers:
            x = layer(x, encoder_output, self_mask, cross_mask)
        x = self.norm(x)
        return self.output_layer(x)


# ======== 主Transformer类 ========

class Transformer(nn.Module):
    """完整Transformer，支持消融选项"""
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model, num_heads, d_ff,
                 num_encoder_layers, num_decoder_layers, max_len=5000, dropout=0.1,
                 disable_pos_encoding=False, disable_residual=False):
        super().__init__()
        self.encoder = TransformerEncoder(src_vocab_size, d_model, num_heads, d_ff,
                                          num_encoder_layers, max_len, dropout,
                                          disable_pos_encoding, disable_residual)
        self.decoder = TransformerDecoder(tgt_vocab_size, d_model, num_heads, d_ff,
                                          num_decoder_layers, max_len, dropout,
                                          disable_pos_encoding, disable_residual)

    def forward(self, src, tgt, src_mask=None, tgt_mask=None, memory_mask=None):
        memory = self.encoder(src, src_mask)
        output = self.decoder(tgt, memory, tgt_mask, memory_mask)
        return output

    def get_num_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
