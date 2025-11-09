import torch
import torch.nn as nn
import math

# ======== Scaled Dot-Product Attention ========
class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, Q, K, V, mask=None):
        d_k = Q.size(-1)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        attn = torch.softmax(scores, dim=-1)
        output = torch.matmul(attn, V)
        return output, attn


# ======== Multi-Head Attention ========
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_k = d_model // num_heads
        self.num_heads = num_heads

        self.W_Q = nn.Linear(d_model, d_model)
        self.W_K = nn.Linear(d_model, d_model)
        self.W_V = nn.Linear(d_model, d_model)
        self.fc = nn.Linear(d_model, d_model)

        self.attn = ScaledDotProductAttention()

    def forward(self, Q, K, V, mask=None):
        batch_size = Q.size(0)

        Q = self.W_Q(Q).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_K(K).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_V(V).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

        # 不要再在这里 unsqueeze 了！
        # if mask is not None:
        #     mask = mask.unsqueeze(1)

        out, attn = self.attn(Q, K, V, mask)
        out = out.transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * self.d_k)
        out = self.fc(out)
        return out


# ======== Feed Forward ========
class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.linear2(self.relu(self.linear1(x)))


# ======== Positional Encoding ========
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return x


# ======== Encoder Layer ========
class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.ff = PositionwiseFeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src, src_mask=None):
        _src = src
        src = self.norm1(src + self.dropout(self.self_attn(src, src, src, src_mask)))
        src = self.norm2(src + self.dropout(self.ff(src)))
        return src


# ======== Decoder Layer ========
class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.cross_attn = MultiHeadAttention(d_model, num_heads)
        self.ff = PositionwiseFeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None):
        tgt = self.norm1(tgt + self.dropout(self.self_attn(tgt, tgt, tgt, tgt_mask)))
        tgt = self.norm2(tgt + self.dropout(self.cross_attn(tgt, memory, memory, memory_mask)))
        tgt = self.norm3(tgt + self.dropout(self.ff(tgt)))
        return tgt


# ======== Transformer ========
class TransformerModel(nn.Module):
    def __init__(self, vocab_size, d_model=256, num_heads=8, num_layers=3, d_ff=512, num_classes=4, dropout=0.1, max_len=50):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_len)
        self.encoder_layers = nn.ModuleList([
            EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)
        ])
        self.decoder_layers = nn.ModuleList([
            DecoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)
        ])
        self.fc_out = nn.Linear(d_model, num_classes)

    def generate_square_subsequent_mask(self, sz):
        mask = torch.tril(torch.ones(sz, sz)).unsqueeze(0)
        return mask

    def forward(self, src, tgt):
        src_mask = (src != 0).unsqueeze(1).unsqueeze(2)  # padding mask
        tgt_mask = self.generate_square_subsequent_mask(tgt.size(1)).to(tgt.device)

        src_embed = self.pos_encoding(self.embedding(src))
        tgt_embed = self.pos_encoding(self.embedding(tgt))

        memory = src_embed
        for layer in self.encoder_layers:
            memory = layer(memory, src_mask)

        out = tgt_embed
        for layer in self.decoder_layers:
            out = layer(out, memory, tgt_mask, src_mask)

        out = self.fc_out(out.mean(dim=1))  # 分类任务用平均池化
        return out
