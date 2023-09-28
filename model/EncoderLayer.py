import torch
import torch.nn as nn
from model.MultiHeadedAttention import MultiHeadAttention
from model.PositionalFeedForward import PositionalFeedForward

class EncoderLayer(nn.Module):
    def __init__(self, dim, n_heads, ff_dim, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(dim, n_heads)
        self.ff_net = PositionalFeedForward(dim, ff_dim)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, resid, mask):
        attn_output = self.self_attn(resid, resid, resid, mask)
        resid = self.norm1(resid + self.dropout(attn_output))
        ff_output = self.ff_net(resid)
        resid = self.norm2(resid + self.dropout(ff_output))
        return resid