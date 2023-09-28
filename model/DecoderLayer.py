from model.MultiHeadedAttention import MultiHeadAttention
from model.PositionalFeedForward import PositionalFeedForward

class DecoderLayer(nn.Module):
    '''
    We need 3 normalization layers to normalize via a residual connection
    - self-attention 
    - cross-attention 
    - feed forward network
    '''
    def __init__(self, dim, n_heads, ff_dim, dropout):
        super(DecoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(dim, n_heads)
        self.cross_attn = MultiHeadAttention(dim, n_heads)
        self.ff_net = PositionalFeedForward(dim, ff_dim)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)
        self.dropout= nn.dropout(dropout)
    
    def forward(self, resid, encoding_out, source_mask, target_mask):
        attn_output = self.self_attn(resid, resid, resid,target_mask)
        resid = self.normal(resid + self.dropout(attn_output))
        attn_output = self.cross_attn(x, encoding_out, encoding_out, source_mask)
        resid = self.norm2(resid + self.dropout(attn_output))
        ff_output = self.feed_forward(x)
        resid = self.norm3(resid + self.dropout(ff_output))
        return resid