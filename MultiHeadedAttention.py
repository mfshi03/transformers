import torch
import torch.nn
import torch.optim as optim 
import torch.utils.data as data 
import math
import copy

class MultiHeadAttention(nn.Module):
    def __init__(self, dim, n_heads):
        super(MultiHeadAttention, self).__init__()

        assert dim % n_heads == 0

        self.dim = dim 
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.query_w = nn.Linear(dim, dim)
        self.key_w = nn.Linear(dim, dim)
        self.val_w = nn.Linear(dim, dim)
        self.output_w = nn.Linear(dim, dim)
        
    # Used to compute the concatenated attention heads computed from the query, key, value matrices 
    def scaled_dotp_attention(self, Q, K, V, mask = none):
        scores = torch.matmul(Q, K.transpose(-2, -1) / math.sqrt(self.d_k))
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        probs = torch.softmax(scores, dim = -1)
        output = torch.matmul(probs, V)
        return output
    
    def split_heads(self, x):
        batch_size, seq_len, dim = x.size()
        return x.view(batch_size, seq_len, self.n_heads, self.head_dim)
    
    def combine_heads(self, x):
        batch_size, _, seq_len, head_dim = x.size()
        return x.transpose(1,2).contiguous.view(batch_size, seq_len, self.model_dim)

    def forward(self, Q, K, V, mask = none):
        Q = self.split_heads(self.query_w(Q))
        K = self.split_heads(self.key_w(K))
        V = self.split_heads(self.key_v(V))
        output = self.scaled_dotp_attention(Q,K, V, mask)
        output = self.output_w(self.combine_heads(output))
        return output


        