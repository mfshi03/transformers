import torch
import torch.nn

'''
The reason why positional encodings work is really elegant. 
The sinuisoidal function used is expressive enough to represent these encodings
because the distances are consistent across time steps and because values are bounded.
The distribution is the same as if you recorded the bits of increasing numbers.

More info here: 
https://kazemnejad.com/blog/transformer_architecture_positional_encoding/)
'''

class PositionalEncoding(nn.Module):
    def __init__(self, dim, max_seq_len):
        super(PositionalEncoding, self).__init__()
        
        pos_encoding = torch.zeros(max_seq_len, dim)
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2).float() * -(math.log(10000.0) / dim))
        
        pos_encoding[:, 0::2] = torch.sin(position * div_term)
        pos_encoding[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pos_encoding.unsqueeze(0))
        
    def forward(self, x):
        return x + self.pos_encoding[:, :x.size(1)]