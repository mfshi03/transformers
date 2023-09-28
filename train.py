import torch
import torch.nn as nn
import torch.optim as optim
from model.Transformer import Transformer

src_vocab_size = 5000
tgt_vocab_size = 5000
dim = 512
num_heads = 8
num_layers = 6
ff_dim  = 2048
max_seq_len = 100
dropout = 0.1

transformer = Transformer(src_vocab_size, tgt_vocab_size, dim, num_heads, num_layers, ff_dim , max_seq_len, dropout)

# Generate random sample data
src_data = torch.randint(1, src_vocab_size, (64, max_seq_len))  # (batch_size, seq_length)
tgt_data = torch.randint(1, tgt_vocab_size, (64, max_seq_len))  # (batch_size, seq_length)
criterion = nn.CrossEntropyLoss(ignore_index=0)
optimizer = optim.Adam(transformer.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)

transformer.train()

for epoch in range(100):
    optimizer.zero_grad()
    output = transformer(src_data, tgt_data[:, :-1])
    loss = criterion(output.contiguous().view(-1, tgt_vocab_size), tgt_data[:, 1:].contiguous().view(-1))
    loss.backward()
    optimizer.step()
    print(f"Epoch: {epoch+1}, Loss: {loss.item()}")