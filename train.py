import torch
import torch.nn as nn
import torch.optim as optim
from model.Transformer import Transformer

batch_size = 64 # how many independent sequences will we process in parallel?
block_size = 256 # what is the maximum context length for predictions?
device = 'cuda' if torch.cuda.is_available() else 'cpu'
src_vocab_size = 5000
tgt_vocab_size = 5000
dim = 512
num_heads = 8
num_layers = 6
ff_dim  = 2048
dropout = 0.1

transformer = Transformer(src_vocab_size, tgt_vocab_size, dim, num_heads, num_layers, ff_dim , block_size, dropout)

with open('data/input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# Process the vocab
chars = sorted(list(set(text)))
vocab_size = len(chars)
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s] # encoder: take a string, outputs list of integers
decode = lambda l: ''.join([itos[i] for i in l]) # decoder: take a list of integers, outputs string


data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9*len(data)) # first 90% will be train, rest val
train_data = data[:n]
val_data = data[n:]

def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y


X, Y = get_batch('train')

# Generate random sample data
src_data = X[:src_vocab_size] # torch.randint(1, src_vocab_size, (64, block_size))  # (batch_size, seq_length)
tgt_data = Y[:tgt_vocab_size] # torch.randint(1, tgt_vocab_size, (64, block_size))  # (batch_size, seq_length)
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