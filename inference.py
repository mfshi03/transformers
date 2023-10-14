import torch
import torch.nn as nn
import torch.optim as optim
from model.Transformer import Transformer

def generate(model, src, max_length=100):
    model.eval()
    with torch.no_grad():
        src = torch.tensor(encode(src), dtype=torch.long).unsqueeze(0).to(device)
        tgt = torch.tensor([1]).to(device) 

        for _ in range(max_length):
            output = model(src, tgt)
            next_word = output.argmax(2)[:, -1].item()
            tgt = torch.cat([tgt, torch.tensor([next_word], dtype=torch.long).to(device)], dim=1)
            
            if next_word == 2:
                break

        return decode(tgt_sequence[0].cpu().numpy())

with open('data/input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# Process the vocab
chars = sorted(list(set(text)))
vocab_size = len(chars)
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s] # encoder: take a string, outputs list of integers
decode = lambda l: ''.join([itos[i] for i in l]) # decoder: take a list of integers, outputs string


batch_size = 64 
block_size = 256 
device = 'cuda' if torch.cuda.is_available() else 'cpu'
src_vocab_size = 5000
tgt_vocab_size = 5000
dim = 512
num_heads = 8
num_layers = 6
ff_dim  = 2048
dropout = 0.1

transformer = Transformer(src_vocab_size, tgt_vocab_size, dim, num_heads, num_layers, ff_dim , block_size, dropout)

transformer.load_state_dict(torch.load(f"weights/transformer_epoch_{epoch+1}.pth"))
transformer.to(device)

generated_sequence = generate(transformer, "What is an apple?")
print(generated_sequence)