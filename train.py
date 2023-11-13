import time
import pickle
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from model.Transformer import Transformer


stoi_file = 'stoi.pkl'
itos_file = 'itos.pkl'

def load_mapping(file_path):
    if os.path.exists(file_path):
        with open(file_path, 'rb') as f:
            return pickle.load(f)
    else:
        with open(file_path, 'wb') as f:
            pickle.dump(stoi, f)
    
if torch.cuda.is_available():
    torch.set_default_tensor_type(torch.cuda.FloatTensor)
    #dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32' or 'bfloat16' or 'float16'
    #ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
    #torch.amp.autocast(device_type='cuda', dtype=ptdtype)


batch_size = 64 # how many independent sequences will we process in parallel?
block_size = 256 # what is the maximum context length for predictions?
device = 'cuda' if torch.cuda.is_available() else 'cpu'
src_vocab_size = 10000
tgt_vocab_size = 10000
dim = 512
num_heads = 8
num_layers = 6
ff_dim  = 2048
dropout = 0.1

transformer = Transformer(src_vocab_size, tgt_vocab_size, dim, num_heads, num_layers, ff_dim , block_size, dropout)

with open('data/input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

try:
    with open('stoi.pickle', 'rb') as f:
        stoi = pickle.load(f)
    with open('itos.pickle', 'rb') as f:
        itos = pickle.load(f)
    vocab_size = len(stoi)
except FileNotFoundError:
    # Read the text file
    with open('data/input.txt', 'r', encoding='utf-8') as f:
        text = f.read()

    # Process the vocab
    chars = sorted(list(set(text)))
    vocab_size = len(chars)
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}

    # Save the dictionaries to pickle files
    with open('stoi.pickle', 'wb') as f:
        pickle.dump(stoi, f)
    with open('itos.pickle', 'wb') as f:
        pickle.dump(itos, f)

encode = lambda s: [stoi[c] for c in s]  # encoder: take a string, outputs a list of integers
decode = lambda l: ''.join([itos[i] for i in l])  # decoder: take a list of integers, outputs a string

start_time = time.time()
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

def generate(model, initial_seq, max_length=100):
    model.eval()
    with torch.no_grad():
        input_seq = torch.tensor(encode(initial_seq)).unsqueeze(0).to(device)
        generated = input_seq.tolist()

        for _ in range(max_length - len(initial_seq)):
            output = model(input_seq, input_seq)
            next_token = output.argmax(dim=-1)[:, -1].item()
            generated[0].append(next_token)
            input_seq = torch.tensor(generated).to(device)

        generated_seq = decode(generated[0])
        
    return generated_seq


def generate_topk(model, initial_seq, max_length=100, k=40, temperature=0.15):
    model.eval()
    with torch.no_grad():
        input_seq = torch.tensor(encode(initial_seq)).unsqueeze(0).to(device)
        generated = input_seq.tolist()

        for _ in range(max_length - len(initial_seq)):
            output = model(input_seq, input_seq) # [batch_size, seq_len, vocab_size]
            # Apply temperature scaling
            scaled_logits = output[0, -1, :] / temperature
            # Get the probabilities for the last token in the sequence
            probs = F.softmax(scaled_logits, dim=-1)
            # Take the top-k tokens only
            top_k_probs, top_k_idx = torch.topk(probs, k=k, dim=-1)
            # Sample from the top-k tokens
            next_token = torch.multinomial(top_k_probs, num_samples=1).item()
            next_token = top_k_idx[next_token].item()
            generated[0].append(next_token)
            input_seq = torch.tensor(generated).to(device)

        generated_seq = decode(generated[0])
        
    return generated_seq

'''
X, Y = get_batch('train')

# Generate random sample data
src_data = X[:src_vocab_size] # torch.randint(1, src_vocab_size, (64, block_size))  # (batch_size, seq_length)
tgt_data = Y[:tgt_vocab_size] # torch.randint(1, tgt_vocab_size, (64, block_size))  # (batch_size, seq_length)
criterion = nn.CrossEntropyLoss(ignore_index=0)
optimizer = optim.Adam(transformer.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)

transformer.load_state_dict(torch.load(f"weights/transformer_epoch_1000.pth"))
for epoch in range(1000):
    transformer.train()
    optimizer.zero_grad()
    output = transformer(src_data, tgt_data[:, :-1])
    loss = criterion(output.contiguous().view(-1, tgt_vocab_size), tgt_data[:, 1:].contiguous().view(-1))
    loss.backward()
    optimizer.step()
    #torch.save(transformer.state_dict(), f"weights/transformer_epoch_{epoch+1}.pth")
    if epoch % 100 == 0:
        generated_sequence = generate_topk(transformer, "You are all resolved rather to die than to famish?")
        print(generated_sequence)
        torch.save(transformer.state_dict(), f"weights/transformer_epoch_mod.pth")

    print(f"Epoch: {epoch+1}, Loss: {loss.item()}")
'''
#torch.save(transformer.state_dict(), f"weights/transformer_epoch_2000.pth")
transformer.load_state_dict(torch.load(f"weights/transformer_epoch_mod.pth"))
transformer.to(device)
input_text = "MENENIUS:"
generated_sequence = generate_topk(transformer, input_text, max_length=100)
end_time = time.time()
print(input_text)
print("\033[33m"  + generated_sequence.replace(input_text,"")  + "\033[0m")
print("Inference speed: ", (end_time - start_time), " seconds")
