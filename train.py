import time
import pickle
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from model.Transformer import Transformer
from bleu import idx_to_word, get_bleu


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

epochs = 2000
batch_size = 32  # sequences we can process in parallel
block_size = 64 # maximum context length for predictions
src_vocab_size = 10000
tgt_vocab_size = 10000
dim = 512 # embedding dimension
num_heads = 8
num_layers = 6
num_batches = 100
ff_dim  = 2048 # feed-forward dimensions
dropout = 0.1

device = 'cuda' if torch.cuda.is_available() else 'cpu'

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

data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.2*len(data)) # first 90% will be train, rest val
train_data = data[:n]
val_data = data[n:]

def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

def batch_to_strings(batch):
    batch = batch.detach().cpu().numpy()
    strings = []
    for sample in batch:
        string = decode(sample) 
        strings.append(string)
        break
        
    return strings

    
'''
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
            if input_seq.size(1) > block_size:
                break

        generated_seq = decode(generated[0])
        
    return generated_seq
'''
def generate(model, initial_seq, max_length=100):
    model.eval()
    with torch.no_grad():
        input_seq = torch.tensor(encode(initial_seq)).unsqueeze(0).to(device)
        next_seq = input_seq
        generated = input_seq.tolist()

        for _ in range(max_length - len(initial_seq)):
            if len(next_seq) > block_size:
                break
            output = model(input_seq, input_seq)
            next_token = output.argmax(dim=-1)[:, -1].item()
            generated[0].append(next_token)
            next_seq = torch.tensor(generated).to(device)
            
        generated_seq = decode(generated[0])
        
    return generated_seq


'''
def generate(model, initial_seq, max_length=100):
    model.eval()
    with torch.no_grad():
        seq_size = len(initial_seq)
        for _ in range(max_length - seq_size):
            data = torch.tensor(encode(initial_seq), dtype=torch.long)
            x, y = torch.stack([data[0: block_size]]), torch.stack([data[1: block_size + 1]])
            #print(batch_to_strings(x))
            #xxprint(batch_to_strings(y))
            output = model(x, x)
            output_words = output[0].max(dim=1)[1]
            output_words = idx_to_word(output_words, itos)
            #print("Tensor,", output_words)
            print(initial_seq) 
            if len(initial_seq) > block_size:
                break
            initial_seq += output_words

    
        
    return output_words
'''
# helps estimate an arbitrarily accurate loss over either split using many batches
@torch.no_grad()
def estimate_loss(model:Transformer, eval_iters):
    out = {}
    model.eval()
    for split in ['train']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

def repetition_loss(logits, target):
    # Get previous target tokens 
    prev_tokens = target[:, :-1]
    prev_tokens = prev_tokens.unsqueeze(-1).expand(-1, -1, logits.shape[-1])
    repeat_logits = (logits == prev_tokens).float()
    # Apply softmax to get probabilities
    repeat_probs = F.softmax(repeat_logits, dim=1)
    # Calculate CE loss to maximize probability of not repeating
    return F.cross_entropy(repeat_probs, torch.zeros_like(repeat_probs))

def evaluate(model, criterion, itos, ind, batch_size=batch_size):
    model.eval()
    epoch_loss = 0
    batch_bleu = []
    with torch.no_grad():
        for i in range(100):
            src, trg = get_batch('val')  #Assuming the validation set is handled similarly to the training set
            output = model(src, trg)
            #loss = criterion(output.contiguous().view(-1, tgt_vocab_size), trg[:, 1:].contiguous().view(-1))
            #epoch_loss += loss.item()

            total_bleu = []
            for j in range(batch_size):
                trg_words = idx_to_word(trg[j], itos)
                output_words = output[j].max(dim=1)[1]
                output_words = idx_to_word(output_words, itos)
                bleu = get_bleu(hypotheses=output_words.split(), reference=trg_words.split()) 
                total_bleu.append(bleu)

            total_bleu = sum(total_bleu) / len(total_bleu) if total_bleu else 0
            print(f"Validation batch: {(ind*100) + i}, BLEU Score: {total_bleu}") #{loss.item()}")
            with open("eval2_bsize64_3.txt", "a") as f:
                f.write(f"Batch: {i}, BLEU Score: {total_bleu}\n") #{loss.item()}
            batch_bleu.append(total_bleu)

    batch_bleu = sum(batch_bleu) / len(batch_bleu) if batch_bleu else 0
    return epoch_loss / (len(val_data) // batch_size), batch_bleu


def generate_topk(model, initial_seq, max_length=100, k=5, temperature=1.0):
    model.eval()
    with torch.no_grad():
        input_seq = torch.tensor(encode(initial_seq)).unsqueeze(0).to(device)
        generated = input_seq.tolist()

        for _ in range(max_length - len(initial_seq)):
            output = model(input_seq, input_seq)               # [batch_size, seq_len, vocab_size]
            scaled_logits = output[0, -1, :] / temperature     # Apply temperature scaling
            probs = F.softmax(scaled_logits, dim=-1)  # Get the probabilities for the last token in the sequence
            top_k_probs, top_k_idx = torch.topk(probs, k=k, dim=-1) # Select top-k
            next_token = torch.multinomial(top_k_probs, num_samples=1).item() # Sample from the top-k tokens
            next_token = top_k_idx[next_token].item()
            generated[0].append(next_token)
            input_seq = torch.tensor(generated).to(device)

        generated_seq = decode(generated[0])
        
    return generated_seq


transformer.load_state_dict(torch.load(f"weights/transformer_new_bsize_64.pth"))
criterion = nn.CrossEntropyLoss(ignore_index=0)
optimizer = optim.Adam(transformer.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)
#optimizer = optim.SGD(transformer.parameters(), lr=0.0001, momentum=0.9)

'''
transformer.train()
for ind in range(100):  
    for epoch in range(100):
        x, y = get_batch('train')
        src_data = x
        tgt_data = y
        optimizer.zero_grad()
        output = transformer(src_data, tgt_data[:, :-1])
        ce_loss = criterion(output.contiguous().view(-1, tgt_vocab_size), tgt_data[:, 1:].contiguous().view(-1))

        repeat_loss = repetition_loss(output, tgt_data) 
        loss = ce_loss + 0.5 * repeat_loss

        print(f"Train Batch: {(ind * 100) + epoch}, Loss: {loss.item()}")
        with open('training_data3_bsize64.txt', 'a') as f:
            f.write(f"Train Batch: {(ind * 100) + epoch}, Loss: {loss.item()}\n")

        loss.backward()
        optimizer.step()
        
    generated_sequence = generate(transformer, "You are all resolved")
    print(generated_sequence)
    if (ind * 100) % 500 == 0:
        torch.save(transformer.state_dict(), f"weights/transformer_epoch_{(ind * 100)}.pth")

    torch.save(transformer.state_dict(), f"weights/transformer_new_bsize_64.pth")
    #transformer.load_state_dict(torch.load(f"weights/transformer_epoch_mod.pth"))
    evaluate(transformer, criterion, itos, ind)
'''

start_time = time.time()
transformer.to(device)
input_text ='''MENENIUS:'''
#x, y = get_batch('train')
#print(batch_to_strings(x))
#print(batch_to_strings(y))

rind = torch.randint(len(data) - block_size, (1,))[0]
x, y = torch.stack([data[rind: rind + block_size]]), torch.stack([data[rind + 1: rind + block_size + 1]])
print(batch_to_strings(x))
print(batch_to_strings(y))
output = transformer(x, y[:, :-1])

total_bleu = []
trg_words = idx_to_word(y[0], itos)
output_words = output[0].max(dim=1)[1]
output_words = idx_to_word(output_words, itos)
bleu = get_bleu(hypotheses=output_words.split(), reference=trg_words.split()) 
total_bleu.append(bleu)

generated_sequence =  generate(transformer, input_text, max_length=100)
end_time = time.time()
print(input_text)

print("\033[33m"  + generated_sequence.replace(input_text,"")  + "\033[0m")
print("Inference speed: ", (end_time - start_time), " seconds")
