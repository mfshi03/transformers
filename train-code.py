import time
import torch
import pickle
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from transformers import GPT2Tokenizer
from model.Transformer import Transformer
from torch.utils.data import DataLoader, TensorDataset
from bleu import idx_to_word, get_bleu
    
if torch.cuda.is_available():
    torch.set_default_tensor_type(torch.cuda.FloatTensor)

epochs = 2000
batch_size = 32  # sequences we can process in parallel
block_size = 64 # maximum context length for predictions
src_vocab_size = 50257
tgt_vocab_size = 50257
dim = 512 # embedding dimension
num_heads = 8
num_layers = 6
num_batches = 100
ff_dim  = 2048 # feed-forward dimensions
dropout = 0.1

device = 'cuda' if torch.cuda.is_available() else 'cpu'

transformer = Transformer(src_vocab_size, tgt_vocab_size, dim, num_heads, num_layers, ff_dim , block_size, dropout)

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

nl_tokens = []
code_tokens = []

block_size = 64 # maximum context length for predictions
try:
    with open('tokenized_data.pickle', 'rb') as f:
        nl_tokens, code_tokens = pickle.load(f)
except FileNotFoundError:
    with open('data/code.txt', 'r', encoding='utf-8') as f:
        for line in f:
            nl, code = line.strip().split('[SEP]')
            nl_tokens.append(torch.tensor(tokenizer.encode(nl, max_length=block_size, truncation=True), dtype=torch.long))
            code_tokens.append(torch.tensor(tokenizer.encode(code, max_length=block_size, truncation=True), dtype=torch.long))
    
    with open('tokenized_data.pickle', 'wb') as f:
        pickle.dump((nl_tokens, code_tokens), f)

# Split data into train and validation sets
n = int(0.5 * len(nl_tokens))  # 50% train, 50% validation
train_data = TensorDataset(torch.nn.utils.rnn.pad_sequence(nl_tokens[:n], batch_first=True),
                           torch.nn.utils.rnn.pad_sequence(code_tokens[:n], batch_first=True))
val_data = TensorDataset(torch.nn.utils.rnn.pad_sequence(nl_tokens[n:], batch_first=True),
                         torch.nn.utils.rnn.pad_sequence(code_tokens[n:], batch_first=True))

# DataLoader for batch generation
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, generator=torch.Generator(device='cuda'))
val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=True, generator=torch.Generator(device='cuda'))

def get_batch(data_loader):
    for x, y in data_loader:
        x, y = x.to(device), y.to(device)
        yield x, y


def batch_to_strings(batch):
    batch = batch.detach().cpu().numpy()
    strings = []
    for sample in batch:
        string = tokenizer.decode(sample) 
        strings.append(string)
        break
        
    return strings

def generate(model, initial_seq, max_length=100):
    model.eval()
    with torch.no_grad():
        input_seq = torch.tensor(tokenizer.encode(initial_seq)).unsqueeze(0).to(device)
        generated = input_seq.tolist()

        for _ in range(max_length - len(initial_seq)):
            output = model(input_seq, input_seq)
            next_token = output.argmax(dim=-1)[:, -1].item()
            generated[0].append(next_token)
            input_seq = torch.tensor(generated).to(device)
            if input_seq.size(1) > block_size:
                break

        generated_seq = tokenizer.decode(generated[0])
        
    return generated_seq

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
        input_seq = torch.tensor(tokenizer.encode(initial_seq)).unsqueeze(0).to(device)
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

        generated_seq = tokenizer.decode(generated[0])
        
    return generated_seq

criterion = nn.CrossEntropyLoss(ignore_index=0)
optimizer = optim.Adam(transformer.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)
#optimizer = optim.SGD(transformer.parameters(), lr=0.0001, momentum=0.9)


transformer.train()
for i in range(1000):  
    train_losses, test_losses, bleu_scores = [], [], []
    for x, y in get_batch(train_loader):
        src_data = x
        tgt_data = y
        #print(batch_to_strings(x))
        #print(batch_to_strings(y))
        optimizer.zero_grad()
        output = transformer(src_data, tgt_data[:, :-1])
        loss = criterion(output.contiguous().view(-1, tgt_vocab_size), tgt_data[:,1:].contiguous().view(-1))
        train_losses.append(loss)
        loss.backward()
        optimizer.step()
    
    
    train_avg = sum(train_losses) / len(train_losses)
    with open('training_code.txt', 'a') as f:
            print(f"Train Batch: {i}, Loss: {train_avg}")
            f.write(f"Train Batch: {i}, Loss: {train_avg}\n")
    
    generated_sequence = generate(transformer, "check if details are parsed")
    print(generated_sequence)
    if i % 50 == 0:
        torch.save(transformer.state_dict(), f"weights/transformer_code_{i}.pth")

torch.save(transformer.state_dict(), f"weights/transformer_code.pth")
#transformer.load_state_dict(torch.load(f"weights/transformer_epoch_mod.pth"))
#evaluate(transformer, criterion)

start_time = time.time()
transformer.to(device)
input_text = '''check the time'''

generated_sequence =  generate(transformer, input_text, max_length=100)
end_time = time.time()
print(input_text)

print("\033[33m"  + generated_sequence.replace(input_text,"")  + "\033[0m")
print("Inference speed: ", (end_time - start_time), " seconds")
