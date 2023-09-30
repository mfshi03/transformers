import os
import requests
import tiktoken
import numpy as np

# Loads tinyshakespeare dataset into binary files for training 
input_path = os.path.join(os.path.dirname(__file__), 'input.txt')

if not os.path.exists(input_path):
    data_url = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'
    with open(input_path, 'w') as f:
        f.write(requests.get(data_url).text)

with open(input_path, 'r') as f:
    data = f.read()

train_data = data[:int(len(data)*0.9)]
val_data = data[int(len(data)*0.9):]

# Tokenize the data
enc = tiktoken.get_encoding("gpt2")
train_ids = enc.encode_ordinary(train_data)
val_ids = enc.encode_ordinary(val_data)
print(f"train has {len(train_ids):,} tokens")
print(f"val has {len(val_ids):,} tokens")

# Save to binary
train_ids = np.array(train_ids, dtype=np.uint16)
val_ids = np.array(val_ids, dtype=np.uint16)
train_ids.tofile(os.path.join(os.path.dirname(__file__), 'train.bin'))
val_ids.tofile(os.path.join(os.path.dirname(__file__), 'val.bin'))
