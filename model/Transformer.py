import torch
import torch.nn as nn
from torch.nn import functional as F

from model.PositionalEncoding import PositionalEncoding
from model.EncoderLayer import EncoderLayer
from model.DecoderLayer import DecoderLayer

class Transformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, dim, n_heads, n_layers, ff_dim, max_seq_len, dropout):
        super(Transformer, self).__init__()
        self.encoder_embedding = nn.Embedding(src_vocab_size, dim)
        self.decoder_embedding = nn.Embedding(tgt_vocab_size, dim)
        self.positional_encoding = PositionalEncoding(dim, max_seq_len)

        self.encoder_layers = nn.ModuleList([EncoderLayer(dim, n_heads, ff_dim, dropout) for _ in range(n_layers)])
        self.decoder_layers = nn.ModuleList([DecoderLayer(dim, n_heads, ff_dim, dropout) for _ in range(n_layers)])

        self.fc = nn.Linear(dim, tgt_vocab_size)
        self.dropout = nn.Dropout(dropout)

    def generate_mask(self, src, tgt):
        src_mask = (src != 0).unsqueeze(1).unsqueeze(2)
        tgt_mask = (tgt != 0).unsqueeze(1).unsqueeze(3)
        tgt_len = tgt.size(1) 
        # Create an upper triangular matrix to prevent the model from attending to future tokens in the target sequence via causal masking.
        nopeak_mask = (1 - torch.triu(torch.ones(tgt_len, tgt_len),diagonal=1)).bool().to(tgt.device)
        tgt_mask = tgt_mask & nopeak_mask
        return src_mask, tgt_mask
    
    def generate_src_mask(self, src):
        src_mask = (src != 0).unsqueeze(1).unsqueeze(2)
        return src_mask

    def generate_tgt_mask(self, tgt):
        tgt_mask = (tgt != 0).unsqueeze(1).unsqueeze(3)
        tgt_len = tgt.size(1) 
        nopeak_mask = (1 - torch.triu(torch.ones(tgt_len, tgt_len),diagonal=1)).bool().to(tgt.device)
        tgt_mask = tgt_mask & nopeak_mask
        return tgt_mask
    
    def repetition_loss(logits, target):
        # Get previous target tokens 
        prev_tokens = target[:, :-1]  
        # Compare logit predictions to previous tokens
        repeat_logits = (logits == prev_tokens).float() 
        # Apply softmax to get probabilities
        repeat_probs = F.softmax(repeat_logits, dim=1)
        # Calculate CE loss to maximize probability of not repeating
        return F.cross_entropy(repeat_probs, torch.zeros_like(repeat_probs))

    def forward(self, src, tgt):
        src_mask, tgt_mask = self.generate_mask(src, tgt)
        src_embedded = self.dropout(self.positional_encoding(self.encoder_embedding(src)))
        tgt_embedded = self.dropout(self.positional_encoding(self.decoder_embedding(tgt)))

        encoder_output = src_embedded
        for encoder_layer in self.encoder_layers:
            encoder_output = encoder_layer(encoder_output, src_mask)

        decoder_output = tgt_embedded
        for decoder_layer in self.decoder_layers:
            decoder_output = decoder_layer(decoder_output, encoder_output, src_mask, tgt_mask)

        output = self.fc(decoder_output)
        return output

    @torch.no_grad()
    def generate(self, src, max_length=100):
        # Generate the source mask
        src_mask = self.generate_source_mask(src)
        src_embedded = self.dropout(self.positional_encoding(self.encoder_embedding(src)))

        # Process the source sequence through the encoder
        encoder_output = src_embedded
        for encoder_layer in self.encoder_layers:
            encoder_output = encoder_layer(encoder_output, src_mask)

        # Initialize the target sequence with a start-of-sequence token
        tgt = [self.start_token_id]
        decoder_output = None

        for _ in range(max_length):
            # Convert the target sequence to a tensor and generate its mask
            tgt_tensor = torch.tensor(tgt).unsqueeze(0).to(src.device) # Assuming batch size of 1
            tgt_mask = self.generate_target_mask(tgt_tensor)

            # Embed the target sequence
            tgt_embedded = self.dropout(self.positional_encoding(self.decoder_embedding(tgt_tensor)))

            # Process the target sequence through the decoder
            decoder_output = tgt_embedded
            for decoder_layer in self.decoder_layers:
                decoder_output = decoder_layer(decoder_output, encoder_output, src_mask, tgt_mask)

            # Obtain the last token's output and convert it to a token ID
            next_token_id = self.get_next_token_id(decoder_output)
            tgt.append(next_token_id)

            # Check for end-of-sequence token
            if next_token_id == self.end_token_id:
                break

        output = self.fc(decoder_output)
        return output
 
    
    
