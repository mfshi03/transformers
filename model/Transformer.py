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
        seq_length = tgt.size(1) 
        # Create an upper triangular matrix to prevent the model from attending to future tokens in the target sequence via causal masking.
        nopeak_mask = (1 - torch.triu(torch.ones(1, seq_length, seq_length), diagonal=1)).bool()
        tgt_mask = tgt_mask & nopeak_mask
        return src_mask, tgt_mask

    def forward(self, src, tgt):
        src_mask, tgt_mask = self.generate_mask(src, tgt)
        src_embedded = self.dropout(self.positional_encoding(self.encoder_embedding(src)))
        tgt_embedded = self.dropout(self.positional_encoding(self.decoder_embedding(tgt)))

        encoder_output = src_embedded
        for encoder_layer in self.encoder_layers:
            encoder_output = encoder_layer(encoder_output, src_mask)

        decoder_output = tgt_embedded
        for decoder_layer in self.decoder_layers:
            decoder_output = dec_layer(decoder_output, encoder_output, src_mask, tgt_mask)

        output = self.fc(decoder_output)
        return output