import torch
import torch.nn as nn

class NMTModel(nn.Module):
    def __init__(self, src_vocab_size=10000, tgt_vocab_size=10000, 
                 hidden_dim=512, num_layers=6):
        super().__init__()
        
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=8),
            num_layers=num_layers
        )
        
        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model=hidden_dim, nhead=8),
            num_layers=num_layers
        )
        
        self.src_embedding = nn.Embedding(src_vocab_size, hidden_dim)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, hidden_dim)
        self.output_layer = nn.Linear(hidden_dim, tgt_vocab_size)
        
    def forward(self, src, tgt):
        src_embed = self.src_embedding(src)
        tgt_embed = self.tgt_embedding(tgt)
        
        encoder_output = self.encoder(src_embed)
        decoder_output = self.decoder(tgt_embed, encoder_output)
        
        return self.output_layer(decoder_output)