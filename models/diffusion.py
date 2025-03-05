import torch
import torch.nn as nn
import math

class DiffusionModel(nn.Module):
    def __init__(self, input_dim=80, hidden_dim=512, num_steps=1000):
        super().__init__()
        self.num_steps = num_steps
        
        # U-Net style architecture
        self.encoder = nn.Sequential(
            nn.Conv1d(input_dim, hidden_dim, 3, padding=1),
            nn.ReLU(),
            nn.Conv1d(hidden_dim, hidden_dim * 2, 3, stride=2, padding=1),
            nn.ReLU(),
        )
        
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(hidden_dim * 2, hidden_dim, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.Conv1d(hidden_dim, input_dim, 3, padding=1),
        )
        
    def forward(self, x, t):
        # x: input spectrogram, t: diffusion timestep
        t_embed = self._get_timestep_embedding(t)
        
        # Encode
        h = self.encoder(x)
        
        # Add time embedding
        h = h + t_embed.unsqueeze(-1)
        
        # Decode
        return self.decoder(h)
    
    def _get_timestep_embedding(self, t):
        # Simple sinusoidal time embedding
        half_dim = self.hidden_dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim) * -emb)
        emb = t[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
        return emb