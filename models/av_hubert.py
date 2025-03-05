import torch
import torch.nn as nn
from transformers import HubertModel, Wav2Vec2FeatureExtractor

class AVHuBERT(nn.Module):
    def __init__(self, input_dim=80, hidden_dim=768, num_layers=12):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Feature encoder
        self.feature_encoder = nn.Sequential(
            nn.Conv1d(input_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim)
        )
        
        # Transformer layers
        self.transformer_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=8)
            for _ in range(num_layers)
        ])
        
    def forward(self, x):
        # x shape: (batch_size, sequence_length, input_dim)
        x = x.transpose(1, 2)  # (batch_size, input_dim, sequence_length)
        x = self.feature_encoder(x)
        x = x.transpose(1, 2)  # (batch_size, sequence_length, hidden_dim)
        
        # Apply transformer layers
        for layer in self.transformer_layers:
            x = layer(x)
            
        return x

class AVHubertEncoder(nn.Module):
    def __init__(self, pretrained_model_name="facebook/hubert-large-ls960-fps50"):
        super().__init__()
        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(pretrained_model_name)
        self.hubert = HubertModel.from_pretrained(pretrained_model_name)
        
        # LoRA adaptation
        self.lora_adapter = nn.Sequential(
            nn.Linear(self.hubert.config.hidden_size, 128),
            nn.ReLU(),
            nn.Linear(128, self.hubert.config.hidden_size)
        )

    def forward(self, audio_input, video_input=None):
        # Preprocess audio
        inputs = self.feature_extractor(
            audio_input, 
            return_tensors="pt", 
            sampling_rate=16000, 
            padding=True
        )
        
        # Extract features
        with torch.no_grad():
            outputs = self.hubert(**inputs)
        
        # Apply LoRA adaptation
        features = outputs.last_hidden_state
        adapted_features = features + self.lora_adapter(features)
        
        return adapted_features