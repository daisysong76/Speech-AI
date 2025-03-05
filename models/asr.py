from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import torch.nn.functional as F
import torch
import torch.nn as nn

class ASRModel(nn.Module):
    def __init__(self, input_dim=768, hidden_dim=512, vocab_size=10000):
        super().__init__()
        self.encoder = nn.LSTM(input_dim, hidden_dim, 
                             num_layers=3, bidirectional=True, batch_first=True)
        self.decoder = nn.Linear(hidden_dim * 2, vocab_size)
        
    def forward(self, x):
        # x shape: (batch_size, sequence_length, input_dim)
        encoder_outputs, _ = self.encoder(x)
        logits = self.decoder(encoder_outputs)
        return logits

class SpeechRecognizer(nn.Module):
    def __init__(self, pretrained_model_name="facebook/wav2vec2-large-960h-lv60-self"):
        super().__init__()
        self.processor = Wav2Vec2Processor.from_pretrained(pretrained_model_name)
        self.model = Wav2Vec2ForCTC.from_pretrained(pretrained_model_name)
        
        # SpecAugment-like data augmentation layer
        self.spec_augment = nn.Sequential(
            nn.Dropout2d(p=0.1),  # Time masking
            nn.Dropout2d(p=0.1)   # Frequency masking
        )

    def forward(self, audio_input):
        # Apply SpecAugment during training
        if self.training:
            audio_input = self.spec_augment(audio_input)
        
        inputs = self.processor(
            audio_input, 
            return_tensors="pt", 
            padding=True, 
            sampling_rate=16000
        )
        
        outputs = self.model(**inputs)
        logits = outputs.logits
        
        # Compute CTC loss and decode
        predicted_ids = torch.argmax(logits, dim=-1)
        transcription = self.processor.batch_decode(predicted_ids)
        
        return transcription, logits