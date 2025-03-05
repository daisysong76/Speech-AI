import torch
from torch.utils.data import Dataset, DataLoader
import torchaudio
import os

class SpeechTranslationDataset(Dataset):
    def __init__(self, english_speech_dir, mandarin_speech_dir, text_pairs_file):
        self.english_speech_dir = english_speech_dir
        self.mandarin_speech_dir = mandarin_speech_dir
        
        # Load text pairs
        self.text_pairs = []
        with open(text_pairs_file, 'r', encoding='utf-8') as f:
            for line in f:
                eng, man = line.strip().split('\t')
                self.text_pairs.append((eng, man))
                
    def __len__(self):
        return len(self.text_pairs)
    
    def __getitem__(self, idx):
        eng_text, man_text = self.text_pairs[idx]
        
        # Load audio files
        eng_audio_path = os.path.join(self.english_speech_dir, f"{idx}.wav")
        man_audio_path = os.path.join(self.mandarin_speech_dir, f"{idx}.wav")
        
        eng_audio, _ = torchaudio.load(eng_audio_path)
        man_audio, _ = torchaudio.load(man_audio_path)
        
        return {
            'eng_audio': eng_audio,
            'man_audio': man_audio,
            'eng_text': eng_text,
            'man_text': man_text
        }

def create_dataloader(metadata_csv, audio_dir, batch_size=32):
    dataset = SpeechTranslationDataset(metadata_csv, audio_dir)
    return DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=4
    )