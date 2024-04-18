import torch
import torch.nn as nn
import torch.nn.functional as F

from src.SpeechLM.SpeechLM import SpeechLM, SpeechLMConfig
from src.model.base_model import BaseModel



class SpeechLMMos(nn.Module):
    def __init__(self, freeze: bool = True, checkpoint_path: str = None):
        super().__init__()
        checkpoint = torch.load(checkpoint_path)
        cfg = SpeechLMConfig(checkpoint['cfg']['model'])
        self.encoder = SpeechLM(cfg)
        self.encoder.load_state_dict(checkpoint['model'])
        self.encoder.eval()
        self.freeze = freeze
        
        self.dense = nn.Sequential(
            nn.Linear(768, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 1)
        )
        
        if self.freeze:
            self.encoder.eval()
            for p in self.encoder.parameters():
                p.requires_grad_(False)
        
    def forward(self, x):
        x = self.encoder.extract_features(x)[0] # [Batch, time, feats]
        x = self.dense(x) # [batch, time, 1]
        x = x.mean(dim=[1,2], keepdims=True) # [batch, 1, 1]
        return x
                