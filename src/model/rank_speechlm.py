import torch

from src.SpeechLM.SpeechLM import SpeechLM, SpeechLMConfig
from src.model.base_model import BaseModel
from src.model.fusion_blocks import LinearFusion, CrossAttentionFusion



class SpeechLMMos(BaseModel):
    def __init__(self, checkpoint_path: str = None, fusion_mode="linear"):
        super().__init__()
        assert fusion_mode in ["linear", "cross"], "Wrong fusion mode!"

        self._setup_encoder(checkpoint_path)
        self.fusion = LinearFusion() if fusion_mode == "linear" else CrossAttentionFusion()
        
    def forward(self, x, y, **batch):
        x = self.encoder.extract_features(x)[0] 
        y = self.encoder.extract_features(y)[0]
        return self.fusion(x, y)
    
    def predict(self, x):
        x = self.encoder.extract_features(x)[0] # [Batch, time, feats]
        return self.fusion.predict(x)
    
    def _setup_encoder(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        cfg = SpeechLMConfig(checkpoint['cfg']['model'])
        self.encoder = SpeechLM(cfg)
        self.encoder.load_state_dict(checkpoint['model'])
        self.encoder.eval()
        
        for p in self.encoder.parameters():
            p.requires_grad_(False)
                  