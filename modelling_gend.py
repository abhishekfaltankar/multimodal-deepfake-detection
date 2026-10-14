import torch
import torch.nn as nn
from transformers import PreTrainedModel, CLIPVisionModel, PretrainedConfig

class GendConfig(PretrainedConfig):
    model_type = "gend"
    def __init__(self, vertex_num=15, embed_dim=768, num_classes=2, **kwargs):
        super().__init__(**kwargs)
        self.vertex_num = vertex_num
        self.embed_dim = embed_dim
        self.num_classes = num_classes

class GendModel(PreTrainedModel):
    config_class = GendConfig
    
    def __init__(self, config):
        super().__init__(config)
        self.backbone = CLIPVisionModel.from_pretrained("openai/clip-vit-large-patch14")
        
        # 1. Safely extract the exact vector dimension directly from the active OpenAI eye
        embed_dim = self.backbone.config.hidden_size  # This guarantees it reads 1024 dynamically!
        
        # 2. Safe dynamic tracking for binary labels
        num_classes = getattr(config, "num_classes", getattr(config, "num_labels", 2))
        
        self.classifier = nn.Linear(embed_dim, num_classes)
        
    def forward(self, pixel_values):
        outputs = self.backbone(pixel_values=pixel_values)
        pooled_output = outputs.pooler_output
        logits = self.classifier(pooled_output)
        return logits