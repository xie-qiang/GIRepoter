from lavis.models.blip2_models.blip2 import LayerNorm
import torch.nn as nn
from PIL import Image
from torchvision.transforms import Compose, Resize, ToTensor, Normalize, InterpolationMode

class PartModel(nn.Module):
    def __init__(
            self, 
            num_classes=7):
        super(PartModel, self).__init__()

        self.ln_vision =  LayerNorm(1408)
        self.num_classes = num_classes
        
        self.classifier = nn.Sequential(
            nn.Linear(1408, 256),
            nn.ReLU(),
            # nn.Dropout(0.5),
            nn.Linear(256, self.num_classes)
        )
  
    def forward(self, feats):
        features = self.ln_vision(feats)
        features = features[:, 0, :]
        logits = self.classifier(features)
        return logits
    
class ACModel(nn.Module):
    def __init__(
            self, 
            num_classes=2):
        super(ACModel, self).__init__()

        self.ln_vision =  LayerNorm(1408)
        self.num_classes = num_classes
        
        self.classifier = nn.Sequential(
            nn.Linear(1408, 256),
            nn.ReLU(),
            # nn.Dropout(0.5),
            nn.Linear(256, self.num_classes)
        )
  
    def forward(self, feats):
        features = self.ln_vision(feats)
        features = features[:, 0, :]
        logits = self.classifier(features)
        return logits
    
class GradeModel(nn.Module):
    def __init__(
            self, 
            num_classes=2):
        super(GradeModel, self).__init__()

        self.ln_vision =  LayerNorm(1408)
        self.num_classes = num_classes
        
        self.classifier = nn.Sequential(
            nn.Linear(1408, 256),
            nn.ReLU(),
            # nn.Dropout(0.5),
            nn.Linear(256, self.num_classes)
        )
    
    def forward(self, feats): 
        feats = self.ln_vision(feats)[:, 0, :]
        mean_tensor = feats.mean(dim=0)
        logits = self.classifier(mean_tensor)
        return logits
    
def _convert_to_rgb(image):
    return image.convert('RGB')

def _build_transform(resolution):
    transform = Compose([
        Resize((resolution, resolution), interpolation=InterpolationMode.BICUBIC),
        _convert_to_rgb,
        ToTensor(),
        Normalize((0.48145466, 0.4578275, 0.40821073),
                  (0.26862954, 0.26130258, 0.27577711)),
    ])
    return transform

def load_and_transform_image(image_path, resolution=224):
    image = Image.open(image_path)
    transform = _build_transform(resolution)
    image_tensor = transform(image)
    return image_tensor  # Tensor shape: [3, resolution, resolution]

if __name__ == '__main__':
    image_path = '/root/autodl-tmp/GIRepoter/sample_data/case_1/img_hfcas_03162_0_00.jpg'
    image_tensor = load_and_transform_image(image_path, resolution=224)
    print("Transformed image tensor shape:", image_tensor.shape)
