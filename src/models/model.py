import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights



class BinaryClassifier(nn.Module):
    def __init__(self, num_classes):
        super(BinaryClassifier, self).__init__()
        weights = EfficientNet_B0_Weights.IMAGENET1K_V1 
        self.efficientnet = models.efficientnet_b0(weights=weights) # efficientnet model, 1280 output feature
        
        num_features = self.efficientnet.classifier[1].in_features
        self.efficientnet.classifier = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes),
        )

    def forward(self, input):
        x = self.efficientnet(input)
        return x
    ######otan tha kano ta embeddings ton images opos m eipe maybe tha piano ta 512 features
    ### pou einai pio pano k na ta kano return edo mazi me to x
    
if __name__=="__main__":
    num_classes = 1
    model = BinaryClassifier(num_classes)
    _input = torch.randn(1, 3, 256, 256)
    with torch.no_grad():
        output = model(_input)

    print("Input shape:", _input.shape)
    print("Output shape:", output[0].shape)