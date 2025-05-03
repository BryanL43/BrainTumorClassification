import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

class DenseCNN(nn.Module):
    def __init__(self, num_classes=4):
        super(DenseCNN, self).__init__();

        # Load pre-trained EfficientNetB0 with default weights (pretrained)
        weights = EfficientNet_B0_Weights.DEFAULT;
        efficientnet = efficientnet_b0(weights=weights);

        # Use only the feature extractor
        self.features = efficientnet.features;

        # Custom dense classifier
        self.classifier = nn.Sequential(
            # Pooling layer
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),

            # First dense layer with dropout
            nn.Linear(1280, 720), # efficientnet_b0 has 1280 features
            nn.BatchNorm1d(720),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Dropout(p=0.25),

            # Second dense layer with dropout
            nn.Linear(720, 360),
            nn.BatchNorm1d(360),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Dropout(p=0.25),

            # Third dense layer with dropout
            nn.Linear(360, 360),
            nn.BatchNorm1d(360),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Dropout(p=0.5),

            # Final dense layer without dropout
            nn.Linear(360, 180),
            nn.BatchNorm1d(180),
            nn.LeakyReLU(negative_slope=0.01),

            # Dense output layer
            nn.Linear(180, num_classes)
        );

    def forward(self, x):
        logits = self.features(x);
        logits = self.classifier(logits);
        return logits;