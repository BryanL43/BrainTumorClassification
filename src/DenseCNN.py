import torch.nn as nn
import torchvision.models as models

class DenseCNN(nn.Module):
    def __init__(self, num_classes=4):
        super(DenseCNN, self).__init__();

        # Load pre-trained EfficientNetB0
        self.base_model = models.efficientnet_b0(pretrained=True);

        # Override the classifier layer (remove original classifier for our custom head)
        self.base_model.classifier = nn.Identity();

        # Custom dense classifier
        self.classifier = nn.Sequential(
            # Pooling layer
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),

            # First dense layer with dropout
            nn.Linear(1280, 720), # efficientnet_b0 has 1280 features
            nn.ReLU(),
            nn.Dropout(p=0.25),

            # Second dense layer with dropout
            nn.Linear(720, 360),
            nn.ReLU(),
            nn.Dropout(p=0.25),

            # Third dense layer with dropout
            nn.Linear(360, 360),
            nn.ReLU(),
            nn.Dropout(p=0.5),

            # Final dense layer without dropout
            nn.Linear(360, 180),
            nn.ReLU(),

            # Dense layer with softmax for output layer
            nn.Linear(180, num_classes),
            nn.Softmax(dim=1) # Final class probability distribution
        );

    def forward(self, x):
        x = self.base_model(x).features(x);
        x = self.classifier(x);
        return x;