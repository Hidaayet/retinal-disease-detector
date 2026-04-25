import torch.nn as nn
import timm


class RetinalClassifier(nn.Module):
    """
    EfficientNet-B3 backbone with a custom two-layer classification head
    for 5-class diabetic retinopathy grading (APTOS 2019).

    Architecture:
        EfficientNet-B3 (pretrained, num_classes=0)  →  1536-d feature vector
        Dropout(0.3) → Linear(1536, 256) → ReLU
        Dropout(0.3) → Linear(256, 5)
    """

    def __init__(self, num_classes: int = 5, dropout: float = 0.3):
        super().__init__()
        self.backbone = timm.create_model(
            "efficientnet_b3", pretrained=True, num_classes=0
        )
        feature_size = self.backbone.num_features  # 1536 for B3
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(feature_size, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.backbone(x))

    def count_parameters(self):
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total, trainable