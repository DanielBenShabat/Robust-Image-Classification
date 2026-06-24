import torch
import torch.nn as nn


def conv_block(in_channels: int, out_channels: int) -> nn.Sequential:
    """A single Conv -> BatchNorm -> ReLU block (3x3, same padding)."""
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    )


class ModelArchitecture(nn.Module):
    """
    From-scratch convolutional classifier for the 20-class hackathon task.

    A compact VGG-style network: four stages of (two conv blocks + max-pool),
    doubling the channel width each stage, followed by global average pooling
    and a linear classifier. No pretrained weights are used.

    Input : float tensor [batch, 3, 224, 224] (ImageNet-normalized)
    Output: logits        [batch, 20]
    """

    def __init__(self, num_classes: int = 20):
        super().__init__()

        self.features = nn.Sequential(
            conv_block(3, 32),
            conv_block(32, 32),
            nn.MaxPool2d(2),                 # 224 -> 112

            conv_block(32, 64),
            conv_block(64, 64),
            nn.MaxPool2d(2),                 # 112 -> 56

            conv_block(64, 128),
            conv_block(128, 128),
            nn.MaxPool2d(2),                 # 56 -> 28

            conv_block(128, 256),
            conv_block(256, 256),
            nn.MaxPool2d(2),                 # 28 -> 14
        )

        # Global average pooling makes the head input-size agnostic.
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

        self._initialize_weights()

    def _initialize_weights(self) -> None:
        """Kaiming init for convs, standard init for BN and the linear head."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.pool(x)
        logits = self.classifier(x)
        return logits
