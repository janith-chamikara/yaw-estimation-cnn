import torch
import torch.nn as nn


class TunnelCNN(nn.Module):
    def __init__(self):
        super(TunnelCNN, self).__init__()

        # Input (60x80) -> Conv (58x78) -> Pool (29x39)
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32,
                      kernel_size=3, padding=0),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # Input (29x39) -> Conv (27x37) -> Pool (13x18)
        self.block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=0),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        # Input (13x18) -> Conv (11x16) -> Pool (5x8)
        self.block3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=0),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        # Input (5x8) -> Conv (3x6) -> GlobalPool (1x1)
        self.block4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=0),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )

        # GlobalAvgPool2D -> FC -> FC
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.global_pool(x)
        x = self.fc_layers(x)
        return x


if __name__ == "__main__":
    # Verification
    dummy_input = torch.randn(1, 3, 60, 80)
    model = TunnelCNN()
    output = model(dummy_input)
    print(f"Strict Model Output Shape: {output.shape}")
