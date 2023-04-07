import torch
import torch.nn as nn

class DnCNN(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, num_layers=17, num_features=64):
        super(DnCNN, self).__init__()

        layers = [
            nn.Conv2d(in_channels, num_features, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        ]

        for _ in range(num_layers-2):
            layers.extend([
                nn.Conv2d(num_features, num_features, kernel_size=3, padding=1),
                nn.BatchNorm2d(num_features),
                nn.ReLU(inplace=True)
            ])

        layers.append(nn.Conv2d(num_features, out_channels, kernel_size=3, padding=1))

        self.dncnn = nn.Sequential(*layers)

    def forward(self, x):
        return x - self.dncnn(x)