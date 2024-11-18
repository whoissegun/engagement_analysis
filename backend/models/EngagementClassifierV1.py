import torch
import torch.nn as nn
import torch.nn.functional as F

class EngagementClassifierV1(nn.Module):
    def __init__(self, input_size=12, output_size=3):
        super().__init__()

        # Smaller network with careful normalization
        self.bn_input = nn.BatchNorm1d(input_size)

        self.block1 = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.3)
        )

        self.block2 = nn.Sequential(
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.3)
        )

        self.classifier = nn.Linear(32, output_size)

        # Initialize weights properly
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)

    def forward(self, x):
        x = self.bn_input(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.classifier(x)
        return x
