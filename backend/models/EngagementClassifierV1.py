from torch import nn
class EngagementClassifierV1(nn.Module):
    def __init__(self, input_size=12, output_size=3):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Linear(in_features=input_size, out_features=24),
            nn.BatchNorm1d(24),
            nn.ReLU(),
            nn.Dropout(0.45),

            nn.Linear(in_features=24, out_features=32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.45),

            nn.Linear(in_features=32, out_features=48),
            nn.BatchNorm1d(48),
            nn.ReLU(),
            nn.Dropout(0.45),

            nn.Linear(in_features=48, out_features=64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.45),

            nn.Linear(in_features=64, out_features=86),
            nn.BatchNorm1d(86),
            nn.ReLU(),
            nn.Dropout(0.45),

            nn.Linear(in_features=86, out_features=64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.45),

            nn.Linear(in_features=64, out_features=32),  # Fixed dimension mismatch
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.45),

            nn.Linear(in_features=32, out_features=16),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Dropout(0.45),

            nn.Linear(in_features=16, out_features=output_size),
        )

    def forward(self, x):
        return self.layers(x)  # Fixed forward method