"""CNN regression model for gas concentration prediction from IR spectra."""

import torch.nn as nn


class CNNRegressor(nn.Module):
    """1-D CNN that maps a single-channel spectrum to gas concentrations."""

    def __init__(
        self,
        in_channels: int = 1,
        conv1_out: int = 32,
        conv2_out: int = 64,
        kernel_size: int = 3,
        pool_size: int = 2,
        fc1_out: int = 128,
        num_targets: int = 7,
        input_length: int = 600,
    ):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, conv1_out, kernel_size)
        self.bn1 = nn.BatchNorm1d(conv1_out)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(pool_size)
        self.conv2 = nn.Conv1d(conv1_out, conv2_out, kernel_size)
        self.bn2 = nn.BatchNorm1d(conv2_out)
        self.flatten = nn.Flatten()

        # Compute flattened size after conv/pool layers
        L = input_length
        L = (L - kernel_size + 1) // pool_size          # after conv1 + pool
        L = (L - kernel_size + 1) // pool_size          # after conv2 + pool
        flat_size = conv2_out * L

        self.fc1 = nn.Linear(flat_size, fc1_out)
        self.bn3 = nn.BatchNorm1d(fc1_out)
        self.fc2 = nn.Linear(fc1_out, num_targets)

    def forward(self, x):
        x = self.pool(self.relu(self.bn1(self.conv1(x))))
        x = self.pool(self.relu(self.bn2(self.conv2(x))))
        x = self.flatten(x)
        x = self.relu(self.bn3(self.fc1(x)))
        x = self.fc2(x)
        return x
