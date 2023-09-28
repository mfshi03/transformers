import torch
import torch.nn as nn

class PositionalFeedForward(nn.Module):
    def __init__(self, dim, ff_dim):
            super(PositionalFeedForward, self).__init__()
            self.fc1 = nn.Linear(dim, ff_dim)
            self.fc2 = nn.Linear(ff_dim, dim)
            self.relu = nn.ReLU()

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x))) 