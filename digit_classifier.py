import torch
import torch.nn as nn
from torchtyping import TensorType


class Solution(nn.Module):
    # def __init__(self):
    #     super().__init__()
    #     torch.manual_seed(0)
    #     self.layers = nn.Sequential(
    #         nn.Linear(28*28, 512),
    #         nn.ReLU(),
    #         nn.Dropout1d(p=0.2),
    #         nn.Linear(512, 10),
    #         nn.Sigmoid()
    #     )

    
    # def forward(self, images: TensorType[float]) -> TensorType[float]:
    #     torch.manual_seed(0)
    #     images = self.layers(images)

    #     return torch.round(images, decimals=4)

    def __init__(self):
        super().__init__()
        torch.manual_seed(0)
        self.first_linear = nn.Linear(784, 512)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.2)
        self.projection = nn.Linear(512, 10)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, images: TensorType[float]) -> TensorType[float]:
        torch.manual_seed(0)
        out = self.sigmoid(self.projection(self.dropout(self.relu(self.first_linear(images)))))
        return torch.round(out, decimals=4)