import torch
import torch.nn as nn
from torchtyping import TensorType

class Solution(nn.Module):
    def __init__(self, vocabulary_size: int):
        super().__init__()
        torch.manual_seed(0)

        self.embed = nn.EmbeddingBag(vocabulary_size, 16)
        self.last_linear = nn.Linear(16, 1)
        self.sigmoid = nn.Sigmoid()


    def forward(self, x: TensorType[int]) -> TensorType[float]:
        # Hint: The embedding layer outputs a B, T, embed_dim tensor
        # but you should average it into a B, embed_dim tensor before using the Linear layer

        # Return a B, 1 tensor and round to 4 decimal places
        x = self.embed(x)
        x = self.last_linear(x)
        x = self.sigmoid(x)

        return torch.round(x, decimals=4)
  


x = torch.tensor([
    [2.0, 7.0, 14.0, 8.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [1.0, 4.0, 12.0, 3.0, 10.0, 5.0, 15.0, 11.0, 6.0, 9.0, 13.0, 7.0]
], dtype=torch.long)

sol = Solution(20)
print(sol(x))