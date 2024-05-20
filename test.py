import torch
import torch.nn as nn

torch.manual_seed(0)

softmax = nn.Softmax(dim=2)

t = torch.rand((2,2,2))

print(t[:,-1,:].shape)