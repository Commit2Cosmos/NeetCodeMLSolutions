import torch
import torch.nn as nn
import torch.nn.functional as F
from torchtyping import TensorType

# 0. Instantiate the linear layers in the following order: Key, Query, Value.
# 1. Biases are not used in Attention, so for all 3 nn.Linear() instances, pass in bias=False.
# 2. torch.transpose(tensor, 1, 2) returns a B x T x A tensor as a B x A x T tensor.
# 3. This function is useful:
#    https://pytorch.org/docs/stable/generated/torch.nn.functional.softmax.html
# 4. Apply the masking to the TxT scores BEFORE calling softmax() so that the future
#    tokens don't get factored in at all.
#    To do this, set the "future" indices to float('-inf') since e^(-infinity) is 0.
# 5. To implement masking, note that in PyTorch, tensor == 0 returns a same-shape tensor 
#    of booleans. Also look into utilizing torch.ones(), torch.tril(), and tensor.masked_fill(),
#    in that order.


class SingleHeadAttention(nn.Module):
    
    def __init__(self, embedding_dim: int, attention_dim: int):
        super().__init__()
        torch.manual_seed(0)
        self.key = nn.Linear(embedding_dim, attention_dim, bias=False)
        self.query = nn.Linear(embedding_dim, attention_dim, bias=False)
        self.value = nn.Linear(embedding_dim, attention_dim, bias=False)

        self.softmax = nn.Softmax(dim=2)


    def forward(self, embedded: TensorType[float]) -> TensorType[float]:

        k = self.key(embedded)
        q = self.query(embedded)
        v = self.value(embedded)

        embedded = q @ torch.transpose(k, 1, 2)
        embedded = embedded / (attention_dim**0.5)


        #* masking
        size = embedded.shape
        embedded = embedded.masked_fill(torch.tril(torch.ones(size[1], size[1])) == 0, float('-inf'))
        embedded = self.softmax(embedded)

        embedded = embedded @ v


        return torch.round(embedded, decimals=4)




class SingleHeadAttention2(nn.Module):
    
    def __init__(self, embedding_dim: int, attention_dim: int):
        super().__init__()
        torch.manual_seed(0)
        self.key_gen = nn.Linear(embedding_dim, attention_dim, bias=False)
        self.query_gen = nn.Linear(embedding_dim, attention_dim, bias=False)
        self.value_gen = nn.Linear(embedding_dim, attention_dim, bias=False)
    

    def forward(self, embedded: TensorType[float]) -> TensorType[float]:

        k = self.key_gen(embedded)
        q = self.query_gen(embedded)
        v = self.value_gen(embedded)

        scores = q @ torch.transpose(k, 1, 2) # @ is the same as torch.matmul()
        context_length, attention_dim = k.shape[1], k.shape[2]
        scores = scores / (attention_dim ** 0.5)

        lower_triangular = torch.tril(torch.ones(context_length, context_length))
        mask = lower_triangular == 0
        scores = scores.masked_fill(mask, float('-inf'))
        scores = nn.functional.softmax(scores, dim = 2)

        return torch.round(scores @ v, decimals=4)



torch.manual_seed(0)
embedding_dim=3
attention_dim=4
embedded=torch.randn(2, 2, 3)

# embedded = torch.tensor([
#     [[-1.4381, 0.1232],
#     [-0.1080, 0.3458]],
#     [[0.1929, -0.8567],
#     [-0.1160, 1.2547]]
# ])

sha = SingleHeadAttention(embedding_dim, attention_dim)
print(sha(embedded))


sha = SingleHeadAttention2(embedding_dim, attention_dim)
print(sha(embedded))