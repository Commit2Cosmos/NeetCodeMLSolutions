import torch
from typing import List, Tuple

class Solution:
    def batch_loader(self, raw_dataset: str, context_length: int, batch_size: int) -> Tuple[List[List[str]]]:
        # You must start by generating batch_size different random indices in the appropriate range
        # using a single call to torch.randint()
        raw_dataset = raw_dataset.split()

        torch.manual_seed(0)
        start_idx = torch.randint(0, len(raw_dataset)-context_length, (batch_size,))



        X = [[] for _ in range(batch_size)]
        Y = [[] for _ in range(batch_size)]
        
        for i in range(batch_size):
            for j in range(start_idx[i], start_idx[i]+context_length):
                X[i].append(raw_dataset[j])
                Y[i].append(raw_dataset[j+1])

        return X, Y


raw_dataset='Once upon a time on a GPU far far away there was an algorithm'
context_length=3
batch_size=2


sol = Solution()
print(sol.batch_loader(raw_dataset, context_length, batch_size))