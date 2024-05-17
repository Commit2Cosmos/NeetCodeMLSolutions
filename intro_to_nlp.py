from typing import List
import torch
import torch.nn as nn
from torchtyping import TensorType


class Solution:
    def get_dataset(self, positive: List[str], negative: List[str]) -> TensorType[float]:

        words = set()

        combo = positive + negative
        
        for sent in combo:
            for word in sent.split():
                words.add(word)


        sorted_lst = sorted(list(words))
        mapping = {word: i+1 for i, word in enumerate(sorted_lst)}

        res = []
        for sent in combo:
            temp = []
            for word in sent.split():
                temp.append(mapping[word])
            res.append(torch.tensor(temp))


        return torch.nn.utils.rnn.pad_sequence(res, batch_first=True)


positive = ["Dogecoin to the moon", "I will do something else"]
negative = ["I will short Tesla today", "I am special help"]

sol = Solution()
print(sol.get_dataset(positive, negative))