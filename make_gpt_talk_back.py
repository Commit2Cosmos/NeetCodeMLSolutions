import torch
import torch.nn as nn
from torchtyping import TensorType

class Solution:
    def generate(self, model, new_chars: int, context: TensorType[int], context_length: int, int_to_char: dict) -> str:
        # 1. Use torch.multinomial() to choose the next token.
        #    This function simulates a weighted draw from a given list of probabilities
        #    It's similar to picking marbles out of a bag.
        # 2. the given model's output is BEFORE softmax is applied,
        #    and the forward() output has shape batch X time X vocab_size
        # 3. Do not alter the code below, only add to it. This is for maintaining reproducibility in testing.

        generator = torch.manual_seed(0)
        initial_state = generator.get_state()
        for i in range(new_chars):

            #* cut context length
            curr_context_len = context.shape[1]
            if curr_context_len > context_length:
                context = context[:, curr_context_len-context_length:, :]

            output = model(context)
            pred = nn.functional.softmax(output[:,-1,:], dim=-1)
            next_char = torch.multinomial(pred, 1, generator=generator)

            generator.set_state(initial_state)
            
            context = torch.concat((context, next_char), dim=1)

        #* remove start character
        context = context[:,1:]

        res = ''

        for intt in context[0]:
            res += int_to_char[intt.item()]
        
        return res


        # Once your code passes the test, check out the Colab link and hit Run to see your code generate new Drake lyrics!
        # Your code's output, ran in this sandbox will be boring because of the computational limits in this sandbox