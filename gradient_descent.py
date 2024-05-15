
class Solution:
    def get_minimizer(self, iterations: int, learning_rate: float, init: int) -> float:

        for _ in range(iterations):
            init -= learning_rate*2*init

        return round(init, 5)


iterations = 10
learning_rate = 0.01
init = 5


sol = Solution()
print(sol.get_minimizer(iterations, learning_rate, init))