import math


# This file contains the definitions of rastrigin problem
class Rastrigin:

    def __init__(self, steps):
        self.minimize = True
        self.min_value = -5.12
        self.max_value = 5.12
        self.step = (self.max_value - self.min_value)/steps


    def evaluate(self, position):
        real_positions = [self.min_value + position_in_dimension*(self.step) for position_in_dimension in position]
        alpha = 10
        result = alpha * len(real_positions)
        for x in real_positions:
            result += x**2 - alpha*math.cos(2*math.pi*x)

        return result

