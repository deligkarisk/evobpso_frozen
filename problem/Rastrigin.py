import math
import numpy as np


# This file contains the definitions of rastrigin problem
class Rastrigin:

    def __init__(self, dimensions):
        self.input_type = 'real'
        self.min_value = -5.12
        self.max_value = 5.12
        self.dimensions = dimensions


    def evaluate(self, position):
        alpha = 10.0
        result = alpha * len(position)
        for x in position:
            result += x**2 - alpha*np.cos(2*np.pi*x)
        return result
