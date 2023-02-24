from evaluator.Evaluator import Evaluator
import random


class MockIncreasingEvaluator(Evaluator):

    def __init__(self, ):
        self.count = 0

    def evaluate(self, position):
        self.count += 1
        dummy_results = {}
        return self.count, dummy_results



