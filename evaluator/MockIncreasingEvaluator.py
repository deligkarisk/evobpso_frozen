from evaluator.Evaluator import Evaluator
import random


class MockIncreasingEvaluator(Evaluator):

    def __init__(self, ):
        self.count = 0

    def evaluate_for_train(self, position, save_model_folder=None):
        self.count += 1
        dummy_results = {}
        return self.count



