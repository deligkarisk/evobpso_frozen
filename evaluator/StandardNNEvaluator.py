from evaluator.Evaluator import Evaluator


class StandardNNEvaluator(Evaluator):

    def evaluate(self, position):
        decoded_position = self.decoder.decode(position)
        raise NotImplementedError