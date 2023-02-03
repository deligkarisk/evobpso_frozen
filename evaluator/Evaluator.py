import abc


class Evaluator(abc.ABC):

    @abc.abstractmethod
    def evaluate(self, position):
        raise NotImplementedError
