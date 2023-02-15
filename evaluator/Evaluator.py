import abc

from decoder.Decoder import Decoder


class Evaluator(abc.ABC):

    def __init__(self, decoder: Decoder) -> None:
        self.decoder = decoder

    @abc.abstractmethod
    def evaluate(self, position):
        raise NotImplementedError
