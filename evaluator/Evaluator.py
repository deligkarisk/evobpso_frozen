import abc

from architecture_decoder.ArchitectureDecoder import ArchitectureDecoder
from model_creator.ModelCreator import ModelCreator


class Evaluator(abc.ABC):

    def __init__(self, architecture_decoder: ArchitectureDecoder, model_creator: ModelCreator) -> None:
        self.architecture_decoder = architecture_decoder
        self.model_creator = model_creator

    @abc.abstractmethod
    def evaluate(self, position):
        raise NotImplementedError
