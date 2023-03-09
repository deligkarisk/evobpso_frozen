import abc

from architecture_decoder.ArchitectureDecoder import ArchitectureDecoder
from model_creator.ModelCreator import ModelCreator
from params.TrainingParams import TrainingParams


class Evaluator(abc.ABC):

    def __init__(self, architecture_decoder: ArchitectureDecoder, model_creator: ModelCreator, training_params: TrainingParams, data_loader) -> None:
        self.architecture_decoder = architecture_decoder
        self.model_creator = model_creator
        self.data_loader = data_loader
        self.training_params = training_params

    @abc.abstractmethod
    def evaluate_for_train(self, position, save_model_folder):
        raise NotImplementedError
