import abc

from problem.NeuralArchitecture import NeuralArchitecture


class Initializer(abc.ABC):

    def __init__(self, architecture: NeuralArchitecture) -> None:
        self.architecture = architecture

    @abc.abstractmethod
    def get_initial_position(self, params):
        raise NotImplementedError

