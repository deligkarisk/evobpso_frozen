import abc

from problem.NeuralArchitecture import NeuralArchitecture
from pso_params.PsoParams import PsoParams


class Initializer(abc.ABC):

    def __init__(self, architecture: NeuralArchitecture, params: PsoParams) -> None:
        self.architecture = architecture
        self.params = params

    @abc.abstractmethod
    def get_initial_position(self):
        raise NotImplementedError

