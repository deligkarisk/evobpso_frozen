import abc

from params.NeuralArchitectureParams import NeuralArchitectureParams
from params.Params import Params
from params.PsoParams import PsoParams


class Initializer(abc.ABC):

    def __init__(self, params: Params) -> None:
        self.architecture = params.architecture_params
        self.pso_params = params.pso_params

    @abc.abstractmethod
    def get_initial_position(self):
        raise NotImplementedError

