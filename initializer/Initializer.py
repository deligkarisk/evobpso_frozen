import abc

from params.NeuralArchitectureParams import NeuralArchitectureParams
from params.OptimizationParams import OptimizationParams
from params.PsoParams import PsoParams


class Initializer(abc.ABC):

    def __init__(self, params: OptimizationParams) -> None:
        self.architecture = params.architecture_params
        self.pso_params = params.pso_params

    @abc.abstractmethod
    def get_initial_position(self):
        raise NotImplementedError

