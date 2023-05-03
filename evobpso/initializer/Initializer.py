import abc

from evobpso.encoding.Encoding import Encoding
from evobpso.params.NeuralArchitectureParams import NeuralArchitectureParams
from evobpso.params.OptimizationParams import OptimizationParams
from evobpso.params.PsoParams import PsoParams


class Initializer(abc.ABC):

    def __init__(self, params: OptimizationParams, encoding: Encoding) -> None:
        self.architecture = params.neural_architecture_params
        self.pso_params = params.pso_params
        self.encoding = encoding

    @abc.abstractmethod
    def get_initial_position(self):
        raise NotImplementedError

