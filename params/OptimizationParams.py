from params import FixedArchitectureParams
from params.NeuralArchitectureParams import NeuralArchitectureParams
from params.PsoParams import PsoParams


class OptimizationParams:
    def __init__(self, pso_params: PsoParams, architecture_params: NeuralArchitectureParams):
        self.pso_params = pso_params
        self.architecture_params = architecture_params
