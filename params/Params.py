from params.NeuralArchitectureParams import NeuralArchitectureParams
from params.PsoParams import PsoParams


class Params:
    def __init__(self, pso_params: PsoParams, architecture_params: NeuralArchitectureParams):
        self.pso_params = pso_params
        self.architecture_params = architecture_params
