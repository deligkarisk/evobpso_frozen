from evobpso.params import FixedArchitectureProperties
from evobpso.params.NeuralArchitectureParams import NeuralArchitectureParams
from evobpso.params.PsoParams import PsoParams
from evobpso.params.TrainingParams import TrainingParams


class OptimizationParams:
    def __init__(self, pso_params: PsoParams, architecture_params: NeuralArchitectureParams, training_params: TrainingParams):
        self.pso_params = pso_params
        self.architecture_params = architecture_params
        self.training_params = training_params
