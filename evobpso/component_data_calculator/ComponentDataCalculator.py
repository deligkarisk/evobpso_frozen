import abc

from evobpso.params.OptimizationParams import OptimizationParams


class ComponentDataCalculator(abc.ABC):

    def __init__(self, params: OptimizationParams) -> None:
        self.params = params
        super().__init__()

    @abc.abstractmethod
    def calculate(self, best_position_data, current_position_data, c_factor):
        raise NotImplementedError