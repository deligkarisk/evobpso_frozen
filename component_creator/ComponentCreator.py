import abc

from component_data_calculator.ComponentDataCalculator import ComponentDataCalculator
from params.Params import Params


class ComponentCreator(abc.ABC):

    def __init__(self, data_calculator: ComponentDataCalculator) -> None:
        self.data_calculator = data_calculator
        super().__init__()

    @abc.abstractmethod
    def create_component(self, best_position, current_position, c_factor):
        raise NotImplementedError
