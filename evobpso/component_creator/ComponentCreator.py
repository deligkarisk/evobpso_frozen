import abc

from evobpso.component_creator_data_calculator.ComponentCreatorDataCalculator import ComponentCreatorDataCalculator


class ComponentCreator(abc.ABC):

    def __init__(self, data_calculator: ComponentCreatorDataCalculator) -> None:
        self.data_calculator = data_calculator
        super().__init__()

    @abc.abstractmethod
    def create_component(self, best_position, current_position, c_factor):
        raise NotImplementedError
