import abc

from params.Params import Params


class ComponentDataCalculator(abc.ABC):

    def __init__(self, params: Params) -> None:
        self.params = params
        super().__init__()

    @abc.abstractmethod
    def calculate(self, best_position_data, current_position_data, c_factor):
        raise NotImplementedError