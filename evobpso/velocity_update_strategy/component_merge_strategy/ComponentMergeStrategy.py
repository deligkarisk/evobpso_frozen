import abc

from evobpso.velocity_update_strategy.component_merge_strategy.component_merger_data_calculator.ComponentMergerDataCalculator import \
    ComponentMergerDataCalculator


class ComponentMergeStrategy(abc.ABC):

    def __init__(self, component_merger_data_calculator: ComponentMergerDataCalculator):
        self.component_merger_data_calculator = component_merger_data_calculator

    @abc.abstractmethod
    def merge_personal_and_global_components(self, personal_component, global_component, pso_params):
        # this method merges the personal and global components of the velocity equation.
        # it corresponds to the formula personal_component + global_component
        raise NotImplementedError