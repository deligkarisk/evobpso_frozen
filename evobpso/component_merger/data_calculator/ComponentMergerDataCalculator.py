import abc


class ComponentMergerDataCalculator(abc.ABC):

    def calculate(self, personal_component_data, global_component_data):
        raise NotImplementedError