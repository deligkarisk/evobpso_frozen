from evobpso.component_merger.data_calculator.ComponentMergerDataCalculator import \
    ComponentMergerDataCalculator


class BooleanComponentMergerDataCalculator(ComponentMergerDataCalculator):
    def calculate(self, personal_component_data, global_component_data):
        result_data = personal_component_data | global_component_data
        return result_data
