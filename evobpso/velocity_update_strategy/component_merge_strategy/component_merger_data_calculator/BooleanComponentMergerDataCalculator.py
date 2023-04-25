from evobpso.velocity_update_strategy.component_merge_strategy.component_merger_data_calculator.ComponentMergerDataCalculator import \
    ComponentMergerDataCalculator


class BooleanComponentMergerDataCalculator(ComponentMergerDataCalculator):
    def calculate(self, personal_component_data, global_component_data):
        result_data = personal_component_data | global_component_data
        return result_data
