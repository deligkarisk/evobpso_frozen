from evobpso.component_merger.data_calculator.ComponentMergerDataCalculator import ComponentMergerDataCalculator


class RealComponentMergerDataCalculator(ComponentMergerDataCalculator):
    def calculate(self, personal_component_data, global_component_data):
        result_data = {}
        result_data['conv_filters'] = personal_component_data['conv_filters'] + global_component_data['conv_filters']
        result_data['kernel_size'] = personal_component_data['kernel_size'] + global_component_data['kernel_size']
        result_data['pooling'] = personal_component_data['pooling'] + global_component_data['pooling']
        return result_data
