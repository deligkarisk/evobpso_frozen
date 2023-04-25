from evobpso.utils import utils
from evobpso.utils.algorithm_utils import equalize_sizes
from evobpso.velocity_factor.VelocityFactor import VelocityFactorEvolve, VelocityFactorRemove
from evobpso.velocity_update_strategy.component_merge_strategy.ComponentMergeStrategy import ComponentMergeStrategy


class VariableLengthCalculateDataComponentMergeStrategy(ComponentMergeStrategy):

    def merge_personal_and_global_components(self, personal_component, global_component, pso_params):

        personal_component, global_component = equalize_sizes(personal_component, global_component)

        if len(personal_component) != len(global_component):
            raise ValueError('input components must have the same size')

        merged_components = []
        for p_entry, g_entry in zip(personal_component, global_component):
            # if both components are Evolve, then just calculate their new data based on the calculator
            if (isinstance(p_entry, VelocityFactorEvolve)) and isinstance(g_entry, VelocityFactorEvolve):
                result_data = self.component_merger_data_calculator.calculate(p_entry.data, g_entry.data)
                #  result_data = p_entry.data | g_entry.data
                new_entry = VelocityFactorEvolve(data=result_data)
                merged_components.append(new_entry)
        # otherwise, just select one of the components, either the personal or the global
            else:
                new_entry = utils.random_choice(p_entry, g_entry, pso_params.k)
                merged_components.append(new_entry)
        return merged_components
