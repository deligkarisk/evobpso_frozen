from utils import utils
from velocity_factor.VelocityFactor import VelocityFactorEvolve
from velocity_update_strategy.component_merge_strategy.ComponentMergeStrategy import ComponentMergeStrategy


class StandardComponentMergeStrategy(ComponentMergeStrategy):

    def merge_personal_and_global_components(self, personal_component, global_component, pso_params):

        if len(personal_component) != len(global_component):
            raise ValueError('input components must have the same size')

        merged_components = []
        for p_entry, g_entry in zip(personal_component, global_component):
            # if both components are Evolve, then just OR their data
            if (isinstance(personal_component, VelocityFactorEvolve)) and isinstance(global_component, VelocityFactorEvolve):
                result_data = p_entry.data | g_entry.data
                new_entry = VelocityFactorEvolve(data=result_data)
                merged_components.append(new_entry)
            # otherwise, just select one of the components, either the personal or the global
            else:
                new_entry = utils.random_choice(p_entry, g_entry, pso_params.k)
                merged_components.append(new_entry)
        return merged_components