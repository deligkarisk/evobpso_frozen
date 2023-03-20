from evobpso.utils import utils
from evobpso.velocity_factor.VelocityFactor import VelocityFactorEvolve
from evobpso.velocity_update_strategy.component_merge_strategy.ComponentMergeStrategy import ComponentMergeStrategy


class PickOneComponentMergeStrategy(ComponentMergeStrategy):

    def merge_personal_and_global_components(self, personal_component, global_component, pso_params):

        if len(personal_component) != len(global_component):
            raise ValueError('input components must have the same size')

        merged_components = []
        for p_entry, g_entry in zip(personal_component, global_component):
            new_entry = utils.random_choice(p_entry, g_entry, pso_params.k)
            merged_components.append(new_entry)
        return merged_components
