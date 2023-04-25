from evobpso.utils import utils
from evobpso.utils.algorithm_utils import equalize_sizes
from evobpso.component_merger.ComponentMerger import ComponentMerger


class VariableLengthPickOneComponentMerger(ComponentMerger):

    def merge_personal_and_global_components(self, personal_component, global_component, pso_params):

        personal_component, global_component = equalize_sizes(personal_component, global_component)

        if len(personal_component) != len(global_component):
            raise ValueError('input components must have the same size')

        merged_components = []
        for p_entry, g_entry in zip(personal_component, global_component):
            new_entry = utils.random_choice(p_entry, g_entry, pso_params.k)
            merged_components.append(new_entry)
        return merged_components
