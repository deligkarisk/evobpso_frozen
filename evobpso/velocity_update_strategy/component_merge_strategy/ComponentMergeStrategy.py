import abc


class ComponentMergeStrategy(abc.ABC):

    @abc.abstractmethod
    def merge_personal_and_global_components(self, personal_component, global_component, pso_params):
        # this method merges the personal and global components of the velocity equation.
        # it corresponds to the formula personal_component + global_component
        raise NotImplementedError