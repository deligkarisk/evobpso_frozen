from component_creator.ComponentCreator import ComponentCreator
from utils.utils import find_largest_size, find_smallest_size, find_largest_index
from velocity_factor.VelocityFactor import VelocityFactorEvolve, VelocityFactorAdd, VelocityFactorRemove


class StandardBooleanComponentCreator(ComponentCreator):

    def create_component(self, best_position, current_position, c_factor):
        # creates the social and personal components of the velocity equation.
        largest_size = find_largest_size(best_position, current_position)
        smallest_size = find_smallest_size(best_position, current_position)
        best_position_is_larger = find_largest_index(best_position, current_position) == 0
        new_component = []

        # for the dimensions that both positions have, produce component using the xor and "and" operations
        for current_index in range(0, smallest_size):
            #component_data = (best_position[current_index] ^ current_position[current_index]) & rnd_vector_partial()
            component_data = self.data_calculator.calculate(best_position[current_index], current_position[current_index], c_factor)
            velocity_component = VelocityFactorEvolve(data=component_data)
            new_component.append(velocity_component)

        # subsequently, fill the rest with either 'Add' or 'Remove'.
        # this depends on whether the best (global or personal) is larger than the current position or vice versa
        for current_index in range(smallest_size, largest_size):
            if best_position_is_larger:
                velocity_component = VelocityFactorAdd(data=best_position[current_index])
                new_component.append(velocity_component)
            else:
                velocity_component = VelocityFactorRemove()
                new_component.append(velocity_component)
        return new_component



