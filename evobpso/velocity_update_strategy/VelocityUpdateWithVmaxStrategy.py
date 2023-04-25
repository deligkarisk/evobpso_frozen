import random
import re

from evobpso.velocity_update_strategy.StandardVelocityUpdateStrategy import StandardVelocityUpdateStrategy


class VelocityUpdateWithVmaxStrategy(StandardVelocityUpdateStrategy):

    def get_new_velocity(self, current_position, pbest_position, gbest_position):
        velocity = super().get_new_velocity(current_position, pbest_position, gbest_position)

        for layer in velocity:
            if layer.__class__.__name__ == "VelocityFactorEvolve":
                current_vel_data = layer.data
                bin_data_ones = current_vel_data.bit_count()
                if (bin_data_ones > self.params.pso_params.vmax):
                    num_bits_to_reduce = bin_data_ones - self.params.pso_params.vmax
                    new_vel_data = current_vel_data
                    for i in range(0, num_bits_to_reduce):
                        new_vel_data = self._set_random_bit_to_zero(new_vel_data)
                    layer.data = new_vel_data
        return velocity


    def _set_random_bit_to_zero(self, number):
        rnd_bit = self._get_random_one_bit(number)
        new_number = self._set_bit_to_zero(number, rnd_bit)
        return new_number

    def _get_random_one_bit(self, number):
        # randomly gets one of the 1 bits in the number
        # returns the position of the bit (counting from 0 and from the right-most end)
        bin_number = bin(number)[2:]
        all_ones = [index.start() for index in re.finditer('1', bin_number)]

        # the previous command gives the positions of the 1s in the string,
        # starting from the left. We need to reverse it and start from the right.
        all_ones = [len(bin_number) - 1 - one_pos for one_pos in all_ones]

        random_one_bit_pos_index = random.randint(0, len(all_ones) - 1)
        random_one_bit_pos = all_ones[random_one_bit_pos_index]
        return random_one_bit_pos

    def _set_bit_to_zero(self, number, bit_pos):
        # set the specific bit (counting from 0 and from right-most end) to zero
        result = number & ~(1 << bit_pos)
        return result
