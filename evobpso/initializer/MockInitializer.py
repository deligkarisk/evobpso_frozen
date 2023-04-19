import random

from evobpso.initializer.Initializer import Initializer
from evobpso.utils.utils import create_rnd_binary_vector


class MockInitializer(Initializer):
    def get_initial_position(self):
        position = [0b0100000000001, 0b0100000000010, 0b0100100000100, 0b0100000011000, 0b1100000010000,
                    0b0000000010000, 0b0100000000001, 0b0100100000010, 0b0100000010100, 0b0110001001000,
                    0b0000100010000, 0b1101001010000]
        return position

