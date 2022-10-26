from statistics import mean
from unittest import TestCase
from utils import create_rnd_binary_vector


class Test(TestCase):
    def test_create_rnd_binary_vector(self):

        assert create_rnd_binary_vector(1, 6) == 0b111111
        assert create_rnd_binary_vector(0, 6) == 0b000000
        assert create_rnd_binary_vector(1, 12) == 0b111111111111

        # Running it for thousands of times with 50% probability at 12 bits should average at about 6 bits
        assert abs(mean([bin(create_rnd_binary_vector(0.5, 12)).count('1') for i in range(0, 10000)]) - 6) < 0.2

