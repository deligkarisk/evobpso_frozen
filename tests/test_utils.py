from statistics import mean
from unittest import TestCase
from utils.utils import create_rnd_binary_vector, find_largest_size, find_smallest_size


class Test(TestCase):
    def test_create_rnd_binary_vector(self):
        assert create_rnd_binary_vector(1, 6) == 0b111111
        assert create_rnd_binary_vector(0, 6) == 0b000000
        assert create_rnd_binary_vector(1, 12) == 0b111111111111

        # Running it for thousands of times with 50% probability at 12 bits should average at about 6 bits
        assert abs(mean([bin(create_rnd_binary_vector(0.5, 12)).count('1') for i in range(0, 10000)]) - 6) < 0.2

    def test_find_largest_size_works_regardless_of_order_of_objects(self):
        list_a = [1, 2, 3]
        list_b = [0, 0, 0, 0]
        assert 4 == find_largest_size(list_a, list_b)
        assert 4 == find_largest_size(list_b, list_a)

    def test_find_largest_size_works_with_empty_lists(self):
        list_a = [1, 2, 3]
        list_b = []
        assert 3 == find_largest_size(list_a, list_b)

    def test_find_smallest_size_works_regardless_of_order_of_objects(self):
        list_a = [1, 2, 3]
        list_b = [0, 0, 0, 0]
        assert 3 == find_smallest_size(list_a, list_b)
        assert 3 == find_smallest_size(list_b, list_a)

    def test_find_smallest_size_works_with_empty_lists(self):
        list_a = [1, 2, 3]
        list_b = []
        assert 0 == find_smallest_size(list_a, list_b)