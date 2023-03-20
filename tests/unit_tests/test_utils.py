from statistics import mean
from unittest import TestCase
from evobpso.utils.utils import create_rnd_binary_vector, find_largest_size, find_smallest_size, extract_integer_from_subset_of_bits


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

    def test_extract_integer_from_bits(self):
        number = 0b101001010101110
        first_nine_bits = extract_integer_from_subset_of_bits(number, 0, 9)
        assert first_nine_bits == 0b010101110
        first_three_bits = extract_integer_from_subset_of_bits(number, 0, 3)
        assert first_three_bits == 6
        assert first_three_bits == 0b110
        nine_bits_from_index1 = extract_integer_from_subset_of_bits(number, 1, 9)
        assert nine_bits_from_index1 == 0b101010111
        three_bits_from_index_9 = extract_integer_from_subset_of_bits(number, 9, 3)
        assert three_bits_from_index_9 == 1
        assert three_bits_from_index_9 == 0b001
        last_bit = extract_integer_from_subset_of_bits(number, 14, 1)
        assert last_bit == 1
        assert last_bit == 0b1
