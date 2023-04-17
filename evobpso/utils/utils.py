import os
import random
import numpy
import tensorflow as tf


def create_rnd_binary_vector(prob, n_bits):
    result = 0

    for x in range(0, n_bits):
        x_rnd = random.random()
        if x_rnd < prob:
            bit_mask = 1 << x
            result = result | bit_mask
    return result


def clear_bit(vector, bit_position):
    vector &= ~(1 << bit_position)
    return vector


def set_bit(vector, bit_position):
    vector |= 1 << bit_position
    return vector


def find_largest_size(vector_a, vector_b):
    # Finds the size of the largest object (e.g. list)
    if len(vector_a) > len(vector_b):
        return len(vector_a)
    else:
        return len(vector_b)


def find_smallest_size(vector_a, vector_b):
    # Finds the size of the smallest object (e.g. list)
    if len(vector_a) < len(vector_b):
        return len(vector_a)
    else:
        return len(vector_b)


def find_smallest_index(vector_a, vector_b):
    # Finds the size of the smallest object (e.g. list)
    if len(vector_a) < len(vector_b):
        return 0
    else:
        return 1


def find_largest_index(vector_a, vector_b):
    # Finds the size of the smallest object (e.g. list)
    if len(vector_a) > len(vector_b):
        return 0
    else:
        return 1


def random_choice(choice_a, choice_b, k):
    # returns choice_a with a probability k%
    if random.uniform(0, 1) < k:
        return choice_a
    else:
        return choice_b


def extract_integer_from_subset_of_bits(number, start_index, length):
    return (number >> start_index) & ((1 << length) - 1)


def set_seed(seed: int = 13) -> None:
    random.seed(seed)
    numpy.random.seed(seed)
    tf.random.set_seed(seed)
    tf.experimental.numpy.random.seed(seed)
    tf.random.set_seed(seed)
    # When running on the CuDNN backend, two further options must be set
    os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    # Set a fixed value for the hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"Random seed set as {seed}")
