import random

def create_rnd_binary_vector(prob, n_bits):
    result = 0

    for x in range(0, n_bits):
        x_rnd = random.random()
        if x_rnd < prob:
            bit_mask = 1 << x
            result = result | bit_mask
    return result