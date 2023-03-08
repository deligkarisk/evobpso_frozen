import abc


class PsoParams(abc.ABC):
    pass

class BooleanPSOParams(PsoParams):
    def __init__(self, pop_size, iters, c1, c2, n_bits, k, mutation_prob):
        self.pop_size = pop_size
        self.iters = iters
        self.n_bits = n_bits
        self.c1 = c1
        self.c2 = c2
        self.k = k  # The probability of selecting the personal factor when either the personal or global component is Add or Remove
        # (during velocity update)
        self.mutation_prob = mutation_prob


class RealPSOParams(PsoParams):
    def __init__(self, c1, c2, omega, min_value, max_value, k):
        self.c1 = c1
        self.c2 = c2
        self.omega = omega
        self.min_value = min_value
        self.max_value = max_value
        self.k = k  # The probability of selecting the personal factor when either the personal or global component is Add or Remove
        # (during velocity update)
