import abc

from evobpso.encoding.BooleanEncoding import BooleanEncoding
from evobpso.encoding.RealEncoding import RealEncoding


class PsoParams(abc.ABC):
    pass

class BooleanPSOParams(PsoParams):
    def __init__(self, pop_size, iters, c1, c2, k, encoding: BooleanEncoding, mutation_prob=0, vmax=2):
        self.pop_size = pop_size
        self.iters = iters
        self.n_bits = encoding.total_bits
        self.c1 = c1
        self.c2 = c2
        self.k = k  # The probability of selecting the personal factor when either the personal or global component is Add or Remove
        # (during velocity update)
        self.mutation_prob = mutation_prob
        self.vmax = vmax


class RealPSOParams(PsoParams):
    def __init__(self, pop_size, iters, c1, c2, k):
        self.pop_size = pop_size
        self.iters = iters
        self.c1 = c1
        self.c2 = c2
        self.k = k  # The probability of selecting the personal factor when either the personal or global component is Add or Remove
        # (during velocity update)
