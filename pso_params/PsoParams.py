import abc


class PsoParams(abc.ABC):
   pass


class StandardPsoParams(PsoParams):
    def __init__(self, c1, c2, omega, n_bits):
        self.n_bits = n_bits
        self.c1 = c1
        self.c2 = c2
        self.omega = omega



class NeuralBPSOParams(PsoParams):
    def __init__(self, c1, c2, omega, n_bits, k):
        self.n_bits = n_bits
        self.c1 = c1
        self.c2 = c2
        self.omega = omega
        self.k = k