from Particle import Particle
from Population import Population
from Rastrigin import Rastrigin
from Decoder import BinToRealDecoder

n_bits = 32

decoder = BinToRealDecoder(n_bits)

problem = Rastrigin(steps=2**32 - 1, dimensions=2)
parent_pop = Population(20, problem, n_bits, decoder)


for i in range(0,1000):
    parent_pop.iterate()
    #print(parent_pop.global_best_result)
    print(parent_pop.particles[0].current_position, parent_pop.particles[0].current_result, parent_pop.global_best_result)
