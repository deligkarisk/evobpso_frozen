from Particle import Particle
from Population import Population
from PsoParams import PsoParams
from Rastrigin import Rastrigin
from Decoder import BinToRealDecoder
from VelocityStrategy import BooleanPSOStandardVelocityStrategy

n_bits = 32

params = PsoParams(0.3, 0.3, 0.1)
velocity_strategy = BooleanPSOStandardVelocityStrategy()

decoder = BinToRealDecoder(n_bits)

problem = Rastrigin(dimensions=2)
parent_pop = Population(20, problem, n_bits, decoder, params, velocity_strategy)


for i in range(0,1000):
    parent_pop.iterate()
    #print(parent_pop.global_best_result)
    print("iteration: " + str(i))
    particle = parent_pop.particles[0]
    print("particle 0: velocity: " + str(particle.current_velocity) + ", position: " + str(particle.current_position) + ", result: " + str(particle.current_result))
    particle = parent_pop.particles[1]
    print("particle 1: velocity: " + str(particle.current_velocity) + ", position: " + str(particle.current_position) + ", result: " + str(particle.current_result))
    particle = parent_pop.get_best_particle()
    print("global best: velocity: " + str(particle.current_velocity) + ", position: " + str(particle.current_position) + ", result: " + str(particle.current_result))





