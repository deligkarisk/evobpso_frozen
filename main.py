from Population import Population
from PsoParams import PsoParams
from Rastrigin import Rastrigin
from Decoder import BinToRealDecoder, RealToRealDecoder
from velocity_strategy.BooleanPSOVelocityStrategy import BooleanPSOStandardVelocityStrategy, RealPSOVelocityStrategy

n_bits = 32

params = PsoParams(0.3, 0.3, 0.1, n_bits)
velocity_strategy = BooleanPSOStandardVelocityStrategy()

decoder = BinToRealDecoder(n_bits)

problem = Rastrigin(dimensions=2)
parent_pop_bool = Population(20, problem, decoder, params, velocity_strategy, particle_type='boolean')


for i in range(0,10):
    parent_pop_bool.iterate()
    #print(parent_pop.global_best_result)
    print("iteration: " + str(i))
    particle = parent_pop_bool.particles[0]
    print("particle 0: velocity_strategy: " + str(particle.current_velocity) + ", position: " + str(particle.current_position) + ", result: " + str(particle.current_result))
    particle = parent_pop_bool.particles[1]
    print("particle 1: velocity_strategy: " + str(particle.current_velocity) + ", position: " + str(particle.current_position) + ", result: " + str(particle.current_result))
    particle = parent_pop_bool.get_best_particle()
    print("global best: velocity_strategy: " + str(particle.current_velocity) + ", position: " + str(particle.current_position) + ", result: " + str(particle.current_result))





velocity_strategy = RealPSOVelocityStrategy()
decoder = RealToRealDecoder()
parent_pop_real = Population(20, problem, decoder, params, velocity_strategy, particle_type='real')

for i in range(0,100):
    parent_pop_real.iterate()
    #print(parent_pop.global_best_result)
    print("iteration: " + str(i))
    particle = parent_pop_real.particles[0]
    print("particle 0: velocity_strategy: " + str(particle.current_velocity) + ", position: " + str(particle.current_position) + ", result: " + str(particle.current_result))
    particle = parent_pop_real.particles[1]
    print("particle 1: velocity_strategy: " + str(particle.current_velocity) + ", position: " + str(particle.current_position) + ", result: " + str(particle.current_result))
    particle = parent_pop_real.get_best_particle()
    print("global best: velocity_strategy: " + str(particle.current_velocity) + ", position: " + str(particle.current_position) + ", result: " + str(particle.current_result))


