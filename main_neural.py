from Population import Population
from PsoParams import PsoParams
from Rastrigin import Rastrigin
from Decoder import BinToRealDecoder, RealToRealDecoder
from particle_factory.ParticleFactory import BooleanPSOParticleFactory, RealPSOParticleFactory
from position_update_strategy.BooleanPSOPositionUpdateStrategy import BooleanPSOStandardPositionUpdateStrategy
from position_update_strategy.NeuralPositionUpdateStrategy import BooleanPSONeuralPositionUpdateStrategy
from position_update_strategy.RealPSOPositionUpdateStrategy import RealPSOStandardPositionUpdateStrategy
from velocity_strategy.BooleanPSOVelocityStrategy import BooleanPSOStandardVelocityStrategy
from velocity_strategy.NeuralVelocityStrategy import BooleanPSONeuralVelocityStrategy
from velocity_strategy.RealPSOVelocityStrategy import RealPSOStandardVelocityStrategy

n_bits = 32

params = PsoParams(0.3, 0.3, 0.1, n_bits)
velocity_strategy = BooleanPSONeuralVelocityStrategy()
position_update_strategy = BooleanPSONeuralPositionUpdateStrategy()
neural_component_config = ComponentConfigVisitor()



factory = BooleanPSOParticleFactory()

decoder = BinToRealDecoder(n_bits)

problem = Rastrigin(dimensions=2)
parent_pop_bool = Population(20, problem, decoder, params, velocity_strategy, position_update_strategy, factory)

print("--------------------------------- BOOLEAN PSO ---------------------------------")

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



print("--------------------------------- REAL PSO ---------------------------------")

velocity_strategy = RealPSOStandardVelocityStrategy()
position_update_strategy = RealPSOStandardPositionUpdateStrategy()
factory = RealPSOParticleFactory()

decoder = RealToRealDecoder()
parent_pop_real = Population(20, problem, decoder, params, velocity_strategy, position_update_strategy, factory)

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


