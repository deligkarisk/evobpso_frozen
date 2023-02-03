from population.Population import Population
from position_update_strategy.NeuralRealPSOPositionUpdateStrategy import NeuralRealPSOStandardPositionUpdateStrategy
from params.PsoParams import BooleanPSOParams, RealPSOParams
from problem.Rastrigin import Rastrigin
from decoder.Decoder import BinToRealDecoder
from position_update_strategy.NeuralBooleanPSOPositionUpdateStrategy import NeuralBooleanPSOStandardPositionUpdateStrategy
from velocity_update_strategy.NeuralBooleanPSOVelocityUpdateStrategy import NeuralBooleanPSOStandardVelocityUpdateStrategy
from velocity_update_strategy.NeuralRealPSOVelocityUpdateStrategy import NeuralRealPSOStandardVelocityUpdateStrategy

# This is an example of how to use the standard implementation of the neural real pso

n_bits = 32

params = RealPSOParams()
velocity_strategy = NeuralRealPSOStandardVelocityUpdateStrategy(params)
position_update_strategy = NeuralRealPSOStandardPositionUpdateStrategy()

decoder = BinToRealDecoder(n_bits)

problem = Rastrigin(dimensions=2)
parent_pop_bool = Population(20, problem, decoder, params, velocity_strategy, position_update_strategy, factory)

print("--------------------------------- BOOLEAN PSO ---------------------------------")

for i in range(0,100):
    parent_pop_bool.iterate()
    #print(parent_pop.global_best_result)
    print("iteration: " + str(i))
    particle = parent_pop_bool.particles[0]
    print("particle 0: velocity: " + str(particle.current_velocity) + ", position: " + str(particle.current_position) + ", result: " + str(particle.current_result))
    particle = parent_pop_bool.particles[1]
    print("particle 1: velocity: " + str(particle.current_velocity) + ", position: " + str(particle.current_position) + ", result: " + str(particle.current_result))
    particle = parent_pop_bool._get_best_particle()
    print("global best: velocity: " + str(particle.current_velocity) + ", position: " + str(particle.current_position) + ", result: " + str(particle.current_result))


