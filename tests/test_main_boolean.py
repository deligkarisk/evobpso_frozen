from decoder.PositionToNNDecoder import PositionToNNDecoder
from evaluator.StandardNNEvaluator import StandardNNEvaluator
from initializer.BinaryInitializer import BinaryInitializer
from params.Params import Params
from population.Population import Population
from params.NeuralArchitectureParams import NeuralArchitectureParams
from params.PsoParams import BooleanPSOParams
from problem.Rastrigin import Rastrigin
from position_update_strategy.NeuralBooleanPSOPositionUpdateStrategy import NeuralBooleanPSOStandardPositionUpdateStrategy
from validator.DoNothingValidator import DoNothingValidator
from velocity_update_strategy.NeuralBooleanPSOVelocityUpdateStrategy import NeuralBooleanPSOStandardVelocityUpdateStrategy

# This is an example of how to use the standard implementation of the neural boolean pso

pso_params = BooleanPSOParams(c1=0.3, c2=0.3, omega=0.1, n_bits=32, k=0.5)
architecture = NeuralArchitectureParams(min_out_conv=2, max_out_conv=4,
                                        min_kernel_conv=2, max_kernel_conv=4,
                                        min_layers=10, max_layers=20)
all_params = Params(pso_params=pso_params, architecture_params=architecture)
decoder = PositionToNNDecoder()
validator = DoNothingValidator()
initializer = BinaryInitializer(params=all_params)
evaluator = StandardNNEvaluator()
velocity_strategy = NeuralBooleanPSOStandardVelocityUpdateStrategy(pso_params)
position_update_strategy = NeuralBooleanPSOStandardPositionUpdateStrategy()




problem = Rastrigin(dimensions=2)
parent_pop_bool = Population(20, problem, decoder, pso_params, velocity_strategy, position_update_strategy, factory)

print("--------------------------------- BOOLEAN PSO ---------------------------------")

for i in range(0,100):
    parent_pop_bool.iterate()
    #print(parent_pop.global_best_result)
    print("iteration: " + str(i))
    particle = parent_pop_bool.particles[0]
    print("particle 0: velocity: " + str(particle.current_velocity) + ", position: " + str(particle.current_position) + ", result: " + str(particle.current_result))
    particle = parent_pop_bool.particles[1]
    print("particle 1: velocity: " + str(particle.current_velocity) + ", position: " + str(particle.current_position) + ", result: " + str(particle.current_result))
    particle = parent_pop_bool.get_best_particle()
    print("global best: velocity: " + str(particle.current_velocity) + ", position: " + str(particle.current_position) + ", result: " + str(particle.current_result))


