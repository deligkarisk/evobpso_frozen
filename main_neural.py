from Population import Population
from pso_params.PsoParams import NeuralBPSOParams
from Rastrigin import Rastrigin
from Decoder import BinToRealDecoder
from neural_component_to_position_visitor.ComponentToPositionStandardVisitor import ComponentToPositionStandardVisitor
from particle_factory.ParticleFactory import NeuralBPSOParticleFactory
from position_update_strategy.NeuralBPSOPositionUpdateStrategy import NeuralBPSOStandardPositionUpdateStrategy
from velocity_strategy.NeuralBPSOVelocityStrategy import NeuralBPSOStandardVelocityStrategy

n_bits = 32

params = NeuralBPSOParams(0.3, 0.3, 0.1, n_bits)
velocity_strategy = NeuralBPSOStandardVelocityStrategy()
position_component_visitor = ComponentToPositionStandardVisitor()
position_update_strategy = NeuralBPSOStandardPositionUpdateStrategy(component_to_position_visitor=position_component_visitor)
factory = NeuralBPSOParticleFactory()

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
    particle = parent_pop_bool.get_best_particle()
    print("global best: velocity: " + str(particle.current_velocity) + ", position: " + str(particle.current_position) + ", result: " + str(particle.current_result))


