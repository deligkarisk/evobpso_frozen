from particle.Particle import Particle
from position_update_strategy.NeuralBPSOPositionUpdateStrategy import NeuralBPSOPositionUpdateStrategy
from pso_params.PsoParams import NeuralBPSOParams
from utils.utils import create_rnd_binary_vector
from velocity_update_strategy.NeuralBPSOVelocityUpdateStrategy import NeuralBPSOVelocityStrategy


class NeuralBPSOParticle(Particle):

    def __init__(self, parent_pop, problem, decoder, pso_params: NeuralBPSOParams, velocity_strategy: NeuralBPSOVelocityStrategy,
                 position_update_strategy: NeuralBPSOPositionUpdateStrategy):

        if not isinstance(position_update_strategy, NeuralBPSOPositionUpdateStrategy):
            raise ValueError("Position update strategy in Neural BPSO must be " + str(NeuralBPSOPositionUpdateStrategy.__name__))

        if not isinstance(velocity_strategy, NeuralBPSOVelocityStrategy):
            raise ValueError("Velocity update strategy in Neural BPSO must be " + str(NeuralBPSOVelocityStrategy.__name__))

        Particle.__init__(self, parent_pop, problem, decoder, pso_params, velocity_strategy, position_update_strategy)

    def get_initial_positions(self):

        position = []
        for i in range(0, self.problem.dimensions):
            position.append(create_rnd_binary_vector(0.5, self.params.n_bits))
        return position

    def get_initial_velocity(self):

        velocity = []
        for i in range(0, self.problem.dimensions):
            velocity.append(create_rnd_binary_vector(0.5, self.params.n_bits))
        return velocity