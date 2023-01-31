from particle.Particle import Particle
from position_update_strategy.NeuralBooleanPSOPositionUpdateStrategy import NeuralBooleanPSOPositionUpdateStrategy
from pso_params.PsoParams import NeuralBooleanPSOParams
from utils.utils import create_rnd_binary_vector
from velocity_update_strategy.NeuralBooleanPSOVelocityUpdateStrategy import NeuralBooleanPSOVelocityUpdateStrategy


class NeuralBooleanPSOParticle(Particle):

    def __init__(self, parent_pop, problem, decoder, pso_params: NeuralBooleanPSOParams, velocity_strategy: NeuralBooleanPSOVelocityUpdateStrategy,
                 position_update_strategy: NeuralBooleanPSOPositionUpdateStrategy):

        if not isinstance(position_update_strategy, NeuralBooleanPSOPositionUpdateStrategy):
            raise ValueError("Position update strategy in Neural BPSO must be " + str(NeuralBooleanPSOPositionUpdateStrategy.__name__))

        if not isinstance(velocity_strategy, NeuralBooleanPSOVelocityUpdateStrategy):
            raise ValueError("Velocity update strategy in Neural BPSO must be " + str(NeuralBooleanPSOVelocityUpdateStrategy.__name__))

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