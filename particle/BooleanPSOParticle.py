from particle.Particle import Particle
from position_update_strategy.BooleanPSOPositionUpdateStrategy import BooleanPSOPositionUpdateStrategy
from pso_params import PsoParams
from utils.utils import create_rnd_binary_vector
from velocity_update_strategy.BooleanPSOVelocityUpdateStrategy import BooleanPSOVelocityStrategy


class BooleanPSOParticle(Particle):

    def __init__(self, parent_pop, problem, decoder, pso_params: PsoParams, velocity_strategy: BooleanPSOVelocityStrategy,
                 position_update_strategy: BooleanPSOPositionUpdateStrategy):

        if not isinstance(position_update_strategy, BooleanPSOPositionUpdateStrategy):
            raise ValueError("Position update strategy in Boolean PSO must be " + str(BooleanPSOPositionUpdateStrategy.__name__))

        if not isinstance(velocity_strategy, BooleanPSOVelocityStrategy):
            raise ValueError("Velocity update strategy in Boolean PSO must be " + str(BooleanPSOVelocityStrategy.__name__))

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
