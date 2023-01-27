import random
from particle.Particle import Particle
from position_update_strategy.RealPSOPositionUpdateStrategy import RealPSOPositionUpdateStrategy
from pso_params.PsoParams import PsoParams
from velocity_update_strategy.RealPSOVelocityUpdateStrategy import RealPSOVelocityStrategy


class RealPSOParticle(Particle):

    def __init__(self, parent_pop, problem, decoder, pso_params: PsoParams, velocity_strategy: RealPSOVelocityStrategy,
                 position_update_strategy: RealPSOPositionUpdateStrategy):

        if not isinstance(position_update_strategy, RealPSOPositionUpdateStrategy):
            raise ValueError("Position update strategy in Real PSO must be " + str(RealPSOPositionUpdateStrategy.__name__))

        if not isinstance(velocity_strategy, RealPSOVelocityStrategy):
            raise ValueError("Velocity update strategy in Boolean PSO must be " + str(RealPSOVelocityStrategy.__name__))

        Particle.__init__(self, parent_pop, problem, decoder, pso_params, velocity_strategy, position_update_strategy)


    def get_initial_positions(self):
        position = []
        for i in range(0, self.problem.dimensions):
            position.append(random.uniform(self.problem.min_value, self.problem.max_value))
        return position

    def get_initial_velocity(self):
        velocity = []
        for i in range(0, self.problem.dimensions):
            velocity.append(random.uniform(0, 1))
        return velocity

    def get_new_position(self):
        new_position = []
        for i in range(0, self.problem.dimensions):
            new_position.append(self.current_position[i] + self.current_velocity[i])
        return new_position