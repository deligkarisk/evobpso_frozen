import abc
import copy
import random

from PsoParams import PsoParams
from position_update_strategy.BooleanPSOPositionUpdateStrategy import BooleanPSOPositionUpdateStrategy
from position_update_strategy.NeuralBPSOPositionUpdateStrategy import NeuralBPSOPositionUpdateStrategy
from position_update_strategy.PositionUpdateStrategy import PositionUpdateStrategy
from position_update_strategy.RealPSOPositionUpdateStrategy import RealPSOPositionUpdateStrategy
from velocity_strategy.BooleanPSOVelocityStrategy import BooleanPSOVelocityStrategy
from velocity_strategy.NeuralBPSOVelocityStrategy import NeuralBPSOVelocityStrategy
from utils import create_rnd_binary_vector
from velocity_strategy.RealPSOVelocityStrategy import RealPSOVelocityStrategy
from velocity_strategy.VelocityStrategy import VelocityStrategy


class Particle(abc.ABC):

    def __init__(self, parent_pop, problem, decoder, pso_params: PsoParams, velocity_strategy: VelocityStrategy,
                 position_update_strategy: PositionUpdateStrategy):
        # decoder is a required decoder that implements the method decode(). It is used to convert the positions from
        # binary representation of bPSO to what is required by the Problem instance. If e.g. there is no conversion
        # needed then the converter can return its input as it is.

        self.parent_pop = parent_pop
        self.decoder = decoder
        self.params = pso_params
        self.velocity_strategy = velocity_strategy
        self.position_update_strategy = position_update_strategy
        self.problem = problem

        self.current_position = self.get_initial_positions()
        self.current_velocity = self.get_initial_velocity()

        self.current_result = self.get_current_position_result()
        self.personal_best_result = copy.deepcopy(self.current_result)
        self.personal_best_position = copy.deepcopy(self.current_position)

    def get_initial_positions(self):
        raise NotImplementedError

    def get_initial_velocity(self):
        raise NotImplementedError

    def iterate(self):
        self.current_velocity = self.get_new_velocity()
        self.current_position = self.get_new_position()
        self.current_result = self.evaluate_position()
        self.update_personal_best()

    def evaluate_position(self):
        return self.get_current_position_result()

    def get_current_position_result(self):
        decoded_position = self.decoder.decode(self.problem, self.current_position)
        return self.problem.evaluate(decoded_position)

    def update_personal_best(self):
        if self.current_result < self.personal_best_result:
            self.set_current_position_to_pbest()

    def set_current_position_to_pbest(self):
        self.personal_best_result = copy.deepcopy(self.current_result)
        self.personal_best_position = copy.deepcopy(self.current_position)

    # velocity_strategy is chosen based on strategy pattern
    def get_new_velocity(self):
        return self.velocity_strategy.get_new_velocity(
            self.current_velocity, self.current_position, self.personal_best_position, self.parent_pop.global_best_position, self.params)

    # position update is chosen based on the strategy pattern
    def get_new_position(self):
        return self.position_update_strategy.get_new_position(self.current_position, self.current_velocity)


class BooleanPSOParticle(Particle):

    def __init__(self, parent_pop, problem, decoder, pso_params: PsoParams, velocity_strategy: NeuralBPSOVelocityStrategy,
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


class NeuralBPSOParticle(Particle):

    def __init__(self, parent_pop, problem, decoder, pso_params: PsoParams, velocity_strategy: NeuralBPSOVelocityStrategy,
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