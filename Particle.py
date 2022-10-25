import abc
import copy
import random

from PsoParams import PsoParams
from VelocityStrategy import VelocityStrategy
from utils import create_rnd_binary_vector


class Particle(abc.ABC):

    def __init__(self, parent_pop, problem, decoder, pso_params: PsoParams, velocity_strategy: VelocityStrategy):
        # decoder is a required decoder that implements the method decode(). It is used to convert the positions from
        # binary representation of bPSO to what is required by the Problem instance. If e.g. there is no conversion
        # needed then the converter can return its input as it is.

        self.parent_pop = parent_pop
        self.decoder = decoder
        self.params = pso_params
        self.velocity_strategy = velocity_strategy
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

    def get_new_position(self):
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

    # velocity is chosen based on strategy pattern
    def get_new_velocity(self):
        return self.velocity_strategy.update_velocity(
            self.current_velocity, self.current_position, self.personal_best_position, self.parent_pop.global_best_position, self.params)


class BooleanPSOParticle(Particle):

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

    def get_new_position(self):
        new_position = [current_position ^ current_velocity for (current_position, current_velocity) in
                        zip(self.current_position, self.current_velocity)]
        return new_position
