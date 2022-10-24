import copy
import random

from VelocityStrategy import VelocityStrategy
from utils import create_rnd_binary_vector


class Particle:

    def __init__(self, problem, n_bits, parent_pop, decoder, pso_params, velocity_strategy: VelocityStrategy):
        # decoder is a required decoder that implements the method decode(). It is used to convert the positions from
        # binary representation of bPSO to what is required by the Problem instance. If e.g. there is no conversion
        # needed then the converter can return its input as it is.

        self.parent_pop = parent_pop
        self.decoder = decoder
        self.params = pso_params
        self.velocity_strategy = velocity_strategy

        self.n_bits = n_bits

        self.current_position = []
        for i in range(0, problem.dimensions):
            self.current_position.append(create_rnd_binary_vector(0.5,  self.n_bits))

        self.current_velocity = []
        for i in range(0, problem.dimensions):
            self.current_velocity.append(create_rnd_binary_vector(0.5,  self.n_bits))

        self.problem = problem
        self.current_result = self.get_current_position_result()
        self.personal_best_result = copy.deepcopy(self.current_result)
        self.personal_best_position = copy.deepcopy(self.current_position)

    def iterate(self):
        self.update_velocity()
        self.update_position()
        self.evaluate_position()
        self.update_personal_best()

    def update_velocity(self):
        self.current_velocity = self.velocity_strategy.update_velocity(
            self.current_velocity, self.current_position, self.personal_best_position, self.parent_pop.global_best_position, self.params, self.n_bits)



    def update_position(self):
        self.current_position = [current_position ^ current_velocity for (current_position, current_velocity) in
                                 zip(self.current_position, self.current_velocity)]

    def evaluate_position(self):
        self.current_result = self.get_current_position_result()

    def get_current_position_result(self):
        decoded_position = self.decoder.decode(self.problem, self.current_position)
        return self.problem.evaluate(decoded_position)

    def update_personal_best(self):
        if self.current_result < self.personal_best_result:
            self.set_current_position_to_pbest()

    def set_current_position_to_pbest(self):
        self.personal_best_result = copy.deepcopy(self.current_result)
        self.personal_best_position = copy.deepcopy(self.current_position)
