import copy
import random


class Particle:

    def __init__(self, problem, dimensions, n_bits, parent_pop):

        self.parent_pop = parent_pop

        self.c1 = 0.1
        self.c2 = 0.1
        self.omega = 0.1
        self.n_bits = n_bits

        self.current_position = []
        for i in range(0, dimensions - 1):
            self.current_position.append(self.create_rnd_binary_vector(0.5))

        self.current_velocity = []
        for i in range(0, dimensions - 1):
            self.current_velocity.append(self.create_rnd_binary_vector(0.5))

        self.problem = problem
        self.current_result = self.evaluate_position()
        self.personal_best_result = copy.deepcopy(self.current_result)
        self.personal_best_position = copy.deepcopy(self.current_position)

    def iterate(self):
        self.update_velocity()
        self.update_position()
        self.evaluate_position()
        self.update_personal_best()

    def update_velocity(self):

        new_velocity = [self.omega * current_vel + c1 & (pbest ^ current_pos) + c2 & (gbest ^ current_pos) for
                        (current_vel, current_pos, pbest, gbest, c1, c2) in
                        (self.current_velocity, self.current_position, self.personal_best_position,
                         self.parent_pop.global_best_position, self.create_rnd_binary_vector(self.c1), self.create_rnd_binary_vector(self.c2))]

        self.current_velocity = new_velocity

    def create_rnd_binary_vector(self, prob):
        result = 0

        for x in range(0, self.n_bits - 1):
            x_rnd = random.random()
            if x_rnd < prob:
                bit_mask = 1 << x
                result = result | bit_mask
        return result

    def update_position(self):
        self.current_position = [current_position ^ current_velocity for (current_position, current_velocity) in
                                 (self.current_position, self.current_velocity)]

    def evaluate_position(self):
        result = self.problem.evaluate(self.current_position)
        return result

    def update_personal_best(self):
        if self.problem.minimize:
            if self.current_result < self.personal_best_result:
                self.set_current_position_to_pbest()
        else:
            if self.current_result > self.personal_best_result:
                self.set_current_position_to_pbest()

    def set_current_position_to_pbest(self):
        self.personal_best_result = copy.deepcopy(self.current_result)
        self.personal_best_position = copy.deepcopy(self.current_position)
