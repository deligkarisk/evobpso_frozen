import copy
import random


class Particle:

    def __init__(self, problem, n_bits, parent_pop):

        self.parent_pop = parent_pop

        self.c1 = 0.3
        self.c2 = 0.3
        self.omega = 0.01
        self.n_bits = n_bits

        self.current_position = []
        for i in range(0, problem.dimensions):
            self.current_position.append(self.create_rnd_binary_vector(0.5))

        self.current_velocity = []
        for i in range(0, problem.dimensions):
            self.current_velocity.append(self.create_rnd_binary_vector(0.5))

        self.problem = problem
        self.evaluate_position()
        self.personal_best_result = copy.deepcopy(self.current_result)
        self.personal_best_position = copy.deepcopy(self.current_position)

    def iterate(self):
        self.update_velocity()
        self.update_position()
        self.evaluate_position()
        self.update_personal_best()

    def update_velocity(self):

        new_velocity = [self.create_rnd_binary_vector(self.omega) & current_vel | (self.create_rnd_binary_vector(self.c1) & (pbest ^ current_pos)) |
                        (self.create_rnd_binary_vector(self.c2) & (gbest ^ current_pos)) for
                        (current_vel, current_pos, pbest, gbest) in
                        zip(self.current_velocity, self.current_position, self.personal_best_position,
                            self.parent_pop.global_best_position)]

        if max(new_velocity) > 2**self.n_bits - 1:
           pass
           # print("Velocity overflow")

        if (min(new_velocity)) == 0:
          pass
          #  print("velocity zero")

        self.current_velocity = new_velocity

    def create_rnd_binary_vector(self, prob):
        result = 0

        for x in range(0, self.n_bits):
            x_rnd = random.random()
            if x_rnd < prob:
                bit_mask = 1 << x
                result = result | bit_mask
        return result

    def update_position(self):
        self.current_position = [current_position ^ current_velocity for (current_position, current_velocity) in
                                 zip(self.current_position, self.current_velocity)]

    def evaluate_position(self):
        self.current_result = self.problem.evaluate(self.current_position)

    def update_personal_best(self):
        if self.current_result < self.personal_best_result:
            self.set_current_position_to_pbest()

    def set_current_position_to_pbest(self):
        self.personal_best_result = copy.deepcopy(self.current_result)
        self.personal_best_position = copy.deepcopy(self.current_position)
