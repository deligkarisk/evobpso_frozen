import abc
import copy

from initializer.Initializer import Initializer
from position_update_strategy.PositionUpdateStrategy import PositionUpdateStrategy
from problem.NeuralArchitecture import Problem
from pso_params.PsoParams import PsoParams
from velocity_update_strategy.VelocityUpdateStrategy import VelocityUpdateStrategy


class Particle(abc.ABC):

    def __init__(self, parent_pop, decoder, problem: Problem, initializer: Initializer, pso_params: PsoParams,
                 velocity_update_strategy: VelocityUpdateStrategy,
                 position_update_strategy: PositionUpdateStrategy):
        self.parent_pop = parent_pop
        self.decoder = decoder
        self.problem = problem
        self.initializer = initializer
        self.params = pso_params
        self.velocity_update_strategy = velocity_update_strategy
        self.position_update_strategy = position_update_strategy

        self.current_position = self._get_initial_positions()
        self.current_velocity = []  # past velocity information is not used, so no need to initialize here

        self.current_result = self._evaluate_position()
        self.personal_best_result = copy.deepcopy(self.current_result)
        self.personal_best_position = copy.deepcopy(self.current_position)

    def iterate(self):
        self.current_velocity = self._get_new_velocity()
        self.current_position = self._get_new_position()
        self.current_result = self._evaluate_position()
        self._update_personal_best()

    def _get_initial_positions(self):
        position = self.initializer.get_initial_position(self.params)
        return position

    def _get_initial_velocity(self):
        pass
       # velocity = self.initializer.get_initial_velocity()
       # return velocity

    def _evaluate_position(self):
        decoded_position = self.decoder.decode(self.current_position)
        return self.problem.evaluate(decoded_position)

    def _update_personal_best(self):
        if self.current_result < self.personal_best_result:
            self._set_current_position_to_pbest()

    def _set_current_position_to_pbest(self):
        self.personal_best_result = copy.deepcopy(self.current_result)
        self.personal_best_position = copy.deepcopy(self.current_position)

    # velocity_update_strategy is chosen based on strategy pattern
    def _get_new_velocity(self):
        return self.velocity_update_strategy.get_new_velocity(
            self.current_velocity, self.current_position, self.personal_best_position, self.parent_pop.global_best_position)

    # position update is chosen based on the strategy pattern
    def _get_new_position(self):
        return self.position_update_strategy.get_new_position(self.current_position, self.current_velocity)
