import copy

from evaluator.Evaluator import Evaluator
from initializer.Initializer import Initializer
from params.OptimizationParams import OptimizationParams
from position_update_strategy.PositionUpdateStrategy import PositionUpdateStrategy
from position_validator.PositionValidator import PositionValidator
from velocity_update_strategy.VelocityUpdateStrategy import VelocityUpdateStrategy


class Particle:

    def __init__(self, parent_pop, params: OptimizationParams, validator: PositionValidator, initializer: Initializer, evaluator: Evaluator,
                 velocity_update_strategy: VelocityUpdateStrategy,
                 position_update_strategy: PositionUpdateStrategy):
        self.current_result = None
        self.current_position = None
        self.current_velocity = None
        self.result_history = []
        self.position_history = []
        self.parent_pop = parent_pop
        self.params = params
        self.validator = validator
        self.initializer = initializer
        self.evaluator = evaluator
        self.velocity_update_strategy = velocity_update_strategy
        self.position_update_strategy = position_update_strategy

    def iterate(self, first_iter, save_model_folder=None):

        if first_iter:
            self.current_position = self._get_initial_positions()
            self.current_velocity = []  # past velocity information is not used, so no need to initialize here
            current_result = self._evaluate_position(save_model_folder)
            self.result_history.append(current_result)
            self.position_history.append(copy.deepcopy(self.current_position))
            self.current_result = current_result
            self._set_current_position_to_pbest()
        else:
            self.current_velocity = self._get_new_velocity()
            self.current_position = self._get_new_position()
            current_result = self._evaluate_position(save_model_folder)
            self.result_history.append(current_result)
            self.position_history.append(copy.deepcopy(self.current_position))
            self.current_result = current_result
            self._update_personal_best()
        return current_result

    def _get_initial_positions(self):
        position = self.initializer.get_initial_position()
        return position

    def _evaluate_position(self, save_model_folder):
        fitness_value = self.evaluator.evaluate_for_train(self.current_position, save_model_folder=None)
        return fitness_value

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
        new_position = self.position_update_strategy.get_new_position(self.current_position, self.current_velocity)
        post_validation_position = self.validator.validate(new_position, self.params)
        return post_validation_position

