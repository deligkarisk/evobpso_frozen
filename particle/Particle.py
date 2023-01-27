import abc
import copy
import random

from pso_params.PsoParams import PsoParams, NeuralBPSOParams
from position_update_strategy.BooleanPSOPositionUpdateStrategy import BooleanPSOPositionUpdateStrategy
from position_update_strategy.NeuralBPSOPositionUpdateStrategy import NeuralBPSOPositionUpdateStrategy
from position_update_strategy.PositionUpdateStrategy import PositionUpdateStrategy
from position_update_strategy.RealPSOPositionUpdateStrategy import RealPSOPositionUpdateStrategy
from velocity_update_strategy.BooleanPSOVelocityUpdateStrategy import BooleanPSOVelocityStrategy
from velocity_update_strategy.NeuralBPSOVelocityUpdateStrategy import NeuralBPSOVelocityStrategy
from utils.utils import create_rnd_binary_vector
from velocity_update_strategy.RealPSOVelocityUpdateStrategy import RealPSOVelocityStrategy
from velocity_update_strategy.VelocityUpdateStrategy import VelocityStrategy


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

    # velocity_update_strategy is chosen based on strategy pattern
    def get_new_velocity(self):
        return self.velocity_strategy.get_new_velocity(
            self.current_velocity, self.current_position, self.personal_best_position, self.parent_pop.global_best_position)

    # position update is chosen based on the strategy pattern
    def get_new_position(self):
        return self.position_update_strategy.get_new_position(self.current_position, self.current_velocity)





