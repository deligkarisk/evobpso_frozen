import copy

from decoder.Decoder import Decoder
from evaluator.Evaluator import Evaluator
from initializer.Initializer import Initializer
from params.Params import Params
from particle.Particle import Particle
from position_update_strategy.PositionUpdateStrategy import PositionUpdateStrategy
from position_validator.PositionValidator import PositionValidator
from velocity_update_strategy.VelocityUpdateStrategy import VelocityUpdateStrategy


class Population:

    def __init__(self, pop_size, params: Params, validator: PositionValidator, initializer: Initializer,
                 evaluator: Evaluator,
                 velocity_update_strategy: VelocityUpdateStrategy,
                 position_update_strategy: PositionUpdateStrategy):

        self.particles = []

        for i in range(0, pop_size):
            self.particles.append(
                Particle(self, params, validator, initializer, evaluator, velocity_update_strategy, position_update_strategy))

        best_particle = self._get_best_particle()
        self.global_best_position = copy.deepcopy(best_particle.personal_best_position)
        self.global_best_result = copy.deepcopy(best_particle.personal_best_result)

    def iterate(self):
        for particle in self.particles:
            particle.iterate()
        self._update_pop_best()

    def _update_pop_best(self):
        best_particle = self._get_best_particle()
        if best_particle.personal_best_result < self.global_best_result:
            self.global_best_position = copy.deepcopy(best_particle.personal_best_position)
            self.global_best_result = copy.deepcopy(best_particle.personal_best_result)

    def _get_best_particle(self):
        best_particle = sorted(self.particles, key=lambda x: x.personal_best_result)[0]
        return best_particle
