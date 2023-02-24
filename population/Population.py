import copy

from architecture_decoder.ArchitectureDecoder import ArchitectureDecoder
from evaluator.Evaluator import Evaluator
from initializer.Initializer import Initializer
from params.OptimizationParams import OptimizationParams
from particle.Particle import Particle
from position_update_strategy.PositionUpdateStrategy import PositionUpdateStrategy
from position_validator.PositionValidator import PositionValidator
from velocity_update_strategy.VelocityUpdateStrategy import VelocityUpdateStrategy


class Population:

    def __init__(self, pop_size, params: OptimizationParams, validator: PositionValidator, initializer: Initializer,
                 evaluator: Evaluator,
                 velocity_update_strategy: VelocityUpdateStrategy,
                 position_update_strategy: PositionUpdateStrategy):

        self.particles = []

        for i in range(0, pop_size):
            self.particles.append(
                Particle(self, params, validator, initializer, evaluator, velocity_update_strategy, position_update_strategy))

    def iterate(self, first_iter):
        current_iteration_results = {}
        if first_iter:
            for i in range(0, len(self.particles)):
                particle_evaluation_data = self.particles[i].iterate(first_iter=True)
                current_iteration_results['Particle_' + str(i)] = particle_evaluation_data
            best_particle = self._get_best_particle()
            self._set_particle_as_global_best(best_particle)
        else:
            for i in range(0, len(self.particles)):
                particle_evaluation_data = self.particles[i].iterate(first_iter=False)
                current_iteration_results['Particle_' + str(i)] = particle_evaluation_data
            self._update_pop_best()
        return current_iteration_results

    def _update_pop_best(self):
        best_particle = self._get_best_particle()
        if best_particle.personal_best_result < self.global_best_result:
            self._set_particle_as_global_best(best_particle)

    def _set_particle_as_global_best(self, particle):
        self.global_best_position = copy.deepcopy(particle.personal_best_position)
        self.global_best_result = copy.deepcopy(particle.personal_best_result)

    def _get_best_particle(self):
        best_particle = sorted(self.particles, key=lambda x: x.personal_best_result)[0]
        return best_particle
