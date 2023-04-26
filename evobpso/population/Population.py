import copy

from evobpso.particle.Particle import Particle
from evobpso.scheme.Scheme import Scheme
from evobpso.utils.filesystem_utils import get_save_folder


class Population:

    def __init__(self, scheme: Scheme):

        self.particles = []
        self.global_best_position = None
        self.global_best_result = None
        self.global_best_result_history = []
        self.iter_number = -1  # used to track internally the iter number
        self.results_folder = scheme.results_folder
        self.pop_size = scheme.optimization_params.pso_params.pop_size
        self.scheme = scheme

        for i in range(0, self.pop_size):
            self.particles.append(
                Particle(self, params=scheme.optimization_params,
                         position_validator=scheme.position_validator,
                         initializer=scheme.initializer,
                         evaluator= scheme.evaluator,
                         velocity_update_strategy=scheme.velocity_update_strategy,
                         position_update_strategy=scheme.position_update_strategy))

    def iterate(self, first_iter):
        current_iteration_results = {}
        self.iter_number += 1
        if first_iter:
            for i in range(0, len(self.particles)):
                save_folder = get_save_folder(self.results_folder, self.iter_number, i)
                particle_evaluation_data = self.particles[i].iterate(first_iter=True, save_model_folder=save_folder)
                current_iteration_results['Particle_' + str(i)] = particle_evaluation_data
            best_particle = self._get_best_particle()
            self._set_particle_as_global_best(best_particle)
        else:
            for i in range(0, len(self.particles)):
                save_folder = get_save_folder(self.results_folder, self.iter_number, i)
                particle_evaluation_data = self.particles[i].iterate(first_iter=False, save_model_folder=save_folder)
                current_iteration_results['Particle_' + str(i)] = particle_evaluation_data
            self._update_pop_best()
        self.global_best_result_history.append(copy.deepcopy(self.global_best_result))
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

