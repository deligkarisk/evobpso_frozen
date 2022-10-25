import copy

from Particle import Particle, BooleanPSOParticle


class Population:

    def __init__(self, pop_size, problem, decoder, pso_params, velocity_strategy):

        self.particles = []

        for id in range(0, pop_size):
            self.particles.append(BooleanPSOParticle(self, problem, decoder, pso_params, velocity_strategy))

        best_particle = self.get_best_particle()
        self.global_best_position = copy.deepcopy(best_particle.personal_best_position)
        self.global_best_result = copy.deepcopy(best_particle.personal_best_result)

    def iterate(self):
        for particle in self.particles:
            particle.iterate()
        self.update_pop_best()

    def update_pop_best(self):
        best_particle = self.get_best_particle()
        if best_particle.personal_best_result < self.global_best_result:
            print("new global best found " + str(best_particle.personal_best_result))
            self.copy_particle_to_population_best(best_particle)

    def copy_particle_to_population_best(self, particle):
        self.global_best_position = copy.deepcopy(particle.personal_best_position)
        self.global_best_result = copy.deepcopy(particle.personal_best_result)

    def get_best_particle(self):
        best_particle = sorted(self.particles, key=lambda x: x.personal_best_result)[0]
        return best_particle
