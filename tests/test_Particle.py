from unittest import TestCase

from Decoder import BinToRealDecoder
from Particle import Particle
from Population import Population
from Rastrigin import Rastrigin


class TestParticle(TestCase):
    def test_update_velocity_personal_only(self):
        nbits = 6
        dimensions = 1

        decoder = BinToRealDecoder(nbits)
        problem = Rastrigin(dimensions=dimensions)

        parent_pop = Population(pop_size=1, problem=problem, decoder=decoder, nbits=nbits)

        particle = Particle(problem, nbits, parent_pop, decoder)

        particle.current_velocity = [0b000000]
        particle.current_position = [0b000000]
        particle.personal_best_position = [0b111111]
        particle.parent_pop.global_best_position = [0b010101]
        particle.c1 = 1
        particle.c2 = 0
        particle.omega = 0

        particle.update_velocity()
        assert particle.current_velocity == [0b111111]

    def test_update_velocity_global_only(self):
        nbits = 6
        dimensions = 1

        decoder = BinToRealDecoder(nbits)
        problem = Rastrigin(dimensions=dimensions)

        parent_pop = Population(pop_size=1, problem=problem, decoder=decoder, nbits=nbits)

        particle = Particle(problem, nbits, parent_pop, decoder)

        particle.current_velocity = [0b000000]
        particle.current_position = [0b000000]
        particle.personal_best_position = [0b111111]
        particle.parent_pop.global_best_position = [0b010101]
        particle.c1 = 0
        particle.c2 = 1
        particle.omega = 0

        particle.update_velocity()
        assert particle.current_velocity == [0b010101]




    def test_create_rnd_binary_vector(self):
        nbits = 6
        dimensions = 1

        decoder = BinToRealDecoder(nbits)
        problem = Rastrigin(dimensions=dimensions)


        parent_pop = Population(pop_size=1, problem=problem, decoder=decoder, nbits=nbits)

        particle = Particle(problem, nbits, parent_pop, decoder)
        assert particle.create_rnd_binary_vector(1) == 0b111111
        assert particle.create_rnd_binary_vector(0) == 0b000000

        nbits = 12
        dimensions=3
        problem = Rastrigin(dimensions=dimensions)
        particle = Particle(problem, nbits, parent_pop, decoder=decoder)
        assert particle.create_rnd_binary_vector(1) == 0b111111111111

