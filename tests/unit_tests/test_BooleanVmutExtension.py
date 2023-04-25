from unittest import TestCase
from unittest.mock import patch, Mock

from evobpso.velocity_factor.VelocityFactor import VelocityFactorAdd, VelocityFactorEvolve
from evobpso.velocity_update_extension.BooleanVmutExtension import BooleanVmutExtension


class TestBooleanVmutExtension(TestCase):


    @patch('evobpso.velocity_update_extension.BooleanVmutExtension.create_rnd_binary_vector', return_value=0b100000)
    def test_get_new_velocity(self, mock_rnd_binary_vector_creator):
        # exact values of the position are irrelevant as we specify the return value of the super's get_new_velocity() a few lines below.

        base_velocity = []
        base_velocity.append(VelocityFactorEvolve(data=0b010101))
        base_velocity.append(VelocityFactorEvolve(data=0b000000))
        base_velocity.append(VelocityFactorAdd(data=0b000111))

        extension = BooleanVmutExtension(params=Mock())
        new_velocity = extension.get_new_velocity(base_velocity)
        expected_velocity = []
        expected_velocity.append(VelocityFactorEvolve(data=0b110101))  # same as the base velocity , but with the left-most bit now set
        # to 1 (due to the create_rnd_binary_vector return value being set)
        expected_velocity.append(VelocityFactorEvolve(data=0b100000))  # same as the base velocity, but with the left-most bit now set to
        # 1 (due to the create_rnd_binary_vector return value being set)
        expected_velocity.append(VelocityFactorAdd(data=0b100111))  # same as the base velocity, but with the left-most bit now set to 1
        # (due to the create_rnd_binary_vector return value being set)
        assert new_velocity == expected_velocity


