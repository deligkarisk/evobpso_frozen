import copy
from unittest import TestCase

from evobpso.utils.algorithm_utils import equalize_sizes
from evobpso.velocity_factor.VelocityFactor import VelocityFactorEvolve, VelocityFactorAdd, VelocityFactorRemove


class Test(TestCase):
    def test__equalize_sizes(self):

        personal_component = [VelocityFactorEvolve(data=0b000000),
                              VelocityFactorAdd(data=0b000000)]
        global_component = [VelocityFactorEvolve(data=0b000000),
                            VelocityFactorEvolve(data=0b000000),
                            VelocityFactorAdd(data=0b000000),
                            VelocityFactorAdd(data=0b000000)
                            ]

        expected_personal_component = copy.deepcopy(personal_component)
        expected_personal_component.append(VelocityFactorRemove())
        expected_personal_component.append(VelocityFactorRemove())
        expected_global_component = copy.deepcopy(global_component)

        updated_personal_component, updated_global_component = equalize_sizes(personal_component, global_component)

        assert expected_global_component == updated_global_component
        assert expected_personal_component == updated_personal_component