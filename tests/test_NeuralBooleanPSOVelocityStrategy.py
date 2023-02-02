import copy
from unittest import TestCase
from unittest.mock import patch, Mock

from params.PsoParams import PsoParams, BooleanPSOParams
from velocity_component.VelocityComponent import VelocityComponentRemove, VelocityComponentAdd, VelocityComponentEvolve
from velocity_update_strategy.NeuralBooleanPSOVelocityUpdateStrategy import NeuralBooleanPSOStandardVelocityUpdateStrategy


class TestNeuralBooleanPSOStandardVelocityStrategy(TestCase):


    def test__create_component_when_best_position_longer_than_current(self):
        # This vector is composed of 1s, and it is necessary to set this for testing, in normal operations a rnd vector is created.
        # It is representative to the c vectors in the standard boolean PSO formula.
        def vector_rnd_all_ones():
            return 0b111111

        best_position = [0b000001, 0b000000, 0b000111]
        current_position = [0b010000, 0b000001]
        mock_params = Mock()

        velocity_strategy = NeuralBooleanPSOStandardVelocityUpdateStrategy(mock_params)
        result = velocity_strategy._create_component(best_position=best_position, current_position=current_position,
                                                     rnd_vector_partial=vector_rnd_all_ones)

        expected_result = [VelocityComponentEvolve(data=0b010001),
                           VelocityComponentEvolve(data=0b000001),
                           VelocityComponentAdd(data=0b000111)]

        assert result == expected_result


    def test__create_component_when_best_position_shorter_than_current(self):
        def vector_rnd_all_ones():
            return 0b111111

        best_position = [0b111101]
        current_position = [0b010000, 0b000001]

        mock_params = Mock()

        velocity_strategy = NeuralBooleanPSOStandardVelocityUpdateStrategy(mock_params)
        result = velocity_strategy._create_component(best_position=best_position, current_position=current_position,
                                                     rnd_vector_partial=vector_rnd_all_ones)

        expected_result = [VelocityComponentEvolve(data=0b101101),
                           VelocityComponentRemove()]

        assert result == expected_result

    # noinspection PyTypeChecker
    def test__equalize_sizes(self):
        personal_component = [VelocityComponentEvolve(data=0b000000),
                              VelocityComponentAdd(data=0b000000)]
        global_component = [VelocityComponentEvolve(data=0b000000),
                            VelocityComponentEvolve(data=0b000000),
                            VelocityComponentAdd(data=0b000000),
                            VelocityComponentAdd(data=0b000000)
                            ]

        expected_personal_component = copy.deepcopy(personal_component)
        expected_personal_component.append(VelocityComponentRemove())
        expected_personal_component.append(VelocityComponentRemove())
        expected_global_component = copy.deepcopy(global_component)

        mock_params = Mock()
        velocity_strategy = NeuralBooleanPSOStandardVelocityUpdateStrategy(mock_params)
        updated_personal_component, updated_global_component = velocity_strategy._equalize_sizes(personal_component, global_component)

        assert expected_global_component == updated_global_component
        assert expected_personal_component == updated_personal_component

    def test_get_new_velocity_only_personal_factor(self):
        current_velocity = [0b111111, 0b101010]
        current_position = [0b010101, 0b000000]
        pbest_position = [0b000000, 0b000000, 0b000111]
        gbest_position = [0b111111, 0b111111]
        params = BooleanPSOParams(1, 0, 0, 6, k=1)
        strategy = NeuralBooleanPSOStandardVelocityUpdateStrategy(params)
        new_velocity = strategy.get_new_velocity(current_velocity, current_position, pbest_position, gbest_position)
        expected_velocity = []
        expected_velocity.append(VelocityComponentEvolve(data=0b010101))
        expected_velocity.append(VelocityComponentEvolve(data=0b000000))
        expected_velocity.append(VelocityComponentAdd(data=0b000111))
        assert new_velocity == expected_velocity

    def test_get_new_velocity_only_global_factor(self):
        current_velocity = [0b111111, 0b101010]
        current_position = [0b010101, 0b000000]
        pbest_position = [0b000000, 0b000000, 0b000111]
        gbest_position = [0b111101]
        params = BooleanPSOParams(0, 1, 0, 6, k=0)


        strategy = NeuralBooleanPSOStandardVelocityUpdateStrategy(params)

        new_velocity = strategy.get_new_velocity(current_velocity, current_position, pbest_position, gbest_position)
        expected_velocity = []
        expected_velocity.append(VelocityComponentEvolve(data=0b101000))
        expected_velocity.append(VelocityComponentRemove())
        expected_velocity.append(VelocityComponentRemove())
        assert new_velocity == expected_velocity
