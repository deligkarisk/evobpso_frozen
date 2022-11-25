import copy
from unittest import TestCase
from unittest.mock import patch

from pso_params.PsoParams import PsoParams, StandardPsoParams, NeuralBPSOParams
from velocity_component.VelocityComponent import VelocityComponentRemove, VelocityComponentAdd, VelocityComponentEvolve, VelocityComponentProcessor
from velocity_update_strategy.NeuralBPSOVelocityUpdateStrategy import NeuralBPSOStandardVelocityStrategy


class TestNeuralBPSOStandardVelocityStrategy(TestCase):

    @patch('velocity_component.VelocityComponent.VelocityComponentProcessor')
    def test__create_component_when_best_position_longer_than_current(self, mock_processor):
        def vector_rnd_all_ones():
            return 0b111111

        best_position = [0b000001, 0b000000, 0b000111]
        current_position = [0b010000, 0b000001]

        velocity_strategy = NeuralBPSOStandardVelocityStrategy(processor=mock_processor)
        result = velocity_strategy._create_factor(best_position=best_position, current_position=current_position,
                                                  rnd_vector_partial=vector_rnd_all_ones)

        expected_result = [VelocityComponentEvolve(data=0b010001, processor=mock_processor),
                           VelocityComponentEvolve(data=0b000001, processor=mock_processor),
                           VelocityComponentAdd(data=0b000111, processor=mock_processor)]

        assert result == expected_result

    @patch('velocity_component.VelocityComponent.VelocityComponentProcessor')
    def test__create_component_when_best_position_shorter_than_current(self, mock_processor):
        def vector_rnd_all_ones():
            return 0b111111

        best_position = [0b111101]
        current_position = [0b010000, 0b000001]

        velocity_strategy = NeuralBPSOStandardVelocityStrategy()
        result = velocity_strategy._create_factor(best_position=best_position, current_position=current_position,
                                                  rnd_vector_partial=vector_rnd_all_ones)
        expected_result = [{'operation': 'XOR', 'data': 0b101101}, {'operation': 'Remove', 'data': None}]

        expected_result = [VelocityComponentEvolve(data=0b101101, processor=mock_processor),
                           VelocityComponentRemove(processor=mock_processor)]

        assert result == expected_result

    # noinspection PyTypeChecker
    @patch('velocity_component.VelocityComponent.VelocityComponentProcessor')
    def test__equalize_sizes(self, mock_processor):
        personal_component = [VelocityComponentEvolve(data=0b000000, processor=mock_processor),
                              VelocityComponentAdd(data=0b000000, processor=mock_processor)]
        global_component = [VelocityComponentEvolve(data=0b000000, processor=mock_processor),
                            VelocityComponentEvolve(data=0b000000, processor=mock_processor),
                            VelocityComponentAdd(data=0b000000, processor=mock_processor),
                            VelocityComponentAdd(data=0b000000, processor=mock_processor)
                            ]

        expected_personal_component = copy.deepcopy(personal_component)
        expected_personal_component.append(VelocityComponentRemove(processor=mock_processor))
        expected_personal_component.append(VelocityComponentRemove(processor=mock_processor))
        expected_global_component = copy.deepcopy(global_component)

        velocity_strategy = NeuralBPSOStandardVelocityStrategy()
        velocity_strategy.processor = mock_processor
        updated_personal_component, updated_global_component = velocity_strategy._equalize_sizes(personal_component, global_component)

        assert expected_global_component == updated_global_component
        assert expected_personal_component == updated_personal_component

    def test_get_new_velocity_only_personal_factor(self):
        current_velocity = [0b111111, 0b101010]
        current_position = [0b010101, 0b000000]
        pbest_position = [0b000000, 0b000000, 0b000111]
        gbest_position = [0b111111, 0b111111]
        params = NeuralBPSOParams(1, 0, 0, 6, k=1)
        fixed_processor = VelocityComponentProcessor(params)

        strategy = NeuralBPSOStandardVelocityStrategy()
        strategy.processor = fixed_processor

        new_velocity = strategy.get_new_velocity(current_velocity, current_position, pbest_position, gbest_position, params)
        expected_velocity = []
        expected_velocity.append(VelocityComponentEvolve(data=0b010101, processor=fixed_processor))
        expected_velocity.append(VelocityComponentEvolve(data=0b000000, processor=fixed_processor))
        expected_velocity.append(VelocityComponentAdd(data=0b000111, processor=fixed_processor))
        assert new_velocity == expected_velocity

    def test_get_new_velocity_only_global_factor(self):
        current_velocity = [0b111111, 0b101010]
        current_position = [0b010101, 0b000000]
        pbest_position = [0b000000, 0b000000, 0b000111]
        gbest_position = [0b111101]
        params = NeuralBPSOParams(0, 1, 0, 6, k=0)
        fixed_processor = VelocityComponentProcessor(params)

        strategy = NeuralBPSOStandardVelocityStrategy()
        strategy.processor = fixed_processor

        new_velocity = strategy.get_new_velocity(current_velocity, current_position, pbest_position, gbest_position, params)
        expected_velocity = []
        expected_velocity.append(VelocityComponentEvolve(data=0b101000, processor=fixed_processor))
        expected_velocity.append(VelocityComponentRemove(processor=fixed_processor))
        expected_velocity.append(VelocityComponentRemove(processor=fixed_processor))
        assert new_velocity == expected_velocity
