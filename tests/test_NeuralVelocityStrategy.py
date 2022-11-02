import copy
from unittest import TestCase
from unittest.mock import Mock, patch

import VelocityComponent
from VelocityComponent import VelocityComponentRemove, VelocityComponentAdd, VelocityComponentEvolve
from velocity_strategy.NeuralVelocityStrategy import BooleanPSONeuralVelocityStrategy


class TestBooleanPSONeuralVelocityStrategy(TestCase):

    @patch('VelocityComponent.VelocityComponentProcessor')
    def test__create_component_when_best_position_longer_than_current(self, mock_processor):
        def vector_rnd_all_ones():
            return 0b111111

        best_position = [0b000001, 0b000000, 0b000111]
        current_position = [0b010000, 0b000001]

        velocity_strategy = BooleanPSONeuralVelocityStrategy(processor=mock_processor)
        result = velocity_strategy._create_component(best_position=best_position, current_position=current_position,
                                                     rnd_vector_partial=vector_rnd_all_ones)

        expected_result = [VelocityComponentEvolve(data=0b010001, processor=mock_processor), VelocityComponentEvolve(data=0b000001, processor=mock_processor),
                           VelocityComponentAdd(data=0b000111, processor=mock_processor)]

        assert result == expected_result

    @patch('VelocityComponent.VelocityComponentProcessor')
    def test__create_component_when_best_position_shorter_than_current(self, mock_processor):
        def vector_rnd_all_ones():
            return 0b111111

        best_position = [0b111101]
        current_position = [0b010000, 0b000001]

        velocity_strategy = BooleanPSONeuralVelocityStrategy()
        result = velocity_strategy._create_component(best_position=best_position, current_position=current_position,
                                                     rnd_vector_partial=vector_rnd_all_ones)
        expected_result = [{'operation': 'XOR', 'data': 0b101101}, {'operation': 'Remove', 'data': None}]

        expected_result = [VelocityComponentEvolve(data=0b101101, processor=mock_processor), VelocityComponentRemove(processor=mock_processor)]

        assert result == expected_result

    # noinspection PyTypeChecker
    @patch('VelocityComponent.VelocityComponentProcessor')
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

        velocity_strategy = BooleanPSONeuralVelocityStrategy()
        velocity_strategy.processor = mock_processor
        updated_personal_component, updated_global_component = velocity_strategy._equalize_sizes(personal_component, global_component)

        assert expected_global_component == updated_global_component
        assert expected_personal_component == updated_personal_component


