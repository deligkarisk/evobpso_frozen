import copy
from unittest import TestCase
from unittest.mock import Mock, patch

import VelocityComponent
from VelocityComponent import VelocityComponentRemove, VelocityComponentAdd, VelocityComponentXOR
from velocity_strategy.NeuralBPSOVelocityStrategy import BooleanPSONeuralVelocityStrategy


class TestBooleanPSONeuralVelocityStrategy(TestCase):

    def test__create_component_when_best_position_longer_than_current(self):
        def vector_rnd_all_ones():
            return 0b111111

        best_position = [0b000001, 0b000000, 0b000111]
        current_position = [0b010000, 0b000001]

        velocity_strategy = BooleanPSONeuralVelocityStrategy()
        result = velocity_strategy._create_component(best_position=best_position, current_position=current_position,
                                                     rnd_vector_partial=vector_rnd_all_ones)
        expected_result = [{'operation': 'XOR', 'data': 0b010001}, {'operation': 'XOR', 'data': 0b000001},
                           {'operation': 'Add', 'data': 0b000111}]

        assert result == expected_result

    def test__create_component_when_best_position_shorter_than_current(self):
        def vector_rnd_all_ones():
            return 0b111111

        best_position = [0b111101]
        current_position = [0b010000, 0b000001]

        velocity_strategy = BooleanPSONeuralVelocityStrategy()
        result = velocity_strategy._create_component(best_position=best_position, current_position=current_position,
                                                     rnd_vector_partial=vector_rnd_all_ones)
        expected_result = [{'operation': 'XOR', 'data': 0b101101}, {'operation': 'Remove', 'data': None}]

        assert result == expected_result

    # noinspection PyTypeChecker
    @patch('VelocityComponent.VelocityComponentProcessor')
    def test__equalize_sizes(self, mock_processor):
        personal_component = [VelocityComponentXOR(data=0b000000, velocity_component_processor=mock_processor),
                              VelocityComponentAdd(data=0b000000, velocity_component_processor=mock_processor)]
        global_component = [VelocityComponentXOR(data=0b000000, velocity_component_processor=mock_processor),
                            VelocityComponentXOR(data=0b000000, velocity_component_processor=mock_processor),
                            VelocityComponentAdd(data=0b000000, velocity_component_processor=mock_processor),
                            VelocityComponentAdd(data=0b000000, velocity_component_processor=mock_processor)
                            ]

        expected_personal_component = copy.deepcopy(personal_component)
        expected_personal_component.append(VelocityComponentRemove(velocity_component_processor=mock_processor))
        expected_personal_component.append(VelocityComponentRemove(velocity_component_processor=mock_processor))
        expected_global_component = copy.deepcopy(global_component)

        velocity_strategy = BooleanPSONeuralVelocityStrategy()
        velocity_strategy.processor = mock_processor
        updated_personal_component, updated_global_component = velocity_strategy._equalize_sizes(personal_component, global_component)

        assert expected_global_component == updated_global_component
        assert expected_personal_component == updated_personal_component
