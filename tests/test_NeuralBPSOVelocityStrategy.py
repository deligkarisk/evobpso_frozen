from unittest import TestCase

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