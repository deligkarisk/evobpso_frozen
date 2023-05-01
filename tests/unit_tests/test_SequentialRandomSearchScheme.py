from unittest import TestCase
from unittest.mock import Mock

from evobpso.initializer.BinaryInitializer import BinaryInitializer
from evobpso.position_update_strategy.BooleanRandomPositionUpdateStrategy import BooleanRandomPositionUpdateStrategy
from evobpso.scheme.SequentialRandomSearchScheme import SequentialRandomSearchScheme
from evobpso.velocity_update_strategy.ReturnZeroVelocityUpdateStrategy import ReturnZeroVelocityUpdateStrategy


class TestSequentialRandomSearchScheme(TestCase):

    def test_random_search_boolean_version(self):
        scheme = SequentialRandomSearchScheme(version='boolean', optimization_params=Mock(), encoding=Mock())
        assert scheme.initializer.__class__.__name__ == BinaryInitializer.__name__
        assert scheme.velocity_update_strategy.__class__.__name__ == ReturnZeroVelocityUpdateStrategy.__name__
        assert scheme.position_update_strategy.__class__.__name__ == BooleanRandomPositionUpdateStrategy.__name__

