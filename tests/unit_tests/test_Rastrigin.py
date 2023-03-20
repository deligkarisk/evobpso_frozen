from unittest import TestCase

from evobpso.problem.Rastrigin import Rastrigin


class TestRastrigin(TestCase):
    def test_evaluate(self):

        rastrigin2 = Rastrigin(dimensions=2)
        position = [1, 1]
        result = rastrigin2.evaluate(position)
        assert result == 2

        position = [0, 0]
        result = rastrigin2.evaluate(position)
        assert result == 0

