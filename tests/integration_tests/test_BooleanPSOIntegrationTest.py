from unittest import TestCase

from component_creator.StandardBooleanComponentCreator import StandardBooleanComponentCreator
from component_data_calculator.StandardBoolenComponentDataCalculator import StandardBooleanComponentDataCalculator
from decoder.PositionToNNDecoder import PositionToNNDecoder
from evaluator.StandardNNEvaluator import StandardNNEvaluator
from initializer.BinaryInitializer import BinaryInitializer
from params.NeuralArchitectureParams import NeuralArchitectureParams
from params.Params import Params
from params.PsoParams import PsoParams, BooleanPSOParams
from population.Population import Population
from position_update_strategy.StandardPositionUpdateStrategy import StandardPositionUpdateStrategy
from position_validator.StandardBooleanPSOPositionValidator import StandardBooleanPSOPositionValidator
from velocity_update_strategy.StandardVelocityUpdateStrategy import StandardVelocityUpdateStrategy


class TestBooleanPSOIntegrationTest(TestCase):

    def test_runs_without_errors(self):

        pso_params = BooleanPSOParams(c1=0.5, c2=0.5, n_bits=8, k=0.5)
        architecture = NeuralArchitectureParams(min_out_conv=8, max_out_conv=64, min_kernel_conv=2, max_kernel_conv=8, min_layers=8, max_layers=32)
        all_params = Params(pso_params=pso_params, architecture_params=architecture)

        data_calculator = StandardBooleanComponentDataCalculator(params=all_params)
        component_creator = StandardBooleanComponentCreator(data_calculator=data_calculator)
        velocity_strategy = StandardVelocityUpdateStrategy(component_creator=component_creator, params=all_params)
        position_update_strategy = StandardPositionUpdateStrategy()


        decoder = PositionToNNDecoder()
        evaluator = StandardNNEvaluator()


        validator = StandardBooleanPSOPositionValidator()
        initializer = BinaryInitializer(params=all_params)
        parent_pop = Population(pop_size=20, params=all_params, decoder=decoder, validator=validator, initializer=initializer,
                                evaluator=evaluator, velocity_update_strategy=velocity_strategy, position_update_strategy=position_update_strategy)


