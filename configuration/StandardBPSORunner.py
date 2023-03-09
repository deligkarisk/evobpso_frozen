import copy
import os
import time
from pathlib import Path

from architecture_decoder.StandardArchitectureDecoder import StandardArchitectureDecoder
from component_creator.StandardBooleanComponentCreator import StandardBooleanComponentCreator
from component_data_calculator.StandardBoolenComponentDataCalculator import StandardBooleanComponentDataCalculator
from evaluator.StandardNNEvaluator import StandardNNEvaluator
from initializer.BinaryInitializer import BinaryInitializer
from model_creator.TensorflowModelCreator import TensorflowModelCreator
from population.Population import Population
from position_update_strategy.StandardPositionUpdateStrategy import StandardPositionUpdateStrategy
from position_validator.DoNothingPositionValidator import DoNothingPositionValidator
from utils.filesystem_utils import save_object
from velocity_update_strategy.StandardVelocityUpdateStrategy import StandardVelocityUpdateStrategy
from velocity_update_strategy.component_merge_strategy.StandardComponentMergeStrategy import StandardComponentMergeStrategy


class StandardBPSORunner:
    def __init__(self, optimization_params, architecture_properties, data_loader, results_folder) -> None:
        data_calculator = StandardBooleanComponentDataCalculator(params=optimization_params)
        component_creator = StandardBooleanComponentCreator(data_calculator=data_calculator)
        component_merger = StandardComponentMergeStrategy()
        velocity_strategy = StandardVelocityUpdateStrategy(component_creator=component_creator, component_merger=component_merger, params=optimization_params)
        position_update_strategy = StandardPositionUpdateStrategy(optimization_params=optimization_params)
        decoder = StandardArchitectureDecoder()
        model_creator = TensorflowModelCreator(fixed_architecture_properties=architecture_properties)
        self.evaluator = StandardNNEvaluator(architecture_decoder=decoder, model_creator=model_creator,
                                        training_params=optimization_params.training_params, data_loader=data_loader)
        validator = DoNothingPositionValidator()
        initializer = BinaryInitializer(params=optimization_params)
        self.population = Population(params=optimization_params, validator=validator, initializer=initializer,
                                     evaluator=self.evaluator, velocity_update_strategy=velocity_strategy,
                                     position_update_strategy=position_update_strategy, results_folder=results_folder)
        self.iterations = optimization_params.pso_params.iters
        self.results_folder = results_folder

        Path(self.results_folder).mkdir(parents=True, exist_ok=True)

    def run(self):
        start_time = time.time()

        aggregated_history = []

        results = self.population.iterate(first_iter=True)
        aggregated_history.append(results)
        for i in range(0, self.iterations):
            print('starting iteration: ' + str(i))
            results = self.population.iterate(first_iter=False)
            aggregated_history.append(results)

        end_time = time.time()
        elapsed_time = ((end_time - start_time) / 60) / 60
        print("Runtime: " + str(elapsed_time) + " hours.")

        results = {'optimization_elapsed_time': elapsed_time,
                           'optimization_results_history': aggregated_history,
                           'optimization_population': self.population}

        return results



