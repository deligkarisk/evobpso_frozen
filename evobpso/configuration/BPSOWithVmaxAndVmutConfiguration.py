from evobpso.architecture_decoder.StandardArchitectureDecoder import StandardArchitectureDecoder
from evobpso.component_creator.VariableLengthComponentCreator import VariableLengthComponentCreator
from evobpso.component_creator_data_calculator.BoolenComponentCreatorDataCalculator import BooleanComponentCreatorDataCalculator
from evobpso.configuration.Configuration import Configuration
from evobpso.evaluator.StandardNNEvaluator import StandardNNEvaluator
from evobpso.initializer.BinaryInitializer import BinaryInitializer
from evobpso.initializer.Initializer import Initializer
from evobpso.model_creator.TensorflowModelCreator import TensorflowModelCreator
from evobpso.population.Population import Population
from evobpso.position_update_strategy.StandardPositionUpdateStrategy import StandardPositionUpdateStrategy
from evobpso.position_validator.ValidatePoolingLayers import ValidatePoolingLayers
from evobpso.velocity_update_strategy.VelocityUpdateWithVmaxAndVmutStrategy import VelocityUpdateWithVmaxAndVmutStrategy
from evobpso.velocity_update_strategy.component_merge_strategy.VariableLengthCalculateDataComponentMergeStrategy import VariableLengthCalculateDataComponentMergeStrategy


class BPSOWithVmaxAndVmutConfiguration(Configuration):
    def __init__(self, optimization_params, architecture_properties, data_loader, initializer: Initializer, results_folder) -> None:
        data_calculator = BooleanComponentCreatorDataCalculator(params=optimization_params)
        component_creator = VariableLengthComponentCreator(data_calculator=data_calculator)
        component_merger = VariableLengthCalculateDataComponentMergeStrategy()
        velocity_strategy = VelocityUpdateWithVmaxAndVmutStrategy(component_creator=component_creator, component_merger=component_merger, params=optimization_params)
        position_update_strategy = StandardPositionUpdateStrategy(optimization_params=optimization_params)
        decoder = StandardArchitectureDecoder()
        model_creator = TensorflowModelCreator(fixed_architecture_properties=architecture_properties)
        validator = ValidatePoolingLayers(pooling_layer_bit_num=decoder.pooling_layer_bit_position)
        initializer = initializer

        self.evaluator = StandardNNEvaluator(architecture_decoder=decoder, model_creator=model_creator,
                                        training_params=optimization_params.training_params, data_loader=data_loader)
        self.population = Population(params=optimization_params, validator=validator, initializer=initializer,
                                     evaluator=self.evaluator, velocity_update_strategy=velocity_strategy,
                                     position_update_strategy=position_update_strategy, results_folder=results_folder)
        self.iterations = optimization_params.pso_params.iters
        self.results_folder = results_folder





