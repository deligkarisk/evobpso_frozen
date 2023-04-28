from evobpso.architecture_decoder.StandardArchitectureDecoder import StandardArchitectureDecoder
from evobpso.component_creator.VariableLengthComponentCreator import VariableLengthComponentCreator
from evobpso.component_creator.data_calculator.BoolenComponentCreatorDataCalculator import BooleanComponentCreatorDataCalculator
from evobpso.component_merger.VariableLengthCalculateDataComponentMerger import VariableLengthCalculateDataComponentMerger
from evobpso.component_merger.data_calculator.BooleanComponentMergerDataCalculator import BooleanComponentMergerDataCalculator
from evobpso.evaluator.StandardNNEvaluator import StandardNNEvaluator
from evobpso.initializer.BinaryInitializer import BinaryInitializer
from evobpso.model_creator.TensorflowModelCreator import TensorflowModelCreator
from evobpso.position_update_strategy.StandardPositionUpdateStrategy import StandardPositionUpdateStrategy
from evobpso.position_validator.ValidatePoolingLayers import ValidatePoolingLayers
from evobpso.velocity_update_extension.BooleanVmaxExtension import BooleanVmaxExtension
from evobpso.velocity_update_extension.BooleanVmutExtension import BooleanVmutExtension
from evobpso.velocity_update_strategy.StandardVelocityUpdateStrategy import StandardVelocityUpdateStrategy


class Scheme:
    def __init__(self, version, variable_length, vmax, vmut, optimization_params):

        self.evaluator = None
        self.position_validator = None
        self.results_folder = None
        self.initializer = None
        self.velocity_update_strategy = None
        self.position_update_strategy = None
        self.optimization_params = optimization_params

        velocity_update_extensions = []

        if version == 'boolean':
            self.initializer = BinaryInitializer(params=optimization_params)
        elif version == 'real':
            raise NotImplementedError

        if version == 'boolean' and variable_length is True:
            component_creator, component_merger = self._initialize_with_boolean_and_variable_length(optimization_params)

            if vmax:
                velocity_update_extensions.append(BooleanVmaxExtension(params=optimization_params))
            if vmut:
                velocity_update_extensions.append(BooleanVmutExtension(params=optimization_params))

            self.velocity_update_strategy = StandardVelocityUpdateStrategy(component_creator=component_creator, component_merger=component_merger,
                                                                           params=optimization_params,
                                                                           velocity_update_extensions=velocity_update_extensions)

            self.position_update_strategy = StandardPositionUpdateStrategy(optimization_params=optimization_params)



    def compile(self, fixed_architecture_properties, data_loader, results_folder):
        decoder = StandardArchitectureDecoder()
        model_creator = TensorflowModelCreator(fixed_architecture_properties=fixed_architecture_properties)

        self.evaluator = StandardNNEvaluator(architecture_decoder=decoder, model_creator=model_creator,
                                        training_params=self.optimization_params.training_params, data_loader=data_loader)
        self.position_validator = ValidatePoolingLayers(pooling_layer_bit_num=decoder.pooling_layer_bit_position)
        self.results_folder=results_folder

    def _initialize_with_boolean_and_variable_length(self, optimization_params):
        data_calculator = BooleanComponentCreatorDataCalculator(params=optimization_params)
        component_creator = VariableLengthComponentCreator(data_calculator=data_calculator)
        component_merger_data_calculator = BooleanComponentMergerDataCalculator()
        component_merger = VariableLengthCalculateDataComponentMerger(component_merger_data_calculator=component_merger_data_calculator)
        return component_creator, component_merger

