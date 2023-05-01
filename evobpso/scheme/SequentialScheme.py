from evobpso.architecture_decoder.BooleanSequentialArchitectureDecoder import BooleanSequentialArchitectureDecoder
from evobpso.component_creator.VariableLengthComponentCreator import VariableLengthComponentCreator
from evobpso.component_creator.data_calculator.BoolenComponentCreatorDataCalculator import BooleanComponentCreatorDataCalculator
from evobpso.component_merger.VariableLengthCalculateDataComponentMerger import VariableLengthCalculateDataComponentMerger
from evobpso.component_merger.data_calculator.BooleanComponentMergerDataCalculator import BooleanComponentMergerDataCalculator
from evobpso.encoding.Encoding import Encoding
from evobpso.evaluator.StandardNNEvaluator import StandardNNEvaluator
from evobpso.initializer.BinaryInitializer import BinaryInitializer
from evobpso.model_creator.TensorflowModelCreator import TensorflowModelCreator
from evobpso.position_update_strategy.StandardPositionUpdateStrategy import StandardPositionUpdateStrategy
from evobpso.position_validator.ValidatePoolingLayers import ValidatePoolingLayers
from evobpso.velocity_update_extension.BooleanVmaxExtension import BooleanVmaxExtension
from evobpso.velocity_update_extension.BooleanVmutExtension import BooleanVmutExtension
from evobpso.velocity_update_strategy.StandardVelocityUpdateStrategy import StandardVelocityUpdateStrategy


class SequentialScheme:
    def __init__(self, version, variable_length, vmax, vmut, optimization_params, encoding: Encoding):
        self.compiled = False
        self.evaluator = None
        self.position_validator = None
        self.results_folder = None
        self.initializer = None
        self.velocity_update_strategy = None
        self.position_update_strategy = None
        self.optimization_params = optimization_params

        self.initializer = self._get_initializer(version=version, optimization_params=optimization_params)
        self.decoder = self._get_decoder(version=version, encoding=encoding)

        component_creator_data_calculator, component_merger_data_calculator = self._get_data_calculators(version=version,
                                                                                                         optimization_params=optimization_params)
        component_creator, component_merger = self._get_component_creator_and_merger(
            component_merger_data_calculator=component_merger_data_calculator,
            component_creator_data_calculator=component_creator_data_calculator,
            variable_length=variable_length)

        velocity_update_extensions = self._get_velocity_update_extensions(vmax, vmut, optimization_params)

        self.velocity_update_strategy = StandardVelocityUpdateStrategy(component_creator=component_creator,
                                                                       component_merger=component_merger,
                                                                       params=optimization_params,
                                                                       velocity_update_extensions=velocity_update_extensions)

        self.position_update_strategy = StandardPositionUpdateStrategy(optimization_params=optimization_params)


    def compile(self, fixed_architecture_properties, data_loader, results_folder):
        """ The compile method sets-up the classes that do not have different implementations.
        If you create different implementations of the below classes, and you would like to use them in configurations,
        then move them to the init method, following similar pattern to the other classes (e.g. initializer, component_creator). """
        model_creator = TensorflowModelCreator(fixed_architecture_properties=fixed_architecture_properties)

        self.evaluator = StandardNNEvaluator(architecture_decoder=self.decoder, model_creator=model_creator,
                                             training_params=self.optimization_params.training_params, data_loader=data_loader)
        self.position_validator = ValidatePoolingLayers(pooling_layer_bit_num=self.decoder.encoding.pooling_layer_bit_position)
        self.results_folder = results_folder

        self.compiled = True


    def _get_initializer(self, version, optimization_params):
        if version == 'boolean':
            initializer = BinaryInitializer(params=optimization_params)
        else:
            raise NotImplementedError
        return initializer



    def _get_decoder(self, version, encoding):
        if version == 'boolean':
            decoder = BooleanSequentialArchitectureDecoder(encoding=encoding)
        else:
            raise NotImplementedError
        return decoder



    def _get_data_calculators(self, version, optimization_params):
        if version == 'boolean':
            component_creator_data_calculator = BooleanComponentCreatorDataCalculator(params=optimization_params)
            component_merger_data_calculator = BooleanComponentMergerDataCalculator()
        else:
            raise NotImplementedError

        return component_creator_data_calculator, component_merger_data_calculator


    def _get_component_creator_and_merger(self, component_creator_data_calculator, component_merger_data_calculator, variable_length):
        if variable_length:
            component_creator = VariableLengthComponentCreator(data_calculator=component_creator_data_calculator)
            component_merger = VariableLengthCalculateDataComponentMerger(component_merger_data_calculator=component_merger_data_calculator)
        else:
            raise NotImplementedError

        return component_creator, component_merger

    def _get_velocity_update_extensions(self, vmax, vmut, optimization_params):
        velocity_update_extensions = []
        if vmax:
            velocity_update_extensions.append(BooleanVmaxExtension(params=optimization_params))
        if vmut:
            velocity_update_extensions.append(BooleanVmutExtension(params=optimization_params))
        return velocity_update_extensions
