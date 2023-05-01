from evobpso.architecture_decoder.BooleanSequentialArchitectureDecoder import BooleanSequentialArchitectureDecoder
from evobpso.encoding.Encoding import Encoding
from evobpso.evaluator.StandardNNEvaluator import StandardNNEvaluator
from evobpso.initializer.BinaryInitializer import BinaryInitializer
from evobpso.model_creator.TensorflowModelCreator import TensorflowModelCreator
from evobpso.position_update_strategy.BooleanRandomPositionUpdateStrategy import BooleanRandomPositionUpdateStrategy
from evobpso.position_validator.ValidatePoolingLayers import ValidatePoolingLayers
from evobpso.velocity_update_strategy.ReturnZeroVelocityUpdateStrategy import ReturnZeroVelocityUpdateStrategy


class SequentialRandomSearchScheme:
    def __init__(self, version, optimization_params, encoding: Encoding):
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


        self.velocity_update_strategy = ReturnZeroVelocityUpdateStrategy()
        self.position_update_strategy = BooleanRandomPositionUpdateStrategy(optimization_params=optimization_params)


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



