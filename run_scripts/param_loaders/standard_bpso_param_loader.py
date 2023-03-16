from params.FixedArchitectureProperties import FixedArchitectureProperties
from params.NeuralArchitectureParams import NeuralArchitectureParams
from params.OptimizationParams import OptimizationParams
from params.PsoParams import BooleanPSOParams
from params.TrainingParams import TrainingParams


def load_standard_bpso_params(num_classes_in_dataset):
# Basic PSO parameters
    population_size = 2
    iterations = 10
    c1 = 0.5
    c2 = 0.5
    n_bits = 14
    k = 0.5
    mutation_prob = 0

    # Particle initialization parameters
    min_layers = 1
    max_layers = 2

    # Training parameters
    train_eval_epochs = 1  # Number of epochs to run during the optimization procedure
    best_solution_training_epochs = 1  # Number of epochs to run for the global best position
    batch_size = 64
    optimizer = 'adam'
    loss = 'categorical_crossentropy'
    metrics = ['accuracy']


    # Fixed architecture properties
    conv_stride = 1
    activation_function = 'relu'
    pool_layer_kernel_size = 2
    pool_layers_stride = 2
    padding = 'same'

    # Dataset parameters
    image_input_shape = (28, 28, 1)

    pso_params = BooleanPSOParams(pop_size=population_size, iters=iterations, c1=c1, c2=c2, n_bits=n_bits, k=k, mutation_prob=mutation_prob)
    optimizable_architecture_params = NeuralArchitectureParams(min_layers=min_layers, max_layers=max_layers)
    training_params = TrainingParams(batch_size=batch_size, train_eval_epochs=train_eval_epochs, best_solution_training_epochs=best_solution_training_epochs,
                                     loss=loss, optimizer=optimizer, metrics=metrics)

    optimization_params = OptimizationParams(pso_params=pso_params, architecture_params=optimizable_architecture_params,
                                             training_params=training_params)
    fixed_architecture_properties = FixedArchitectureProperties(input_shape=image_input_shape, conv_stride=conv_stride,
                                                                activation_function=activation_function,
                                                                pool_layer_kernel_size=pool_layer_kernel_size,
                                                                pool_layer_stride=pool_layers_stride,
                                                                padding=padding,
                                                                dense_layer_units=num_classes_in_dataset)
    return pso_params, optimizable_architecture_params, training_params, optimization_params, fixed_architecture_properties