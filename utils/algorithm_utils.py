import os
import copy

def train_and_evaluate_global_best(run_results, runner, training_params, results_folder):
    best_position = copy.deepcopy(run_results['optimization_population'].global_best_position)
    evaluator = runner.evaluator
    evaluator.training_params.train_eval_epochs = training_params.best_solution_training_epochs  # set the training epochs to the best solution training epochs
    best_position_test_results = evaluator.evaluate_for_test(best_position, os.path.join(results_folder, 'best_model'))
    return best_position_test_results