import os
import pickle
from pathlib import Path

import pandas as pd
from statistics import mean
import matplotlib.pyplot as plt
import seaborn as sns



run_times = 10
# Experiment name
experiment_name = 'first_trial'
dataset_names = ['mnist', 'mnist_bg', 'mnist_bg_rnd', 'mnist_rot', 'mnist_bg_rot', 'rectangle', 'rect_images',
                 'convex']  # Give a meaningful name, it will be used as a sub-folder for saving data
num_classes = [10, 10, 10, 10, 10, 2, 2, 2]

# Output folder
results_base_folder = '/home/kosmas-deligkaris/DeepBPSOResults'
results_folder = os.path.join(results_base_folder, experiment_name, 'analysis_results')

test_error_mean = {}
test_error_best = {}
means_accuracy = {}
bests_accuracy = {}
test_error_all = {}
test_accuracy_all = {}
results_all_datasets = {}

Path(results_folder).mkdir(parents=True, exist_ok=True)



for k in range(0, len(dataset_names)):
    dataset = dataset_names[k]
    dataset_test_error = []
    dataset_test_accuracy = []
    for i in range(0, run_times):
        results_base_folder_run = os.path.join(results_base_folder, experiment_name, 'run_' + str(i), dataset_names[k])

        results_file = os.path.join(results_base_folder_run, 'run_results.pickle')
        with open(results_file, "rb") as input_file:
            run_results = pickle.load(input_file)
        population = run_results['optimization_population']
        test_accuracy = round((run_results['best_position_test_results'][1]) * 100, 2)
        test_error = round(100 - test_accuracy, 2)
        dataset_test_error.append(test_error)
        dataset_test_accuracy.append(test_accuracy)
    test_error_all[dataset] = dataset_test_error
    test_accuracy_all[dataset] = dataset_test_accuracy
    test_error_mean[dataset] = mean(dataset_test_error)
    test_error_best[dataset] = min(dataset_test_error)


test_error_results = pd.DataFrame.from_records([test_error_mean, test_error_best])
test_error_results.index = ['test_error_mean', 'test_error_best']
filename = os.path.join(results_folder, 'test_error_standard_bpso.csv')
test_error_results.to_csv(filename, index=True)


test_error_individual_runs = pd.DataFrame.from_dict(test_error_all)
filename = os.path.join(results_folder, 'test_error_individual_runs_standard_bpso.csv')
test_error_individual_runs.to_csv(filename, index=True)

test_accuracy_individual_runs = pd.DataFrame.from_dict(test_accuracy_all)
filename = os.path.join(results_folder, 'test_accuracy_individual_runs_standard_bpso.csv')
test_accuracy_individual_runs.to_csv(filename, index=True)



test_ = test_accuracy_individual_runs.melt(var_name='dataset',value_name='test_error')
plot(test_accuracy_individual_runs)
print(data_df.head())
sns.boxplot(x="dataset",
            y="test_error",
            data=data_df,
            showfliers=False)
sns.stripplot(x="dataset",
              y="test_error",
              data=data_df)

#ax = test_accuracy_individual_runs[['mnist', 'mnist_bg']].plot(kind='box', title='boxplot')

# Display the plot
plt.show()





