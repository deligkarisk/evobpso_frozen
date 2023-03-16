import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

def plot_scattered_boxplots(dataframe, var_name, value_name, fig_size):
    fig = plt.figure(figsize=fig_size)
    dataframe_melt = dataframe.melt(var_name=var_name, value_name=value_name)
    sns.boxplot(x=var_name,
                y=value_name,
                data=dataframe_melt,
                showfliers=False)
    sns.stripplot(x=var_name,
                  y=value_name,
                  data=dataframe_melt, color='black')
    return fig



def plot_gbest_convergence(all_data, dataset, fig_size):
    fig = plt.figure(figsize=fig_size)
    df = pd.DataFrame(all_data).transpose()
    df.columns = ['Run 1', 'Run 2', 'Run 3', 'Run 4',
                  'Run 5', 'Run 6', 'Run 7', 'Run 8',
                  'Run 9', 'Run 10']
    ax = sns.lineplot(df)
    plt.title(dataset)
    plt.xlabel("Iteration")
    plt.ylabel("Validation loss")
    sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))
    plt.tight_layout()
    return fig



