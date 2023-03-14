import seaborn as sns
import matplotlib.pyplot as plt

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
