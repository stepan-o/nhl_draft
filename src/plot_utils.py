import matplotlib.pyplot as plt
import seaborn as sns


def plot_regplot(df, x, y, name=None, alpha=1,
                 create_figure=True, ax=None, show_plot=True, suptitle=True):
    """
    plot a regression plot for variables x and y
    :param df: pandas DataFrame
        DataFrame containing the variables
    :param x: string
        name of the column containing the 1st variable
    :param y: string
        name of the column containing the 2st variable
    :param name: string
        name to display at the top of the plot
    :param alpha: float
        transparency coefficient for the scatter plot
    :param show_plot: boolean
        whether to show the plot at the end (default=True)
    :param ax: matplotlib axis
        (optional): provide axis to plot via function parameters
    :param create_figure: boolean
        whether to create figure and axis (default=True)
    :param suptitle: boolean
        whether to add figure suptitle (default=True)
    :return: None
    """
    if create_figure:
        f, ax = plt.subplots(1)
    sns.regplot(data=df, x=x, y=y, ax=ax, scatter_kws={'alpha': alpha})
    if suptitle:
        ax.set_title(name + ' dataset'
                            '\nrelationship between ' +
                     x + ' and ' + y)
    else:
        ax.set_title('relationship between ' +
                     x + ' and ' + y)
    if show_plot:
        plt.show()
