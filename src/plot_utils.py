import matplotlib.pyplot as plt
import seaborn as sns


def plot_regplot(df, x, y, name=None, alpha=1):
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
    :return: None
    """
    f, ax = plt.subplots(1)
    sns.regplot(data=df, x=x, y=y, ax=ax, scatter_kws={'alpha': alpha})
    ax.set_title(name + ' dataset'
                 '\nrelationship between ' +
                 x + ' and ' + y)
    plt.show()
