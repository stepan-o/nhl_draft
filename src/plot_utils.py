import seaborn as sns


def plot_regplot(df, x, y, name=None):
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
    :return: None
    """
    ax = sns.regplot(data=df, x=x, y=y)
    ax.set_title(name +
                 '\nrelationship between ' +
                 x + ' and ' + y)
