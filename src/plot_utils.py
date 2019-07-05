import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import ListedColormap


def plot_regplot(df, x, y, name=None, alpha=1, order=1, color='blue',
                 create_figure=True, ax=None, show_plot=True, suptitle=True):
    """
    plot a linear model plot for variables x and y
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
    :param order: int
        order of the linear model (default=1)
    :param color: string
        color to use for scatter points
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
    sns.regplot(data=df, x=x, y=y, order=order, color=color,
                ax=ax, scatter_kws={'alpha': alpha})
    if suptitle:
        ax.set_title(name + ' dataset'
                            '\nrelationship between ' +
                     x + ' and ' + y)
    else:
        ax.set_title('relationship between ' +
                     x + ' and ' + y)
    if show_plot:
        plt.show()


def plot_decision_regions(X, y, classifier, title="", test_idx=None, xlabel="x1", ylabel="x2",
                          x1_min_rat=0.99, x1_max_rat=1.01, x2_min_rat=0.99, x2_max_rat=1.01,
                          resolution=0.02, alpha=0.5, figsize=(6, 6), legend_loc='best',
                          markers=('s', 'x', 'o', '^', 'v'),
                          colors=('red', 'blue', 'lightgreen', 'gray', 'cyan'),
                          result='show', name="_", save_path=""):
    # create figure and axis
    f, ax = plt.subplots(1, figsize=figsize)
    # setup color map
    cmap = ListedColormap(colors[:len(np.unique(y))])
    if type(X) == pd.core.frame.DataFrame:
        # get features
        x1, x2 = X.iloc[:, 0].values, X.iloc[:, 1].values
    elif type(X) == np.ndarray:
        x1, x2 = X[:, 0], X[:, 1]
    else:
        raise AttributeError("Parameter 'X' must be either a NumPy array or a Pandas DataFrame."
                             "{0} provided.".format(type(X)))

    # plot the decision surface
    x1_min, x1_max = x1.min() * x1_min_rat, x1.max() * x1_max_rat
    x2_min, x2_max = x2.min() * x2_min_rat, x2.max() * x2_max_rat
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.3, cmap=cmap)
    ax.set_xlim(xx1.min(), xx1.max())
    ax.set_ylim(xx2.min(), xx2.max())

    for idx, cl in enumerate(np.unique(y)):
        mask = y == cl
        plt.scatter(x=x1[mask], y=x2[mask],
                    alpha=alpha, c=colors[idx],
                    marker=markers[idx], label=cl,
                    edgecolor='black')

    # highlight test samples
    if test_idx:
        # plot all samples
        x1_test, x2_test = x1[test_idx, :], x2[test_idx]

        plt.scatter(x1_test[:, 0], x2_test[:, 1],
                    c='', edgecolor='black', alpha=1.0,
                    linewidth=1, marker='o',
                    s=100, label='test set')

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    plt.legend(loc=legend_loc)

    if result == 'show':
        plt.show()
    elif result == 'save':
        path = 'img/decision_boundaries/' + save_path + '_'.join([xlabel, ylabel, name]) + '.png'
        f.savefig(path, dpi=300)
        plt.close(f)
        print("Plot saved to file", path)
    else:
        return "Parameter 'result' must be set to either 'show' or 'save'."
