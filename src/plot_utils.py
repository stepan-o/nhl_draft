import matplotlib.pyplot as plt
import numpy as np
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


def plot_decision_regions(df, xcol1, xcol2, ycol, classifier, test_idx=None,
                          resolution=0.02, alpha=0.5):
    # setup marker generator and color map
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    # plot the decision surface
    x1_min, x1_max = df[xcol1].min() - 1, df[xcol1].max() + 1
    x2_min, x2_max = df[xcol2].min() - 1, df[xcol2].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.3, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    for idx, cl in enumerate(np.unique(df[ycol])):
        mask = df[ycol] == cl
        plt.scatter(x=df.loc[mask, xcol1], y=df.loc[mask, xcol2],
                    alpha=alpha, c=colors[idx],
                    marker=markers[idx], label=cl,
                    edgecolor='black')

    # highlight test samples
    if test_idx:
        # plot all samples
        X_test, y_test = X[test_idx, :], y[test_idx]

        plt.scatter(X_test[:, 0], X_test[:, 1],
                    c='', edgecolor='black', alpha=1.0,
                    linewidth=1, marker='o',
                    s=100, label='test set')
