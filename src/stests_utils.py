import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import norm


def fit_norm_dist(series, h_bins='auto',
                  create_figure=True, ax=None,
                  show_plot=True, suptitle=None,
                  title=None, xlabel=None, ylabel='Distribution',
                  figsize=(6, 6), lab2='from_mean',
                  mean_lift=0.99, std_lift=1.007,
                  sig_lift=0.03, per_lift=0.1, val_lift=0.23,
                  x_between=None, x_min=None, x_max=None):
    """

    :param series: pandas Series
        Series to be plotted
    :param h_bins: int
        number of bins to for histogram (default='auto')
    :param create_figure: Boolean
        whether to initialize matplotlib figure and axis
        (default=True)
    :param ax: matplotlib axis
        matplotlib axis to plot on (for multi-plot figures)
    :param show_plot: Boolean
        whether to show plot at the end (default=True)
    :param suptitle: string
        string to use for plot suptitle
    :param title: string
        string to use for plot title
    :param xlabel: string
        string to use for x axis label
    :param ylabel: string
        string to use for y axis label
    :param figsize: tuple(float, float)
        size of the figure
    :param lab2: string (must be 'cdf' or 'from_mean'
        which percentage values to display (CDF or from mean)
        (default='from_mean')
    :param mean_lift: float
        lift for mean caption
    :param std_lift: float
        lift for std caption
    :param sig_lift: float
        lift for sigma captions
    :param per_lift: float
        lift for percentage captions
    :param val_lift: float
        lift for values captions
    :param x_min: float
        estimate probability that x > x_min
    :param x_max: float
        estimate probability that x < x_max
    :param x_between: tuple(float, float)
        estimate probability that x_min < x < x_max
    :return: series.describe()
        statistical summary of the plotted Series
    """
    if create_figure:
        f, ax = plt.subplots(1, figsize=figsize)
        if suptitle:
            f.suptitle(suptitle)
    # plot data
    n, bins, patches = plt.hist(x=series, bins=h_bins)
    mu = series.mean()
    sigma = series.std()
    # initialize a normal distribution
    nd = norm(mu, sigma)
    # plot mean std
    ax.axvline(mu, color='black', linestyle='--')
    ax.text(mu * mean_lift, n.max() * 0.4,
            "Mean: {0:.2f}".format(mu),
            rotation='vertical')
    ax.text(mu * std_lift, n.max() * 0.4,
            "StDev: {0:.2f}".format(sigma),
            rotation='vertical')
    # generate sigma lines and labels
    i = np.arange(-3, 4)
    vlines = mu + i * sigma
    labels1 = pd.Series(i).astype('str') + '$\sigma$'
    if lab2 == 'cdf':
        labels2 = pd.Series(nd.cdf(vlines) * 100) \
                      .round(2).astype('str') + '%'
    elif lab2 == 'from_mean':
        labels2 = pd.Series(abs(50 - nd.cdf(vlines) * 100) * 2) \
                      .round(2).astype('str') + '%'
    else:
        raise AttributeError("Parameter 'lab2' must be either set to 'cdf' or 'from_mean'")
    labels2 = labels2.astype('str')

    # plot sigma lines and labels
    for vline, label1, label2 in zip(vlines, labels1, labels2):
        # plot sigma lines
        if vline != mu:
            ax.axvline(vline, linestyle=':', color='salmon')
            ax.text(vline, n.max() * sig_lift, label1)
        ax.text(vline, n.max() * per_lift, label2, rotation=45)
        ax.text(vline, n.max() * val_lift,
                round(vline, 2), rotation=45)

    # fit a normal curve
    # generate x in range of mu +/- 5 sigma
    x = np.linspace(mu - 5 * sigma, mu + 5 * sigma,
                    1000)
    # calculate PDF
    y = nd.pdf(x)
    # plot fitted distribution
    ax2 = ax.twinx()
    ax2.plot(x, y, color='red', label='Fitted normal curve')
    ax2.legend(loc='best')
    ax2.set_ylim(0)

    if not xlabel:
        xlabel = series.name
    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)
    if not title:
        title = "Histogram of {0}".format(series.name)
    ax.set_title(title)

    if x_min:
        # estimate probability that x > x_min
        p = (1 - nd.cdf(x_min)) * 100
        ax.text(mu - 5 * sigma, n.max() * 0.9, "$P ~ ( X > {0} ) ~ = $".format(x_min) +
                "\n$ = {0:.2f}\%$".format(p))
        ax2.fill_between(x, 0, y, color='red', alpha=0.2,
                         where=x > x_min)
        mask = series > x_min
        print("Using normal distribution, from the total {0:,} records in the Series, "
              "{1:,.0f} are expected to have {2} > {3}"
              .format(len(series), len(series) * p / 100, series.name, x_min),
              "\n\nActual number of records with {0} > {1}: {2:,}"
              .format(series.name, x_min, len(series[mask])))
    elif x_max:
        # estimate probability that x > x_min
        p = nd.cdf(x_max) * 100
        ax.text(mu - 5 * sigma, n.max() * 0.9, "$P ~ ( X < {0} ) ~ = $".format(x_max) +
                "\n$ = {0:.2f}\%$".format(p))
        ax2.fill_between(x, 0, y, color='red', alpha=0.2,
                         where=x < x_max)
        mask = series < x_max
        print("Using normal distribution, from the total {0:,} records in the Series, "
              "{1:,.0f} are expected to have {2} < {3}"
              .format(len(series), len(series) * p / 100, series.name, x_max),
              "\n\nActual number of records with {0} < {1}: {2:,}"
              .format(series.name, x_max, len(series[mask])))
    elif x_between:
        # estimate probability that x_min < x < x_max
        x_min, x_max = x_between
        p = (nd.cdf(x_max) - nd.cdf(x_min)) * 100
        ax.text(mu - 5 * sigma, n.max() * 0.9, "$P ~ ( {0} < X < {1} ) ~ = $".format(x_min, x_max) +
                "\n$ = {0:.2f}\%$".format(p))
        ax2.fill_between(x, 0, y, color='red', alpha=0.2,
                         where=np.logical_and(x > x_min, x < x_max))
        mask = np.logical_and(series < x_max, series > x_min)
        print("Using normal distribution, from the total {0:,} records in the Series, "
              "{1:,.0f} are expected to have {2} < {3} < {4}"
              .format(len(series), len(series) * p / 100, x_min, series.name, x_max),
              "\n\nActual number of records with {0} < {1} < {2}: {3:,}"
              .format(x_min, series.name, x_max, len(series[mask])))

    if show_plot:
        plt.show()

    return series.describe()
