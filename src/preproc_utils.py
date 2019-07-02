from math import sqrt
from sklearn.preprocessing import MinMaxScaler, StandardScaler


def norm_cols(df, cols, op='norm'):
    """
    normalize or standardize selected columns
    :param df: pandas DataFrame
        DataFrame containing columns to be normalized
    :param cols: list
        list of columns to be normalized
    :param op: string
        type of normalization to be performed
        'norm': normalize (scale) values (default)
        'std': standardize (z-score normalize) values
    :return: df : pandas DataFrame
        DataFrame with new columns containing normalized values
    """
    # set labels for captions
    if op == 'norm':
        labels = ['Normalizing', 'normalized']
    elif op == 'std':
        labels = ['Standardizing', 'standardized']
    else:
        raise AttributeError("Parameter 'op' must be specified as either 'norm' or 'std'.")

    print('-----', labels[0], "features:\n", cols)

    # loop over all columns to be normalized
    for col in cols:
        # prepare data for normalization
        values = df[col].values
        values = values.reshape((len(values), 1))
        # initialize scaler
        if op == 'norm':
            scaler = MinMaxScaler(feature_range=(0, 1))
        elif op == 'std':
            scaler = StandardScaler()
        else:
            raise AttributeError("Parameter 'op' must be specified as either 'norm' or 'std'.")
        # train the normalization
        scaler = scaler.fit(values)
        print("\nFeature:", col)
        if op == 'norm':
            print('Min: %f, Max: %f' % (scaler.data_min_, scaler.data_max_))
        elif op == 'std':
            print('Mean: %f, StandardDeviation: %f' % (scaler.mean_,
                                                       sqrt(scaler.var_)))  # normalize feature and save as a new column
        # normalize feature and save as a new column
        normalized = scaler.transform(values)
        df[col + '_' + op] = normalized
    print("\n ----- All columns " + labels[1] + "!")
    return df
