from time import time

import pandas as pd


def csv_to_df(file):
    """
    load a DataFrame from a .csv file
    :param file: string
        path to .csv file
    :return: df: pandas DataFrame
        DataFrame with data from .csv file
    """
    t = time()
    df = pd.read_csv(file)
    elapsed = time() - t
    print("----- DataFrame with NHL Draft Data loaded"
          "\nin {0:.2f} seconds".format(elapsed) +
          "\nwith {0:,} rows\nand {1:,} columns"
          .format(df.shape[0], df.shape[1]) +
          "\n-- Column names:\n", df.columns)
    return df


def csv_to_df_rec(rec_name, suffix=None):
    """
    load a DataFrame from a .csv file obtained
    from NHL Records API Records endpoint
    :param rec_name: string
        name of the record to be loaded
    :param suffix: string
        suffix to the file name (e.g., '_new_cols')
    :return: df: pandas DataFrame
        DataFrame with data from .csv file
             name: string
        name of the Records file
    """
    rec_file = 'data/nhl_api/records/records_main.csv'
    df_rec = pd.read_csv(rec_file)
    mask = df_rec['descriptionKey'] == rec_name
    name = df_rec.loc[mask, 'description'].values[0]
    print("----- NHL Records\n---", name, '\n')

    file = 'data/nhl_api/records/' + \
           rec_name + suffix + '.csv'
    t = time()
    df = pd.read_csv(file)
    elapsed = time() - t
    print("----- DataFrame with NHL Records Data loaded"
          "\nin {0:.2f} seconds".format(elapsed) +
          "\nwith {0:,} rows\nand {1:,} columns"
          .format(df.shape[0], df.shape[1]) +
          "\n-- Column names:\n", df.columns)
    return df, name


def df_to_csv(df: pd.DataFrame, save_path: object):
    """
    save DataFrame to a file
    :param df: pandas DataFrame
        DataFrame to be saved
    :param save_path: string
        where to save the file
    :return:
    """
    t = time()
    df.to_csv(save_path, index=False)
    elapsed = time() - t
    print("DataFrame saved to file:\n", save_path,
          "\ntook {0:.2f} seconds".format(elapsed))
