from pandas import DataFrame
import pandas as pd

# Methods for cleaning and inspecting data that might be useful
# with any pandas dataframe

def show_all_col_data(df: DataFrame, head_count: int=15):
    with pd.option_context('display.max_columns', None):
        print(df.head(head_count))


def show_all_col_counts(df: DataFrame):
    # max_rows option because the results is a df with many rows
    with pd.option_context('display.max_rows', None):
        print(df.count(axis=0))


def drop_columns_missing_data(df: DataFrame, th: float) -> DataFrame:
    colnames = df.columns
    miss_colnames = []
    for colname in colnames:
        miss_pct = df[colname].isnull().sum()/len(df[colname])
        if miss_pct > th:
            miss_colnames.append(colname)
    return df.drop(columns = miss_colnames)


def drop_rows_missing_data(df: DataFrame, th: float) -> DataFrame:
    num_cols = len(df.columns)
    return df[(df.isnull().sum(axis=1) / num_cols) > th] 


