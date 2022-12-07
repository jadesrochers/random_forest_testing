from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
from bokeh.plotting import figure, output_file, show
from bokeh.models import Label, LabelSet
import pandas as pd
from pandas import DataFrame
import numpy as np
from pathlib import Path


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


cur_path = Path(".")
all_paths = list(map(lambda p : p.as_posix(), list(cur_path.glob('20220909*'))))
df = pd.concat(map(pd.read_csv, all_paths))

show_all_col_counts(df)
df = df.rename(columns = {'di05': 'regressor'})
# Could be a good place for some descriptive stats
# df['label'].value_counts()

df = df.select_dtypes('number')
df = drop_rows_missing_data(df, 0.8)
df = drop_columns_missing_data(df, 0.75)
df = df[df['regressor'].notnull()] 

regressor = np.array(df.pop('regressor'))
train_df, test_df, train_value, test_value = train_test_split(df, regressor, test_size=0.5)

# Filling missing with a mean - don't like, but cant fit with missing
train_df = train_df.fillna(train_df.mean())
test_df = test_df.fillna(test_df.mean())

import pdb; pdb.set_trace()
model = RandomForestRegressor(n_estimators=60, criterion='squared_error', max_depth=20, max_features='sqrt', bootstrap=True, max_samples=0.8)
model.fit(train_df, train_value)

