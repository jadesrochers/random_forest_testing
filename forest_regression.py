from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, roc_auc_score, roc_curve
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
from bokeh.plotting import figure, output_file, show
from bokeh.models import Label, LabelSet
import pandas as pd
from pandas import DataFrame
import numpy as np
from pathlib import Path
import data_cleaning as dc


cur_path = Path(".")
all_paths = list(map(lambda p : p.as_posix(), list(cur_path.glob('20220909*csv'))))
df = pd.concat(map(pd.read_csv, all_paths))

dc.show_all_col_counts(df)
df = df.rename(columns = {'di05': 'regressor'})
# Could be a good place for some descriptive stats
# df['label'].value_counts()

df = df.select_dtypes('number')
df = dc.drop_rows_missing_data(df, 0.8)
df = dc.drop_columns_missing_data(df, 0.75)
df = df[df['regressor'].notnull()] 

regressor = np.array(df.pop('regressor'))
train_df, test_df, train_value, test_value = train_test_split(df, regressor, test_size=0.5)

# Filling missing with a mean - don't like, but cant fit with missing
train_df = train_df.fillna(train_df.mean())
test_df = test_df.fillna(test_df.mean())

model = RandomForestRegressor(n_estimators=60, criterion='squared_error', max_depth=20, max_features='sqrt', bootstrap=True, max_samples=0.8)
model.fit(train_df, train_value)

import pdb; pdb.set_trace()
predicted = model.predict(test_df)
## Calculate:
# Mean Squared error (mean of sum of squared errors)
mse = mean_squared_error(test_value, predicted, squared=True)
# Root mean squared error (sqrt of above)
rmse = mean_squared_error(test_value, predicted, squared=False)
# Absolute error (Mean of sum of absolute, not squared, errors)
abse = mean_absolute_error(test_value, predicted)

