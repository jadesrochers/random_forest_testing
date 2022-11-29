from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from pathlib import Path

import pdb; pdb.set_trace()
cur_path = Path(".")
all_paths = list(map(lambda p : p.as_posix(), list(cur_path.glob("20*.csv"))))
df = pd.concat(map(pd.read_csv, all_paths))

# Some data cleaning. This is recommended by the notebook and I am following that 
# for the sake of keeping things fast right now. Edit or adjust if you want.
# There appear to be three labels, so you could avoid the lumping step to make things 
# a bit more complicated if you want
df['_RFHLTH'] = df['_RFHLTH'].replace({2: 0})
df = df.loc[df['_RFHLTH'].isin([0, 1])].copy()
df = df.rename(columns = {'_RFHLTH': 'label'})
df['label'].value_counts()
# And drop columns that should not be used in the model - some of these seem
# to be aliases for the label column

labels = np.array(df.pop('label'))
train_df, test_df, train_labels, test_labels = train_test_split(df, labels, test_size=0.4)

# Here is theensemble. It will create the Decision Trees under the hood automatically
model = RandomForestClassifier(n_estimators=50, bootstrap=True, max_features='sqrt')


