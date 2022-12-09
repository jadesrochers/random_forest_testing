from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import precision_score, recall_score, roc_auc_score, roc_curve, confusion_matrix
from sklearn.tree import export_graphviz
import matplotlib.pyplot as plt
from bokeh.plotting import figure, output_file, show
from bokeh.models import Label, LabelSet
import pandas as pd
from pandas import DataFrame
import numpy as np
from numpy import array as np_array
from pathlib import Path


def show_all_col_data(df: DataFrame, head_count: int=15):
    # Use set_option to change option, option_context to change just in with: block
    # pd.set_option('display.max_columns', None)
    with pd.option_context('display.max_columns', None):
        print(df.head(15))


def nodes_and_depth(model: RandomForestClassifier): 
    nodes = []
    depth = []
    for tree in model.estimators_:
        nodes.append(tree._tee_.node_count)
        depth.append(tree.tree_.max_depth)
    print(f'Avg num nodes: {int(np.mean(nodes))}')
    print(f'Avg maximum depth: {int(np.mean(nodes))}')


# Create baseline roc figure with a line for default case
def create_roc_performance_figure(title: str, test_labels):
    baseline = {}
    baseline['recall'] = recall_score(test_labels, [1 for _ in range(len(test_labels))])
    baseline['precision'] = precision_score(test_labels, [1 for _ in range(len(test_labels))])
    baseline['roc'] = 0.5
    fg = figure(title=title, width=800, height=600, tools="pan, reset, save")
    fg.xaxis.axis_label = "False Positive Rate" 
    fg.yaxis.axis_label = "True Positive Rate"
    # get a baseline curve for comparing the others against
    base_fpr, base_tpr, _ = roc_curve(test_labels, [1 for _ in range(len(test_labels))])
    fg.line(base_fpr, base_tpr, line_width=1.5, legend_label='baseline', line_color="green")
    return fg


def add_to_performance_figure(predictions, probs, labels, fg: figure, name, color: str):
    """ Computes statistics and add ROC curve to figure from pred data"""
    print('Adding: {} to figure'.format(name))
    # Test data set results
    results = {}
    results['recall'] = recall_score(labels, predictions)
    results['precision'] = precision_score(labels, predictions)
    results['roc'] = roc_auc_score(labels, probs)
    for metric in ['recall', 'precision', 'roc']:
        print(f'{metric.capitalize()} Value: {round(results[metric], 2)}')
    # Calculate false positive rates and true positive rates, add to plot
    model_fpr, model_tpr, _ = roc_curve(labels, probs)
    fg.line(model_fpr, model_tpr, line_width=1.5, legend_label=name, line_color=color)
    return fg


def drop_columns_missing_data(df: DataFrame, th: float) -> DataFrame:
    colnames = df.columns
    miss_colnames = []
    for colname in colnames:
        miss_pct = df[colname].isnull().sum()/len(df[colname])
        if miss_pct > th:
            miss_colnames.append(colname)
    return df.drop(columns = miss_colnames)


def get_predict_and_probs(model: RandomForestClassifier, data_df: DataFrame):
    predictions = model.predict(data_df)
    output_probabilities = model.predict_proba(data_df)
    probabilities = output_probabilities[:,1]
    return predictions, probabilities



cur_path = Path(".")
all_paths = list(map(lambda p : p.as_posix(), list(cur_path.glob("2012.csv"))))
df = pd.concat(map(pd.read_csv, all_paths))

# Some data cleaning. Mostly based on notebook rec's
# There appear to be three labels, could lump to have a 3 class problem.
# a bit more complicated if you want
df['_RFHLTH'] = df['_RFHLTH'].replace({2: 0})
df = df.loc[df['_RFHLTH'].isin([0, 1])].copy()
df = df.rename(columns = {'_RFHLTH': 'label'})
df['label'].value_counts()

# Get only numeric columns; others cannot be interpreted without transformation
df = df.select_dtypes('number')

# Drop columns that seem to be aliases for the good/poor health label
df = df.drop(columns = ['POORHLTH', 'PHYSHLTH', 'GENHLTH', 'PAINACT2', 
                        'QLMENTL2', 'QLSTRES2', 'QLHLTH2', 'HLTHPLN1', 'MENTHLTH'])

# Drop columns that are more than 50% missing data
df = drop_columns_missing_data(df, 0.5)
# A quick data check to see if things look okay
# show_all_col_data(df)

labels = np.array(df.pop('label'))
train_df, test_df, train_labels, test_labels = train_test_split(df, labels, stratify=labels, test_size=0.8)

# Filling missing with a mean (probably not great practice)
train_df = train_df.fillna(train_df.mean())
test_df = test_df.fillna(test_df.mean())

# Here is theensemble. It will create the Decision Trees under the hood automatically
model = RandomForestClassifier(n_estimators=80, max_depth=10, bootstrap=True, max_features='sqrt')
model.fit(train_df, train_labels)

# Get categorical predictions on test/valid data, as well as probabalistic
test_predicts, test_probs = get_predict_and_probs(model, test_df)
train_predicts, train_probs = get_predict_and_probs(model, train_df)

# The roc curve needs probabilities so it can see how results would change
# at different sensitivities
fg = create_roc_performance_figure('ROC Curve Random Forests', test_labels)
fg = add_to_performance_figure(test_predicts, test_probs, test_labels, fg, 'Test performance', "blue")
fg = add_to_performance_figure(train_predicts, train_probs, train_labels, fg, 'Train performance', "red")
# evaluate_model(rf_predicts, rf_probs, rf_train_predicts, rf_train_probs, 'ROC Curve Random Forest')
show(fg)


## Use a Random Search to optimize model selection
# The random search will select hyperparameters in a grid search pattern
# in the ranges you specify
param_grid = {
    'n_estimators': np.linspace(10, 200).astype(int),
    'max_depth': [None] + list(np.linspace(3, 20).astype(int)),
    'max_features': ['log2', 'sqrt', None] + list(np.arange(0.5, 1, 0.1)),
    'max_leaf_nodes': [None] + list(np.linspace(10, 50, 500).astype(int)),
    'min_samples_split': [2, 5, 10],
    'bootstrap': [True, False]
}

estimator = RandomForestClassifier(random_state=1)
grid_search = RandomizedSearchCV(estimator, param_grid, n_jobs=-1, scoring='roc_auc', cv=3, n_iter=10, verbose=1, random_state=1)
grid_search.fit(train_df, train_labels)

# It saves the best version of the model automatically
best_model = grid_search.best_estimator_
print(grid_search.best_params_)

# Then use the best model to make predictions and get probabilities
test_predicts, test_probs = get_predict_and_probs(best_model, test_df)

import pdb; pdb.set_trace()
fg = add_to_performance_figure(test_predicts, test_probs, test_labels, fg, 'Grid model', "violet")
show(fg)


# Show a decision tree graphically
estimator = best_model.estimators_[1]
# Export a tree from the forest
export_graphviz(estimator, 'tree_from_optimized_forest.dot', rounded = True, 
                feature_names=train_df.columns, max_depth = 8, 
                class_names = ['Poor health', 'Good health'], filled = True)


