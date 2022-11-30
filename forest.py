from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import pandas as pd
from pandas import DataFrame
import numpy as np
from pathlib import Path

def show_all_col_data(df: DataFrame):
    # Use set_option to change option, option_context to change just in with: block
    # pd.set_option('display.max_columns', None)
    with pd.option_context('display.max_columns', None):
        print(df.head())


def nodes_and_depth(model: RandomForestClassifier): 
    nodes = []
    depth = []
    for tree in model.estimators_:
        nodes.append(tree._tee_.node_count)
        depth.append(tree.tree_.max_depth)
    print(f'Avg num nodes: {int(np.mean(nodes))}')
    print(f'Avg maximum depth: {int(np.mean(nodes))}')


def evaluate_model(predictions, probs, train_predictions, train_probs):
    """Compare machine learning model to baseline performance.
    Computes statistics and shows ROC curve."""
    
    baseline = {}
    
    import pdb; pdb.set_trace()
    # Baseline just means predict everything in a single category and see how that shakes out?
    baseline['recall'] = recall_score(test_labels, [1 for _ in range(len(test_labels))])
    baseline['precision'] = precision_score(test_labels, [1 for _ in range(len(test_labels))])
    baseline['roc'] = 0.5
    
    results = {}
    
    results['recall'] = recall_score(test_labels, predictions)
    results['precision'] = precision_score(test_labels, predictions)
    results['roc'] = roc_auc_score(test_labels, probs)
    
    train_results = {}
    train_results['recall'] = recall_score(train_labels, train_predictions)
    train_results['precision'] = precision_score(train_labels, train_predictions)
    train_results['roc'] = roc_auc_score(train_labels, train_probs)
    
    for metric in ['recall', 'precision', 'roc']:
        print(f'{metric.capitalize()} Baseline: {round(baseline[metric], 2)} Test: {round(results[metric], 2)} Train: {round(train_results[metric], 2)}')
    
    # Calculate false positive rates and true positive rates
    base_fpr, base_tpr, _ = roc_curve(test_labels, [1 for _ in range(len(test_labels))])
    model_fpr, model_tpr, _ = roc_curve(test_labels, probs)

    plt.figure(figsize = (8, 6))
    plt.rcParams['font.size'] = 16
    
    # Plot both curves
    plt.plot(base_fpr, base_tpr, 'b', label = 'baseline')
    plt.plot(model_fpr, model_tpr, 'r', label = 'model')
    plt.legend();
    plt.xlabel('False Positive Rate'); plt.ylabel('True Positive Rate'); plt.title('ROC Curves');


cur_path = Path(".")
all_paths = list(map(lambda p : p.as_posix(), list(cur_path.glob("2012.csv"))))
df = pd.concat(map(pd.read_csv, all_paths))

# Some data cleaning. This is recommended by the notebook and I am following that 
# for the sake of keeping things fast right now. Edit or adjust if you want.
# There appear to be three labels, so you could avoid the lumping step to make things 
# a bit more complicated if you want
df['_RFHLTH'] = df['_RFHLTH'].replace({2: 0})
df = df.loc[df['_RFHLTH'].isin([0, 1])].copy()
df = df.rename(columns = {'_RFHLTH': 'label'})
df['label'].value_counts()
import pdb; pdb.set_trace()
show_all_col_data(df)

# And drop columns that should not be used in the model - some of these seem
# to be aliases for the label column

labels = np.array(df.pop('label'))
train_df, test_df, train_labels, test_labels = train_test_split(df, labels, test_size=0.4)

# Here is theensemble. It will create the Decision Trees under the hood automatically
model = RandomForestClassifier(n_estimators=50, max_depth=20, bootstrap=True, max_features='sqrt')
model.fit(train_df, train_labels)

# Get categorical predictions on test/valid data, as well as probabalistic
rf_predicts = model.predict(test_df)
rf_outputs = model.predict_proba(test_df)
rf_probs = rf_outputs[:,1]

# Get the data for training data as well, see how performance and training performance relate
rf_train_predicts = model.predict(train_df)
rf_train_outputs = model.predict_proba(train_df)
rf_train_probs = rf_outputs[:,1]

# The roc curve needs probabilities so it can see how results would change
# at different sensetivities
roc_value = roc_auc_score(test_labels, rf_probs)
evaluate_model(rf_predicts, rf_probs, rf_train_predicts, rf_train_probs)


