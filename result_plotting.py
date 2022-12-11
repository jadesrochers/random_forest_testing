from sklearn.metrics import precision_score, recall_score, roc_auc_score, roc_curve
from bokeh.plotting import figure

## Plot results from classification problems. This
# handles binary, probably have to modify it for multi class case.

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


