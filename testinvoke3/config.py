# config.py
PARAMS_FILE = 'best_hyperparameters.json'

# List of datasets to evaluate
datasets = ["emotions", "yeast"]
#datasets = ["emotions"]
#datasets = ["mediamill"]
#datasets = ["delicious","emotions"]
#available dataset {'delicious', 'genbase', 'Corel5k', 'mediamill', 'yeast', 'birds', 'rcv1subset4', 'rcv1subset5', 'enron', 'rcv1subset2', 'bibtex', 'medical', 'rcv1subset1', 'tmc2007_500', 'rcv1subset3', 'scene', 'emotions'}


# Define parameter grids for each classifier
param_grids = {
    'DecisionTreeClassifier': {
        'estimator__max_depth': [1, 5],
        'estimator__min_samples_split': [2, 5]
    }
}

#param_grids = {
#    'SVC': {
#        'estimator__svc__C': [0.1, 1, 10],
#        'estimator__svc__gamma': [0.01, 0.1, 1]
#    },
#    'DecisionTreeClassifier': {
#        'estimator__max_depth': [None, 5, 10, 15],
#        'estimator__min_samples_split': [2, 5, 10]
#    }
#}
