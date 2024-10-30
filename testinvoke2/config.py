# config.py
PARAMS_FILE = 'best_hyperparameters.json'

# List of datasets to evaluate
datasets = ["emotions", "yeast"]

# Define parameter grids for each classifier
param_grids = {
    'SVC': {
        'estimator__svc__C': [0.1, 1, 10],
        'estimator__svc__gamma': [0.01, 0.1, 1]
    },
    'DecisionTreeClassifier': {
        'estimator__max_depth': [None, 5, 10, 15],
        'estimator__min_samples_split': [2, 5, 10]
    }
}

