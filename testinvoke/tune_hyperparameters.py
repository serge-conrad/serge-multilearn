from invoke import task
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.multioutput import MultiOutputClassifier
from sklearn.tree import DecisionTreeClassifier

#'svc__C': [0.1, 1, 10],
#'svc__gamma': [0.01, 0.1, 1]
#'classifier__max_depth': [None, 5, 10, 15],
#'classifier__min_samples_split': [2, 5, 10]
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

@task
def tune_hyperparameters(c, X_train, y_train):
    """Tune hyperparameters for classifiers."""
    best_classifiers = {}

    # 
    for classifier_name, param_grid in param_grids.items():
        if classifier_name == 'SVC':
            #classifier = make_pipeline(StandardScaler(with_mean=False), SVC(probability=True))
            classifier = MultiOutputClassifier(make_pipeline(StandardScaler(with_mean=False), SVC(probability=True)))
        elif classifier_name == 'DecisionTreeClassifier':
            classifier = MultiOutputClassifier(DecisionTreeClassifier())
            #classifier = DecisionTreeClassifier()
        else:
            continue  # Skip if classifier is not recognized

        # Perform hyperparameter tuning
        print('aà')
        grid_search = GridSearchCV(classifier, param_grid, cv=5, scoring='f1_micro')
        print('ab')
        grid_search.fit(X_train, y_train.toarray())
        print('ac')

        # Store the best classifier
        best_classifiers[classifier_name] = grid_search.best_estimator_
        print(f"Best hyperparameters for {classifier_name}: {grid_search.best_params_}")

    return best_classifiers



