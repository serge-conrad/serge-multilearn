


import json
import os
from invoke import task
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.multioutput import MultiOutputClassifier
from config import PARAMS_FILE,param_grids  # Import the global variable


def save_hyperparameters(best_classifiers, dataset):
    """Save the best hyperparameters for a specific dataset to a JSON file."""
    # Prepare a serializable dictionary
    serializable_classifiers = {}

    for classifier_name, best_info in best_classifiers.items():
        # Clean the parameter keys by removing 'estimator__' and 'svc__' prefixes
        cleaned_params = {
            key.replace('estimator__', '').replace('svc__', ''): value
            for key, value in best_info['params'].items()
        }
        serializable_classifiers[classifier_name] = {
            'params': cleaned_params
        }

    # Save with dataset key
    if os.path.exists(PARAMS_FILE):
        with open(PARAMS_FILE, 'r') as file:
            existing_data = json.load(file)
    else:
        existing_data = {}

    existing_data[dataset] = serializable_classifiers

    with open(PARAMS_FILE, 'w') as file:
        json.dump(existing_data, file, indent=4)

#def load_hyperparameters(dataset):
#    """Load the best hyperparameters for a specific dataset from a JSON file."""
#    try:
#        with open(PARAMS_FILE, 'r') as file:
#            data = json.load(file)
#            return data.get(dataset, {})
#    except FileNotFoundError:
#        print(f"{PARAMS_FILE} not found. No hyperparameters loaded.")
#        return {}


def load_hyperparameters(dataset):
    """Load the best hyperparameters for a specific dataset from a JSON file."""
    try:
        with open(PARAMS_FILE, 'r') as file:
            data = json.load(file)
            if dataset not in data:
                print (f"Dataset '{dataset}' not found in {PARAMS_FILE}.")
                return {}

            return data[dataset]
    except FileNotFoundError:
        print(f"{PARAMS_FILE} not found. No hyperparameters loaded.")
        return {}



@task
def tune_hyperparameters(c, X_train, y_train):
    """Tune hyperparameters for classifiers and return the best classifier and its parameters."""
    #best_classifiers = load_hyperparameters()
    tuned_classifiers = {}

    # Convert y_train to a dense format if it is sparse
    if hasattr(y_train, 'toarray'):
        y_train_dense = y_train.toarray()
    else:
        y_train_dense = y_train

    #'classifier': best_classifiers[classifier_name]['classifier'],
    #            'params': best_classifiers[classifier_name]['params']
    for classifier_name, param_grid in param_grids.items():
        #if classifier_name in best_classifiers:
        #    print(f"Loading best hyperparameters for {classifier_name}: {best_classifiers[classifier_name]}")
        #    tuned_classifiers[classifier_name] = {
        #        'classifier_type': classifier_name,
        #        'params': best_classifiers[classifier_name]['params']
        #    }
        #else:
            if classifier_name == 'SVC':
                # Create a pipeline for SVC
                classifier = MultiOutputClassifier(make_pipeline(StandardScaler(with_mean=False), SVC(probability=True)))
            elif classifier_name == 'DecisionTreeClassifier':
                # Create a pipeline for DecisionTreeClassifier
                classifier = MultiOutputClassifier(DecisionTreeClassifier())
            else:
                continue  # Skip if classifier is not recognized

            # Perform hyperparameter tuning
            grid_search = GridSearchCV(classifier, param_grid, cv=5, scoring='f1_micro')
            grid_search.fit(X_train, y_train_dense)  # Use the dense y_train

            # Store the best classifier and its parameters
            tuned_classifiers[classifier_name] = {
                'classifier': grid_search.best_estimator_,
                'params': grid_search.best_params_
            }
            print(f"Best hyperparameters for {classifier_name}: {grid_search.best_params_}")

    #save_hyperparameters(tuned_classifiers)
    return tuned_classifiers


