
from sklearn.ensemble import RandomForestClassifier

from sklearn.naive_bayes import GaussianNB
from skmultilearn.problem_transform import BinaryRelevance
from sklearn.neural_network import MLPClassifier
from sklearn.multiclass import OneVsRestClassifier



import json
import os
from invoke import task
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.multioutput import MultiOutputClassifier
from config import PARAMS_FILE,param_grids,param_methods  # Import the global variable


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
    tuned_classifiers = {}

    # Convert y_train to a dense format if it is sparse
    if hasattr(y_train, 'toarray'):
        y_train_dense = y_train.toarray()
    else:
        y_train_dense = y_train

    for method in param_methods:
        for classifier_name, param_grid in param_grids.items():
            if (method=="BinaryRelevance") and  (classifier_name == 'DecisionTreeClassifier'):
                classifier = BinaryRelevance(DecisionTreeClassifier())
            elif (method=="BinaryRelevance") and  (classifier_name == 'RandomForestClassifier'):
                classifier = BinaryRelevance(RandomForestClassifier())
            if (method=="OneVsRestClassifier") and  (classifier_name == 'DecisionTreeClassifier'):
                classifier = OneVsRestClassifier(DecisionTreeClassifier())
            elif (method=="OneVsRestClassifier") and  (classifier_name == 'RandomForestClassifier'):
                classifier = OneVsRestClassifier(RandomForestClassifier())
            ## DIFFERENCE
            # la methode BinaryRelevance implique un classifier.get_params() : classifier__max_depth
            # la methode OneVsRestClassifier implique un classifier.get_params() : estimator__max_depth
            print ("method",method)
            print ("classifier",classifier_name)

            if method=="OneVsRestClassifier" :
                # We change the param grid to estimator
                param_grid = {k.replace('classifier__', 'estimator__'): v for k, v in param_grid.items()}
            grid_search = GridSearchCV(classifier, param_grid, cv=5, scoring='f1_micro')
            grid_search.fit(X_train, y_train_dense)

            if method=="OneVsRestClassifier" :
                best_params = {k.replace("estimator__", ""): v for k, v in grid_search.best_params_.items()}
            else:
                best_params = {k.replace("classifier__", ""): v for k, v in grid_search.best_params_.items()}
            tuned_classifiers[f"{classifier_name}_{method}"] = {
                'classifier': grid_search.best_estimator_,
                'params': best_params
            }
            print(f"Best hyperparameters for {classifier_name} {method}: {best_params}")
    print("=======================================================")
    return tuned_classifiers

#    for classifier_name, param_grid in param_grids.items():
#        # Initialize BinaryRelevance
#        if classifier_name == 'DecisionTreeClassifier':
#            br_classifier = BinaryRelevance(DecisionTreeClassifier())
#            ovr_classifier = OneVsRestClassifier(DecisionTreeClassifier())
#        elif classifier_name == 'RandomForestClassifier':
#            br_classifier = BinaryRelevance(RandomForestClassifier())
#            ovr_classifier = OneVsRestClassifier(RandomForestClassifier())
#
#        elif classifier_name == 'GaussianNB':
#            br_classifier = BinaryRelevance(GaussianNB())
#            ovr_classifier = OneVsRestClassifier(GaussianNB())
#        elif classifier_name == 'MLPClassifier':
#            br_classifier = BinaryRelevance(MLPClassifier())
#            ovr_classifier = OneVsRestClassifier(MLPClassifier())
#        else:
#            continue  # Skip if classifier is not recognized
#
#        # Perform hyperparameter tuning for BinaryRelevance
#        br_grid_search = GridSearchCV(br_classifier, param_grid, cv=5, scoring='f1_micro')
#        br_grid_search.fit(X_train, y_train_dense)
#        best_br_params = {k.replace("classifier__", ""): v for k, v in br_grid_search.best_params_.items()}
#        tuned_classifiers[f"{classifier_name}_BinaryRelevance"] = {
#            'classifier': br_grid_search.best_estimator_,
#            'params': best_br_params
#        }
#        print(f"Best hyperparameters for {classifier_name} (BinaryRelevance): {best_br_params}")
#
#        # Perform hyperparameter tuning for OneVsRestClassifier
#        #{'classifier__max_depth': [1, 2, 3]}
#
#        ovr_param_grid = {k.replace('classifier__', 'estimator__'): v for k, v in param_grid.items()}
#        ovr_grid_search = GridSearchCV(ovr_classifier, ovr_param_grid, cv=5, scoring='f1_micro')
#        ovr_grid_search.fit(X_train, y_train_dense)
#        best_ovr_params = {k.replace("estimator__", ""): v for k, v in ovr_grid_search.best_params_.items()}
#        tuned_classifiers[f"{classifier_name}_OneVsRest"] = {
#           'classifier': ovr_grid_search.best_estimator_,
#           'params': best_ovr_params
#       }
#       print(f"Best hyperparameters for {classifier_name} (OneVsRest): {best_ovr_params}")
#
#    return tuned_classifiers





