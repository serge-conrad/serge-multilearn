
from sklearn.ensemble import RandomForestClassifier

from sklearn.naive_bayes import GaussianNB
from skmultilearn.problem_transform import BinaryRelevance
from sklearn.neural_network import MLPClassifier
from sklearn.multiclass import OneVsRestClassifier


from iterativeStratification import CustomStratifiedKFold


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
from config import param_splitting  # Import the global variable



def save_hyperparameters(best_classifiers, dataset, classifier_name):
    """Save the best hyperparameters for a specific classifier in a dataset to a JSON file."""

    # Load existing data or start with an empty dictionary
    if os.path.exists(PARAMS_FILE):
        with open(PARAMS_FILE, 'r') as file:
            existing_data = json.load(file)
    else:
        existing_data = {}

    # Ensure the dataset exists in the file
    if dataset not in existing_data:
        existing_data[dataset] = {}

    # Iterate over classifier variants and store hyperparameters
    for variant_name, classifier_data in best_classifiers.items():
        # Clean parameter keys by removing prefixes like 'estimator__' or 'svc__'
        cleaned_params = {
            key.replace('estimator__', '').replace('svc__', ''): value
            for key, value in classifier_data['params'].items()
        }

        # Ensure the classifier_name exists and add the variant
        if classifier_name not in existing_data[dataset]:
            existing_data[dataset][classifier_name] = {}

        existing_data[dataset][classifier_name][variant_name] = {"params": cleaned_params}

    # Save the updated data back to the file
    with open(PARAMS_FILE, 'w') as file:
        json.dump(existing_data, file, indent=4)

    print(f"Hyperparameters for all variants of '{classifier_name}' in dataset '{dataset}' saved successfully.")



#def save_hyperparameters(best_classifiers, dataset, classifier_name):
#    """Save the best hyperparameters for a specific classifier in a dataset to a JSON file."""
#    print(best_classifiers)
#    print(dataset)
#    print(classifier_name)
#    # Clean parameter keys by removing prefixes like 'estimator__' or 'svc__'
#    cleaned_params = {
#        key.replace('estimator__', '').replace('svc__', ''): value
#        for key, value in best_classifiers['params'].items()
#    }
#    classifier_data = {
#        'params': cleaned_params
#    }
#
#    # Load existing data or start with an empty dictionary
#    if os.path.exists(PARAMS_FILE):
#        with open(PARAMS_FILE, 'r') as file:
#            existing_data = json.load(file)
#    else:
#        existing_data = {}
#
#    # Ensure dataset exists in the file and update classifier data
#    if dataset not in existing_data:
#        existing_data[dataset] = {}
#
#    existing_data[dataset][classifier_name] = classifier_data
#
#    # Save the updated data back to the file
#    with open(PARAMS_FILE, 'w') as file:
#        json.dump(existing_data, file, indent=4)
#



def load_hyperparameters(dataset, classifier_name):
    """Load the best hyperparameters for a specific dataset and classifier from a JSON file."""
    try:
        with open(PARAMS_FILE, 'r') as file:
            data = json.load(file)

            # Vérifie si le dataset existe dans le fichier
            if dataset not in data:
                print(f"Dataset '{dataset}' not found in {PARAMS_FILE}.")
                return {}

            dataset_data = data[dataset]

            # Vérifie si le classificateur existe pour ce dataset
            if classifier_name not in dataset_data:
                print(f"Classifier '{classifier_name}' not found for dataset '{dataset}'.")
                return {}

            # Retourne toutes les variantes du classificateur avec leurs hyperparamètres
            return dataset_data[classifier_name]

    except FileNotFoundError:
        print(f"{PARAMS_FILE} not found. No hyperparameters loaded.")
        return {}
    except json.JSONDecodeError:
        print(f"Error decoding JSON from {PARAMS_FILE}. Please check the file format.")
        return {}





#def load_hyperparameters(dataset):
#    """Load the best hyperparameters for a specific dataset from a JSON file."""
#    try:
#        with open(PARAMS_FILE, 'r') as file:
#            data = json.load(file)
#            if dataset not in data:
#                print (f"Dataset '{dataset}' not found in {PARAMS_FILE}.")
#                return {}
#
#            return data[dataset]
#    except FileNotFoundError:
#        print(f"{PARAMS_FILE} not found. No hyperparameters loaded.")
#        return {}


@task

#def tune_hyperparameters(c, X_train, y_train):
def tune_hyperparameters(c, X_train, y_train,classifier_name,param_grid_ori):
    """Tune hyperparameters for classifiers and return the best classifier and its parameters."""
    tuned_classifiers = {}

    # Convert y_train to a dense format if it is sparse
    if hasattr(y_train, 'toarray'):
        y_train_dense = y_train.toarray()
    else:
        y_train_dense = y_train

    for method in param_methods:

            if (method=="BinaryRelevance") and  (classifier_name == 'DecisionTreeClassifier'):
                classifier = BinaryRelevance(DecisionTreeClassifier())
            elif (method=="BinaryRelevance") and  (classifier_name == 'RandomForestClassifier'):
                classifier = BinaryRelevance(RandomForestClassifier())
            if (method=="OneVsRestClassifier") and  (classifier_name == 'DecisionTreeClassifier'):
                classifier = OneVsRestClassifier(DecisionTreeClassifier())
            elif (method=="OneVsRestClassifier") and  (classifier_name == 'RandomForestClassifier'):
                classifier = OneVsRestClassifier(RandomForestClassifier())
            elif (method=="OneVsRestClassifier") and  (classifier_name == 'MLPClassifier'):
                classifier = OneVsRestClassifier(MLPClassifier())
            elif (method=="BinaryRelevance") and  (classifier_name == 'MLPClassifier'):
                classifier = BinaryRelevance(MLPClassifier())
            else:
                print("ERREUR CODE 2")

            ## DIFFERENCE
            # la methode BinaryRelevance implique un classifier.get_params() : classifier__max_depth
            # la methode OneVsRestClassifier implique un classifier.get_params() : estimator__max_depth


            for splitting in param_splitting:
                if (splitting=="standard"):


                    if method=="OneVsRestClassifier" :
                        # We change the param grid to estimator
                        param_grid = {k.replace('classifier__', 'estimator__'): v for k, v in param_grid_ori.items()}
                    else:
                        param_grid =  param_grid_ori
                    grid_search = GridSearchCV(classifier, param_grid, cv=5, scoring='f1_micro',n_jobs=-1)
                    grid_search.fit(X_train, y_train_dense)

                    if method=="OneVsRestClassifier" :
                        best_params = {k.replace("estimator__", ""): v for k, v in grid_search.best_params_.items()}
                    else:
                        best_params = {k.replace("classifier__", ""): v for k, v in grid_search.best_params_.items()}
                    tuned_classifiers[f"{method}_{splitting}"] = {
                        'params': best_params
                    }
                    #'classifier': grid_search.best_estimator_,

                elif (splitting=="iterativestratification"):
                    custom_cv = CustomStratifiedKFold(n_splits=5, shuffle=True, random_state=42)

                    #clf = RandomForestClassifier()
                    if method=="OneVsRestClassifier" :
                        # We change the param grid to estimator
                        param_grid = {k.replace('classifier__', 'estimator__'): v for k, v in param_grid_ori.items()}
                    else:
                        param_grid =  param_grid_ori
                    grid_search = GridSearchCV(classifier, param_grid, cv=custom_cv, scoring='f1_micro',n_jobs=-1)
                    grid_search.fit(X_train, y_train_dense)
                    if method=="OneVsRestClassifier" :
                        best_params = {k.replace("estimator__", ""): v for k, v in grid_search.best_params_.items()}
                    else:
                        best_params = {k.replace("classifier__", ""): v for k, v in grid_search.best_params_.items()}

                    tuned_classifiers[f"{method}_{splitting}"] = {
                        'params': best_params
                    }
                print(f"Best hyperparameters for {splitting} {classifier_name} {method}: {best_params}")




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





