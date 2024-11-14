# main.py
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import cross_val_score







from tests import test0
from tests import test1
from tests import teststratkfold
from tests import testrepeatstratkfold
from tests import testshuffle
from tests import teststratcustom
#from tests import test2

import json
import os
import time
from invoke import task
from data_preparation import prepare_data
from tune_hyperparameters import tune_hyperparameters,load_hyperparameters,save_hyperparameters
from model_evaluation import evaluate_pipeline
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.multioutput import MultiOutputClassifier
from config import PARAMS_FILE,RESULT_FILE,datasets  # Import the global variable
from sklearn.neural_network import MLPClassifier


from iterativeStratification import CustomStratifiedKFold


def create_classifier(classifier_name, params):
#make_pipeline(StandardScaler(with_mean=False), SVC(probability=True)),
#    DecisionTreeClassifier()

    """Reconstruct the classifier based on its type and parameters."""
    if classifier_name == 'SVC':
        classifier = make_pipeline(StandardScaler(with_mean=False), SVC(probability=True, **params))
    elif classifier_name.startswith('DecisionTreeClassifier_BinaryRelevance'):
        classifier = BinaryRelevance( DecisionTreeClassifier(**params) )
    elif classifier_name.startswith('DecisionTreeClassifier_OneVsRestClassifier'):
        classifier = OneVsRestClassifier(DecisionTreeClassifier(**params))
        #classifier = DecisionTreeClassifier(**params)
    elif classifier_name.startswith('RandomForestClassifier_BinaryRelevance'):
        classifier = BinaryRelevance( RandomForestClassifier(**params) )
    elif classifier_name.startswith('RandomForestClassifier_OneVsRestClassifier'):
        classifier = OneVsRestClassifier(RandomForestClassifier(**params))



        #classifier = RandomForestClassifier(**params)

    elif classifier_name == 'GaussianNB':
       classifier = BinaryRelevance( GaussianNB(**params) )
    elif classifier_name == 'MLPClassifier':
        classifier = BinaryRelevance(MLPClassifier(**params))



    else:
        raise ValueError(f"Unknown classifier: {classifier_name}")

    return classifier


from config import param_splitting  # Import the global variable

from skmultilearn.problem_transform import BinaryRelevance
@task
def run_all(c):
    """Run all evaluations for all classifiers and datasets."""

    start_time = time.time()
    with open(RESULT_FILE, 'w') as file:
        file.write("dataset;classifier;params;f1_micro;auc_roc;auc_pr;debug\n")
    for var in datasets:
        # Prepare the data
        # do basically a load_dataset from skmultilearn
        X_train, y_train, X_test, y_test = prepare_data(c, var)


        # Load hyperparameters for the current dataset
        # they are stored in a json file best_hyperparameters.json
        best_classifiers = load_hyperparameters(var)

        # Check if the json file do not exist
        if not best_classifiers:
                    # Calculating best hyperparameters for the dataset
                    # Tune hyperparameters if they do not exist
                    best_classifiers = tune_hyperparameters(c, X_train, y_train)
                    print("Tuned hyperparameters for dataset:", var)
                    # Save the best hyperparameters to file
                    save_hyperparameters(best_classifiers, var)
                    # ON recharge pour éliminer estimator__svc__C
                    best_classifiers = load_hyperparameters(var)
    
        print("Loaded hyperparameters for dataset:", var)


        # Evaluate the best classifiers
        for classifier_name, best_info in best_classifiers.items():
                    print(classifier_name,best_info)
                    best_params = best_info['params']
                    # Recreate the classifier with the best parameters
                    best_classifier = create_classifier(classifier_name, best_params)
                    print(f"Evaluating classifier: {classifier_name} with parameters: {best_params}")
                    evaluate_pipeline(c, var,classifier_name,best_params,best_classifier, X_train, y_train, X_test, y_test)

    end_time = time.time()

    # Affiche le temps d'exécution
    print(f"Temps d'exécution : {end_time - start_time} secondes")




# If this script is the main module, run the evaluations
if __name__ == "__main__":
    from invoke import Collection
    ns = Collection(run_all)
    ns.configure({
        'run_all': run_all
    })

