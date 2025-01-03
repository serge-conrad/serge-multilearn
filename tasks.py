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
from config import PARAMS_FILE,RESULT_FILE,datasets,param_grids  # Import the global variable

from sklearn.neural_network import MLPClassifier


from iterativeStratification import CustomStratifiedKFold


def create_classifier(classifier_name,method, params):
#make_pipeline(StandardScaler(with_mean=False), SVC(probability=True)),
#    DecisionTreeClassifier()
    #classifier_name,BinaryRelevance_iterativestratification

    """Reconstruct the classifier based on its type and parameters."""
    if classifier_name == 'SVC':
        classifier = make_pipeline(StandardScaler(with_mean=False), SVC(probability=True, **params))
    elif classifier_name=='DecisionTreeClassifier' and method.startswith('BinaryRelevance'):
        classifier = BinaryRelevance( DecisionTreeClassifier(**params) )
    elif classifier_name=='DecisionTreeClassifier' and method.startswith('OneVsRestClassifier'):
        classifier = OneVsRestClassifier(DecisionTreeClassifier(**params))
        #classifier = DecisionTreeClassifier(**params)
    elif classifier_name=='RandomForestClassifier' and method.startswith('BinaryRelevance'):
        classifier = BinaryRelevance( RandomForestClassifier(**params) )
    elif classifier_name=='RandomForestClassifier' and method.startswith('OneVsRestClassifier'):
        classifier = OneVsRestClassifier(RandomForestClassifier(**params))
    elif classifier_name=='MLPClassifier' and method.startswith('BinaryRelevance'):
        classifier = BinaryRelevance(MLPClassifier (**params) )
    elif classifier_name=='MLPClassifier' and method.startswith('OneVsRestClassifier'):
        classifier = OneVsRestClassifier(MLPClassifier(**params))



        #classifier = RandomForestClassifier(**params)

    elif classifier_name == 'GaussianNB':
       classifier = BinaryRelevance( GaussianNB(**params) )



    else:
        raise ValueError(f"Unknown classifier: {classifier_name}")

    return classifier



def check_all_params_changes(param_grids, existing_hyperparameters):
    recalculate = False
    for hyperparameter, grid in param_grids.items():
        #print("A",hyperparameter)
        #A max_depth
        #print("B",grid)
        #B [1, 2, 3, 4]
        for var_name, var_data in existing_hyperparameters.items():
            #print("C",var_name)
            #C BinaryRelevance_iterativestratification
            #BinaryRelevance_standard
            #print("D",var_data['range'][hyperparameter])
            if var_data['range'][hyperparameter] != grid:
                print("NOT EQUAL")
                recalculate = True
                break

            #D {'max_depth': [1, 2, 3]}

            #D {'params': {'max_depth': 1}, 'range': {'max_depth': [1, 2, 3]}}
    return recalculate 




def check_all_params_exist(param_grids, existing_hyperparameters):
    #missing_params = []


    #print(param_grids)
    #{'max_depth': [1, 2, 3]}

    #print(existing_hyperparameters)

    # Pour chaque classificateur et ses paramètres dans param_grids
    for hyperparameter, grid in param_grids.items():
        # Vérifier si ce paramètre est dans les hyperparamètres existants pour ce classificateur
        recalculate = False
        for var_name, var_data in existing_hyperparameters.items():
                if (hyperparameter not in var_data['params']):
                    recalculate = True
                    break


    return recalculate 



from config import param_splitting  # Import the global variable

from skmultilearn.problem_transform import BinaryRelevance
@task
def run_all(c):
    """Run all evaluations for all classifiers and datasets."""

    start_time = time.time()
    with open(RESULT_FILE, 'w') as file:
        file.write("dataset;classifier;method;params;f1_micro;auc_roc;auc_pr;debug\n")
    for var in datasets:
        print("working on dataset ",var)
        # Prepare the data
        # do basically a load_dataset from skmultilearn
        X_train, y_train, X_test, y_test = prepare_data(c, var)


        # pour les classificateurs dans le fichier config
        for classifier_name, param_grid in param_grids.items():


            # Load hyperparameters for the current dataset
            # they are stored in a json file best_hyperparameters.json
            best_classifiers = load_hyperparameters(var,classifier_name)
            # permet de rajouter des dataset 
            # Check if the json file do not exist
            if not best_classifiers:

                    print("Calculating Tuned hyperparameters for dataset and classifier:", var,classifier_name)
                    # Calculating best hyperparameters for the dataset
                    # Tune hyperparameters if they do not exist
                    best_classifiers = tune_hyperparameters(c, X_train, y_train,classifier_name,param_grid)
                    # Save the best hyperparameters to file
                    save_hyperparameters(best_classifiers, var,classifier_name,param_grid)
                    # ON recharge pour éliminer estimator__svc__C
                    best_classifiers = load_hyperparameters(var,classifier_name)
            else:
                # json file present
                # checking if all hyperparameter are present
                missing_params = check_all_params_exist(param_grid, best_classifiers)
                if missing_params:
                        print(f"{classifier_name} certain hyperparamètres sont manquants : ")
                        best_classifiers = tune_hyperparameters(c, X_train, y_train,classifier_name,param_grid)
                        save_hyperparameters(best_classifiers, var,classifier_name,param_grid)
                else:
                    changing_params = check_all_params_changes(param_grid, best_classifiers)
                    if changing_params:
                        print(f"{classifier_name}  hyperparamètres have changed : ")
                        best_classifiers = tune_hyperparameters(c, X_train, y_train,classifier_name,param_grid)
                        save_hyperparameters(best_classifiers, var,classifier_name,param_grid)
    
            print("Loaded hyperparameters for dataset and classifier:", var,classifier_name)


            # Evaluate the best classifiers
            for method, best_info in best_classifiers.items():
                    best_params = best_info['params']
                    # Recreate the classifier with the best parameters
                    best_classifier = create_classifier(classifier_name,method, best_params)
                    print(f"Evaluating classifier: {classifier_name} method: {method} with parameters: {best_params}")
                    evaluate_pipeline(c, var,classifier_name,method,best_params,best_classifier, X_train, y_train, X_test, y_test)

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

