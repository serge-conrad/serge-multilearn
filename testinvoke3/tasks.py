# main.py




import json
import os
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
from config import PARAMS_FILE,datasets  # Import the global variable




def create_classifier(classifier_name, params):
#make_pipeline(StandardScaler(with_mean=False), SVC(probability=True)),
#    DecisionTreeClassifier()

    """Reconstruct the classifier based on its type and parameters."""
    if classifier_name == 'SVC':
        classifier = make_pipeline(StandardScaler(with_mean=False), SVC(probability=True, **params))
    elif classifier_name == 'DecisionTreeClassifier':
        classifier = DecisionTreeClassifier(**params)
    else:
        raise ValueError(f"Unknown classifier: {classifier_name}")

    return classifier


@task
def run_all(c):
    """Run all evaluations for all classifiers and datasets."""
    for var in datasets:
        # Prepare the data
        X_train, y_train, X_test, y_test = prepare_data(c, var)


        # SERGENEW
        # Load hyperparameters for the current dataset
        best_classifiers = load_hyperparameters(var)

        # Check if the hyperparameters for this dataset exist
        if not best_classifiers:
            # Tune hyperparameters if they do not exist
            best_classifiers = tune_hyperparameters(c, X_train, y_train)
            print("Tuned hyperparameters for dataset:", var)
            print(best_classifiers)
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
            evaluate_pipeline(c, best_classifier, X_train, y_train, X_test, y_test)

        # Evaluate the best classifiers
        #for classifier_name, best_info in best_classifiers.items():
        #    best_classifier = best_info['classifier']
        #    best_params = best_info['params']
        #    print(f"Evaluating classifier: {classifier_name} with parameters: {best_params}")
        #    evaluate_pipeline(c, best_classifier, best_params, X_train, y_train, X_test, y_test)

# If this script is the main module, run the evaluations
if __name__ == "__main__":
    from invoke import Collection
    ns = Collection(run_all)
    ns.configure({
        'run_all': run_all
    })

