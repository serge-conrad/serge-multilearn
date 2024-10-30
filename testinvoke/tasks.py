# main.py

from invoke import task
from data_preparation import prepare_data
from model_evaluation import evaluate_pipeline
from tune_hyperparameters import tune_hyperparameters
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler

# List of datasets to evaluate
datasets = ["emotions", "yeast"]
# Alternatively, uncomment the datasets you want to evaluate
# datasets = ["bibtex","Corel5k","genbase","scene","yeast"]
# datasets = ["bibtex","Corel5k","enron","genbase","scene","yeast"]

# List of classifiers to evaluate
classifiers = [
    make_pipeline(StandardScaler(with_mean=False), SVC(probability=True)),
    DecisionTreeClassifier()
]

@task
def run_all(c):
    """Run all evaluations for all classifiers and datasets."""
    for var in datasets:
        # Prepare the data
        X_train, y_train, X_test, y_test = prepare_data(c, var)

        # Tune hyperparameters and get best classifiers
        #best_classifiers = tune_hyperparameters(c, X_train, y_train)

        # Evaluate the best classifiers
        #for classifier_name, best_classifier in best_classifiers.items():
        #    print(f"Evaluating best classifier: {classifier_name}")
        #    evaluate_pipeline(c, best_classifier, X_train, y_train, X_test, y_test)


        for classifier in classifiers:
            evaluate_pipeline(c, classifier, X_train, y_train, X_test, y_test)

# If this script is the main module, run the evaluations
if __name__ == "__main__":
    run_all(c)
