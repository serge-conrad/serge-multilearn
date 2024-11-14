# model_evaluation.py

from invoke import task
from sklearn.metrics import f1_score, roc_auc_score, average_precision_score
from skmultilearn.problem_transform import BinaryRelevance
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from config import RESULT_FILE  # Import the global variable


@task
def evaluate_pipeline(c, dataset, classifier_name,best_params,classifier, X_train, y_train, X_test, y_test):
    print(f"=========Evaluating: {classifier}")
    clf=classifier
    #clf = BinaryRelevance(
    #    classifier=classifier,
    #    require_dense=[False, True]
    #)

    # Fit the model on the training data
    clf.fit(X_train, y_train)

    # Make predictions on the test set
    predictions = clf.predict(X_test)

    # Ensure y_test is in dense format
    y_test_dense = y_test.toarray() if hasattr(y_test, 'toarray') else y_test

    # Calculate F1 Micro
    f1_micro = f1_score(y_test_dense, predictions, average='micro')

    # Get predicted probabilities
    predicted_probabilities = clf.predict_proba(X_test)
    predicted_probabilities_dense = predicted_probabilities.toarray() if hasattr(predicted_probabilities, 'toarray') else predicted_probabilities

    # AUC-ROC Calculation
    auc_roc = roc_auc_score(y_test_dense, predicted_probabilities_dense, average='macro')

    # AUC-PR Calculation
    auc_pr = average_precision_score(y_test_dense, predicted_probabilities_dense, average='macro')

    with open(RESULT_FILE, "a") as file:
        # Écrire toutes les valeurs sur une seule ligne
        file.write(f"{dataset};{classifier_name};{best_params};{f1_micro};{auc_roc};{auc_pr}\n")


    # Print the evaluation results
    print("F1 Micro:", f1_micro)
    print("AUC-ROC:", auc_roc)
    print("AUC-PR:", auc_pr)

