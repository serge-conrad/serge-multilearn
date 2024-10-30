
# Import necessary libraries
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import f1_score, roc_auc_score, average_precision_score
from skmultilearn.problem_transform import BinaryRelevance
from skmultilearn.dataset import load_dataset

# List of datasets to evaluate
datasets = ["emotions",  "yeast"]
#datasets = ["bibtex","Corel5k","genbase","scene","yeast"]
#datasets = ["bibtex","Corel5k","enron","genbase","scene","yeast"]
# List of classifiers to evaluate
classifiers = [
    make_pipeline(StandardScaler(with_mean=False), SVC(probability=True)),
    DecisionTreeClassifier()
]

# randomforest
# Function to evaluate the pipeline
def evaluate_pipeline(classifier, X_train, y_train, X_test, y_test):
    print(f"Evaluating: {classifier}")
    clf = BinaryRelevance(
        classifier=classifier,
        require_dense=[False, True]
    )

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

    # Print the evaluation results
    print("F1 Micro:", f1_micro)
    print("AUC-ROC:", auc_roc)
    print("AUC-PR:", auc_pr)

# Loop over datasets and classifiers
for var in datasets:
    # Load the dataset (training)
    X_train, y_train, _, _ = load_dataset(var, 'train')

    # Load the dataset (testing)
    X_test, y_test, _, _ = load_dataset(var, 'test')

    for classifier in classifiers:
        evaluate_pipeline(classifier, X_train, y_train, X_test, y_test)

