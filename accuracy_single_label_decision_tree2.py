from skmultilearn.dataset import load_dataset
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
#from sklearn.tree import DecisionTreeRegressor  # Import DecisionTreeRegressor
from sklearn.tree import DecisionTreeClassifier  # Import DecisionTreeRegressor
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_validate

#var="emotions"
#var="enron"
var="yeast"

# Load the emotions dataset (training)
X_train, y_train, _, _ = load_dataset(var, 'train')

# Load the emotions dataset (testing)
X_test, y_test, _, _ = load_dataset(var, 'test')


def train_and_evaluate_label_decision_tree(X_train, y_train, X_test, y_test, label_index):
    """
    Trains and evaluates a decision tree model for a single label, including test set evaluation.

    Args:
        X_train : scipy.sparse matrix
            Training feature matrix (sparse).
        y_train : scipy.sparse matrix
            Training label matrix (sparse).
        X_test : scipy.sparse matrix
            Test feature matrix (sparse).
        y_test : scipy.sparse matrix
            Test label matrix (sparse).
        label_index : int
            The index of the label (column) to be trained on.

    Returns:
        results : dict
            Dictionary containing cross-validation results, test accuracy, and best parameters.
    """
    # Convert y to dense and select the specific label (column)
    y_train_single_label = y_train.toarray()[:, label_index]  # Select label at label_index
    y_test_single_label = y_test.toarray()[:, label_index]  # Select test label at label_index

    # Initialize the Decision Tree Classifier
    tree = DecisionTreeClassifier(random_state=0)

    # Set up hyperparameter grid for tuning max_depth
    params = {"max_depth": np.arange(1, 16)}  # Adjust the range as needed
    search = GridSearchCV(tree, params, cv=10, n_jobs=2, return_train_score=True)

    # Fit the model to find the optimal hyperparameters using cross-validation
    search.fit(X_train, y_train_single_label)

    # Get the best parameters and best score from the grid search
    best_params = search.best_params_
    best_score = search.best_score_

    # Evaluate the model on the test set
    y_pred_test = search.predict(X_test)
    test_accuracy = accuracy_score(y_test_single_label, y_pred_test)

    # Prepare the results to return
    results = {
        'best_params': best_params,
        'best_score': best_score,
        'test_accuracy': test_accuracy,
        'y_pred_test': y_pred_test  # Optional: if you want to return predictions
    }

    return results



num_labels = y_train.shape[1]


for i in range(num_labels):
    # Call the function to train and evaluate the Decision Tree for label i
    results = train_and_evaluate_label_decision_tree(X_train, y_train, X_test, y_test, i)

    # Print results for the current label
    print(f"Label {i}:")
    print("Best parameters:", results['best_params'])
    print("Best cross-validation score:", results['best_score'])
    print("Test accuracy:", results['test_accuracy'])
    print("-" * 30)  # Separator for clarity

