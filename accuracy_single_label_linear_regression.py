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



#def train_and_evaluate_label_decision_tree(X, y, label_index):
#    y_train_single_label = y.toarray()[:, label_index]  # Select label at label_index
#    tree = DecisionTreeRegressor(random_state=0)
#
#    params = {"max_depth": np.arange(1, 16)}
#    search = GridSearchCV(tree, params, cv=10)
#
#    # On va lancer 10 cross_validate, dans chaque cas on vas recalculer le bon hyperparam....
#    cv_results_tree_optimal_depth = cross_validate(
#        search,X , y_train_single_label, cv=10, return_estimator=True, n_jobs=2,
#    )


#    return cv_results_tree_optimal_depth






def train_and_evaluate_label(X, y, label_index):
    """
    Trains and evaluates a logistic regression model for a single label.
    
    Args:
    X : scipy.sparse matrix
        Feature matrix (sparse).
    y : scipy.sparse matrix
        Label matrix (sparse).
    label_index : int
        The index of the label (column) to be trained on.
    
    Returns:
    accuracy : float
        Accuracy of the model for the specific label.
    """
    # Convert y to dense and select the specific label (column)
    y_train_single_label = y.toarray()[:, label_index]  # Select label at label_index

    # Create a pipeline with scaling (without centering due to sparse data) and logistic regression
    pipeline = make_pipeline(StandardScaler(with_mean=False), LogisticRegression(max_iter=1000))

    # Train the classifier
    pipeline.fit(X, y_train_single_label)

    # Predict on the test set
    y_test_single_label = y_test.toarray()[:, label_index]  # Get the corresponding test label
    y_pred = pipeline.predict(X_test)

    # Calculate accuracy for this label
    accuracy = accuracy_score(y_test_single_label, y_pred)
    return accuracy

# Get the number of labels (columns) in y_train
num_labels = y_train.shape[1]

# Iterate over all labels and evaluate using the test dataset
for i in range(num_labels):
    accuracy = train_and_evaluate_label(X_train, y_train, i)
    print(f"Accuracy for label {i}: {accuracy}")

# Optionally, you can also evaluate a specific label
# Uncomment to evaluate a specific label (e.g., label 0)
# accuracy_single_label = train_and_evaluate_label(X_train, y_train, 0)
# print(f"Accuracy for single label 0: {accuracy_single_label}")




