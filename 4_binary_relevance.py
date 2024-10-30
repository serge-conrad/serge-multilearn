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
from skmultilearn.problem_transform import BinaryRelevance
import sklearn.metrics as metrics
from sklearn.svm import SVC
from sklearn.metrics import roc_auc_score, average_precision_score

import sklearn
print("sklearnversion",sklearn.__version__)

#var="emotions"
#var="enron"
print("converts the multi-label problem into multiple independent binary classification problems (one for each label).")
print("classifier=SVC(): The base classifier for each of these independent binary classification problems is an SVM (Support Vector Machine), implemented by SVC() from the scikit-learn library.")
print("require_dense=[False, True], premier argument X, deuxième y")



var="yeast"

# Load the emotions dataset (training)
X_train, y_train, _, _ = load_dataset(var, 'train')

# Load the emotions dataset (testing)
X_test, y_test, _, _ = load_dataset(var, 'test')





def evaluate_pipeline(classifier, X_train, y_train, X_test, y_test):
    print(classifier)
    # Initialize BinaryRelevance with the provided classifier
    clf = BinaryRelevance(
        classifier=classifier,
        require_dense=[False, True]
    )

    # Fit the model on the training data
    clf.fit(X_train, y_train)

    # Make predictions on the test set
    predictions = clf.predict(X_test)
    f1_micro = metrics.f1_score(y_test, predictions, average='micro')

    # Ensure y_test is in dense format
    y_test_dense = y_test.toarray() if hasattr(y_test, 'toarray') else y_test

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


# Create a pipeline with StandardScaler and LogisticRegression
#log_reg_pipeline = make_pipeline(StandardScaler(with_mean=False), LogisticRegression())
# Call the evaluation function
#evaluate_pipeline(log_reg_pipeline, X_train, y_train, X_test, y_test)


#log_reg_pipeline = make_pipeline(StandardScaler(with_mean=False), SVC(probability=True))
#evaluate_pipeline(log_reg_pipeline, X_train, y_train, X_test, y_test)

decision_tree = DecisionTreeClassifier()
evaluate_pipeline(decision_tree, X_train, y_train, X_test, y_test)



from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV

# Define the hyperparameter grid
param_grid = {
    'max_depth': [None, 5, 10, 15, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Create the Decision Tree Classifier
decision_tree = DecisionTreeClassifier(random_state=42)

# Convert y_train to dense format if it's sparse
y_train_dense = y_train.toarray() if hasattr(y_train, 'toarray') else y_train

# Set up GridSearchCV
grid_search = GridSearchCV(estimator=decision_tree,
                           param_grid=param_grid,
                           scoring='f1_micro',  # You can change this to your preferred metric
                           cv=5,  # Number of cross-validation folds
                           n_jobs=-1)  # Use all available cores

# Fit GridSearchCV
grid_search.fit(X_train, y_train_dense)

# Get the best parameters and best score
best_params = grid_search.best_params_
best_score = grid_search.best_score_

print("Best Hyperparameters:", best_params)
print("Best Cross-Validation Score (F1 Micro):", best_score)

# Now evaluate the best model found
best_model = grid_search.best_estimator_
evaluate_pipeline(best_model, X_train, y_train_dense, X_test, y_test)


