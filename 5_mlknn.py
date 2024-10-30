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

#var="emotions"
var="enron"
#var="yeast"
print("converts the multi-label problem into multiple independent binary classification problems (one for each label).")
print("classifier=SVC(): The base classifier for each of these independent binary classification problems is an SVM (Support Vector Machine), implemented by SVC() from the scikit-learn library.")
print("require_dense=[False, True], premier argument X, deuxième y")




# Load the emotions dataset (training)
X_train, y_train, _, _ = load_dataset(var, 'train')

# Load the emotions dataset (testing)
X_test, y_test, _, _ = load_dataset(var, 'test')


#####################MLkNN##############################################
from skmultilearn.adapt import MLkNN
classifier = MLkNN(k=3)
print(classifier.get_params())
prediction = classifier.fit(X_train, y_train).predict(X_test)
#print(metrics.hamming_loss(y_test, prediction))

print("loss",metrics.hamming_loss(y_test, prediction))
print("accuracy",metrics.accuracy_score(y_test, prediction))

# Calculate balanced accuracy per label
balanced_accuracies = []
for i in range(y_test.shape[1]):
    balanced_accuracy = metrics.balanced_accuracy_score(y_test[:, i].toarray().ravel(), prediction[:, i].toarray().ravel())
    balanced_accuracies.append(balanced_accuracy)

# Calculate average balanced accuracy across all labels
average_balanced_accuracy = np.mean(balanced_accuracies)
print("Average Balanced Accuracy:", average_balanced_accuracy)



precision = metrics.precision_score(y_test, prediction, average='micro')
recall = metrics.recall_score(y_test, prediction, average='micro')
f1_score = metrics.f1_score(y_test, prediction, average='micro')

print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1_score)


from sklearn.metrics import make_scorer, f1_score
######################################################
# Define a scorer for multi-label classification
f1_scorer = make_scorer(f1_score, average='micro')

# Parameter grid for k and s
param_grid = {
    'k': [1, 3, 5, 7, 10],
    's': [0.5, 1.0, 1.5, 2.0]
}

# Initialize MLkNN classifier without setting parameters
classifier = MLkNN()

# Perform grid search
grid_search = GridSearchCV(classifier, param_grid, scoring=f1_scorer, cv=5)
grid_search.fit(X_train, y_train)

# Extract the best parameters
best_params = grid_search.best_params_
print("Best parameters found: ", best_params)

# Fit classifier with best hyperparameters
classifier = MLkNN(k=best_params['k'], s=best_params['s'])
#classifier = MLkNN(k=best_params['k'])
prediction = classifier.fit(X_train, y_train).predict(X_test)

print("loss",metrics.hamming_loss(y_test, prediction))
print("accuracy",metrics.accuracy_score(y_test, prediction))

# Calculate Balanced Accuracy

precision = metrics.precision_score(y_test, prediction, average='micro')
recall = metrics.recall_score(y_test, prediction, average='micro')
f1_score = metrics.f1_score(y_test, prediction, average='micro')

## Calculate balanced accuracy per label
balanced_accuracies = []
for i in range(y_test.shape[1]):
    balanced_accuracy = metrics.balanced_accuracy_score(y_test[:, i].toarray().ravel(), prediction[:, i].toarray().ravel())
    balanced_accuracies.append(balanced_accuracy)

# Calculate average balanced accuracy across all labels
average_balanced_accuracy = np.mean(balanced_accuracies)
print("Average Balanced Accuracy:", average_balanced_accuracy)


print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1_score)

