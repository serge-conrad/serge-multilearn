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
#var="enron"
print("converts the multi-label problem into multiple independent binary classification problems (one for each label).")
print("classifier=SVC(): The base classifier for each of these independent binary classification problems is an SVM (Support Vector Machine), implemented by SVC() from the scikit-learn library.")
print("require_dense=[False, True], premier argument X, deuxième y")



var="yeast"

# Load the emotions dataset (training)
X_train, y_train, _, _ = load_dataset(var, 'train')

# Load the emotions dataset (testing)
X_test, y_test, _, _ = load_dataset(var, 'test')


#####################MLkNN##############################################
from skmultilearn.adapt import MLkNN
classifier = MLkNN(k=3)
prediction = classifier.fit(X_train, y_train).predict(X_test)
#print(metrics.hamming_loss(y_test, prediction))

print("loss",metrics.hamming_loss(y_test, prediction))
print("accuracy",metrics.accuracy_score(y_test, prediction))
f1_micro = metrics.f1_score(y_test, prediction, average='micro')
print("F1 Micro:", f1_micro)


