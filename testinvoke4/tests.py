#from skmultilearn.problem_transform import BinaryRelevance
#from sklearn.tree import DecisionTreeClassifier
#from skmultilearn.dataset import load_dataset
#from sklearn.model_selection import GridSearchCV
#from sklearn.ensemble import RandomForestClassifier


from invoke import task
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
import numpy as np
from skmultilearn.dataset import available_data_sets

@task
def teststratkfold(c):
    """test MultilabelStratifiedKFold de iterative-stratification"""


    X = np.array([[1,2], [3,4], [1,2], [3,4], [1,2], [3,4], [1,2], [3,4]])
    y = np.array([[0,0], [0,0], [0,1], [0,1], [1,1], [1,1], [1,0], [1,0]])

    mskf = MultilabelStratifiedKFold(n_splits=2, shuffle=True, random_state=0)

    print("X",X)
    for train_index, test_index in mskf.split(X, y):
       print("TRAIN:", train_index, "TEST:", test_index)

       X_train, X_test = X[train_index], X[test_index]
       y_train, y_test = y[train_index], y[test_index]



from iterstrat.ml_stratifiers import RepeatedMultilabelStratifiedKFold
@task
def testrepeatstratkfold(c):
    """test RepeatedMultilabelStratifiedKFold de iterative-stratification"""
    X = np.array([[1,2], [3,4], [1,2], [3,4], [1,2], [3,4], [1,2], [3,4]])
    y = np.array([[0,0], [0,0], [0,1], [0,1], [1,1], [1,1], [1,0], [1,0]])

    rmskf = RepeatedMultilabelStratifiedKFold(n_splits=2, n_repeats=2, random_state=0)
    
    for train_index, test_index in rmskf.split(X, y):
       print("TRAIN:", train_index, "TEST:", test_index)
       X_train, X_test = X[train_index], X[test_index]
       y_train, y_test = y[train_index], y[test_index]

from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit
@task
def testshuffle(c):
    """test MultilabelStratifiedShuffleSplit de iterative-stratification"""

    X = np.array([[1,2], [3,4], [1,2], [3,4], [1,2], [3,4], [1,2], [3,4]])
    y = np.array([[0,0], [0,0], [0,1], [0,1], [1,1], [1,1], [1,0], [1,0]])

    msss = MultilabelStratifiedShuffleSplit(n_splits=9, test_size=0.3, random_state=0)

    for train_index, test_index in msss.split(X, y):
       print("TRAIN:", train_index, "TEST:", test_index)
       X_train, X_test = X[train_index], X[test_index]
       y_train, y_test = y[train_index], y[test_index]



@task
def test0(c):
    """ check available_data_sets()"""
    dataset = set([x[0] for x in available_data_sets().keys()])
    print("available dataset",dataset)

    #var='emotions'
    var='enron'


    from skmultilearn.dataset import load_dataset
    X, y, _, _ = load_dataset(var, 'train')

    print(f"Features shape (X): {X.shape}")
    print(f"Labels shape (y): {y.shape}")

    #########################################################################
    X_dense = X.toarray()  # Converts the feature matrix to a dense array
    y_dense = y.toarray()  # Converts the label matrix to a dense array

    # Printing the first few rows to inspect
    print("1 feature vectors in dense representation (X):\n", X_dense[1])
    print("1 label vectors in dense representation (y):\n", y_dense[1])
    print("1 label vectors in native sparse representation (y):\n", y[1])



    non_zero_features = X.nnz  # Number of non-zero elements in the feature matrix
    non_zero_labels = y.nnz    # Number of non-zero elements in the label matrix

    print(f"Number of non-zero feature entries: {non_zero_features}")
    print(f"Number of non-zero label entries: {non_zero_labels}")

from skmultilearn.problem_transform import BinaryRelevance
from sklearn.tree import DecisionTreeClassifier
from skmultilearn.dataset import load_dataset
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier


from invoke import task

@task
def test1(c):
    """ example BinaryRelevance( DecisionTreeClassifier()) avec GridSearchCV"""

    X,y,_,_ = load_dataset("yeast", "train")


    print("estimator = BinaryRelevance( DecisionTreeClassifier() )")
    estimator = BinaryRelevance( DecisionTreeClassifier() )
    #print(estimator.get_params())
    #{'classifier': DecisionTreeClassifier(), 'classifier__ccp_alpha': 0.0, 'classifier__class_weight': None, 'classifier__criterion': 'gini', 'classifier__max_depth': None, 'classifier__max_features': None, 'classifier__max_leaf_nodes': None, 'classifier__min_impurity_decrease': 0.0, 'classifier__min_samples_leaf': 1, 'classifier__min_samples_split': 2, 'classifier__min_weight_fraction_leaf': 0.0, 'classifier__monotonic_cst': None, 'classifier__random_state': None, 'classifier__splitter': 'best', 'require_dense': [True, True]}
    params = {'classifier__max_depth': [1, 2, 3]}
    grid = GridSearchCV(estimator, cv=3, param_grid=params)
    grid.fit(X, y)
    best_params = {k.replace("classifier__", ""):v for k, v in grid.best_params_.items()}
    print(best_params)


    print("estimator = BinaryRelevance(RandomForestClassifier())")
    estimator = BinaryRelevance(RandomForestClassifier())
    params = {'classifier__max_depth': [1, 2, 3]}
    grid = GridSearchCV(estimator, cv=3, param_grid=params)
    grid.fit(X, y)
    best_params = {k.replace("classifier__", ""):v for k, v in grid.best_params_.items()}
    print(best_params)



    print("estimator =  DecisionTreeClassifier()")
    estimator =  DecisionTreeClassifier()
    #print(estimator.get_params())
    #{'ccp_alpha': 0.0, 'class_weight': None, 'criterion': 'gini', 'max_depth': None, 'max_features': None, 'max_leaf_nodes': None, 'min_impurity_decrease': 0.0, 'min_samples_leaf': 1, 'min_samples_split': 2, 'min_weight_fraction_leaf': 0.0, 'monotonic_cst': None, 'random_state': None, 'splitter': 'best'}
    
    params = {'max_depth': [1, 2, 3]}
    grid = GridSearchCV(estimator, cv=3, param_grid=params)
    grid.fit(X, y.toarray())
    best_params = grid.best_params_
    print(best_params)




