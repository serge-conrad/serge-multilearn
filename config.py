# config.py
PARAMS_FILE = 'best_hyperparameters.json'
RESULT_FILE = 'results.txt'

# List of datasets to evaluate
datasets = ["emotions", "yeast"]
#datasets = ["emotions"]
#datasets = ["mediamill"]
#datasets = ["delicious","emotions"]
#available dataset {'delicious', 'genbase', 'Corel5k', 'mediamill', 'yeast', 'birds', 'rcv1subset4', 'rcv1subset5', 'enron', 'rcv1subset2', 'bibtex', 'medical', 'rcv1subset1', 'tmc2007_500', 'rcv1subset3', 'scene', 'emotions'}


param_splitting = {
    'standard',
    'iterativestratification',
}
param_methods = {
    'BinaryRelevance',
    'OneVsRestClassifier'
}

# Define parameter grids for each classifier
param_grids = {
    'DecisionTreeClassifier': {
        'classifier__max_depth': [1,2,3],
    },
    'RandomForestClassifier': {
        'classifier__max_depth': [1,2,3],
    },
}
#    'MLPClassifier': {
#        'classifier__hidden_layer_sizes': [(10,), (50,), (100,)],  # Different hidden layer sizes
#        'classifier__max_iter': [100, 200, 300]  # Maximum iterations
#    }
#    'DecisionTreeClassifier': {
#        'classifier__max_depth': [1,2,3],
#    },
#    'RandomForestClassifier': {
#        'classifier__max_depth': [1,2,3],
#    },
#    'GaussianNB': {
#        'classifier__var_smoothing': [1e-9, 1e-8, 1e-7],  # Example parameter
#    },

#'classifier__hidden_layer_sizes': [(10,), (50,), (100,)],  # Different hidden layer sizes
#'classifier__activation': ['logistic', 'tanh', 'relu'],  # Activation functions
#        'classifier__alpha': [0.0001, 0.001, 0.01],  # Regularization term
#        'classifier__max_iter': [100, 200, 300]  # Maximum iterations
# onevsrest, binary relevance, decisiontree, random forest,  SVM, naivebayes, #MultilayerPerceptron NN
#voir les hyperparamerts a considérer dans [2]
# multioutputclassifier vs binaryrelance vs onevsrest ?
#MultilayerPerceptron NN
# gggccc
#'estimator__min_samples_split': [2, 5]

#paper on the stratification of multi label data
#splitting multilabel dataset: trie to keep balance of the label... utiliser la methode pour la cross validation ... implementée

#a thorough expiremental comparison of multilabel methods for classifications performances
# sur les performances des classifiers... binary relevance, decisiontree, random forest,  SVM, naivebayes,
#voir les hyperparamerts


#1 kfold cv on training
#2 best params
#3 train on full train test
#4 test on test set


