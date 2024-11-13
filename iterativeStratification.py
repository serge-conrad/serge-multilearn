
from invoke import task
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold




class CustomStratifiedKFold:
    def __init__(self, n_splits=5, shuffle=True, random_state=None):
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state
        self.mskf = MultilabelStratifiedKFold(n_splits=self.n_splits, shuffle=self.shuffle, random_state=self.random_state)

    def split(self, X, y):
        for train_index, test_index in self.mskf.split(X, y):
            X_train, y_train = X[train_index], y[train_index]

            yield train_index, test_index

    def get_n_splits(self, X, y, groups=None):
        return self.n_splits





