# data_preparation.py

from invoke import task
from skmultilearn.dataset import load_dataset


@task
def prepare_data(c, var):
    """Load and prepare the dataset."""
    print(f"Preparing data for {var}...")

    # Load the dataset (training)
    X_train, y_train, _, _ = load_dataset(var, 'train')

    # Load the dataset (testing)
    X_test, y_test, _, _ = load_dataset(var, 'test')

    # Return the datasets for use in evaluation
    return X_train, y_train, X_test, y_test

