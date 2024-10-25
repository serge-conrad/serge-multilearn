
from skmultilearn.dataset import available_data_sets

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
