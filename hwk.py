from sklearn.datasets import load_iris
import seaborn as sb
import numpy as np
from scipy import sparse

data = sb.load_dataset("iris") #load iris data
print(data.shape)
print(type(data))
print(data.head(3))

print(data.keys())
print(data.shape) 
print(data.info())
eye = np.eye(4)
print("NumPy array:\n", eye)
sparse_matrix = sparse.csr_matrix(eye)
print("\nSciPy sparse CSR matrix:\n", sparse_matrix)
print(data.describe())
print(data['species'].value_counts())
print(data.head())
x = data.iloc[:, [1, 2, 3, 4]].values
print(x) 