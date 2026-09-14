import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from collections import Counter

class KNN:
    def __init__(self, k):
        self.K = k
    
    def _euc(self, p1, p2):
        return np.sqrt(np.sum((p1 - p2)**2))

    def knn_predict(self, x, y, x_new):
        num_samples, features = x.shape
        distances = []
        for _ in range(num_samples): #training loop
            dist = self._euc(x[_], x_new)
            distances.append((dist, y[_]))
        distances = sorted(distances) #distances[_, label]
        nearest = distances[:self.K] #return k nearest distances

        return Counter([label for _ , label in nearest]).most_common(1)[0][0] #returns labels in the nearest
#Synthetic Dataset
X = np.array([
    [1, 1],
    [1, 2],
    [2, 1],
    [2, 2],
    [3, 3],
    [3, 4],
    [4, 3],
    [4, 4],
    [6, 6],
    [6, 7],
    [7, 6],
    [7, 7]
])

y = np.array([
    0, 0, 0, 0,
    0, 0, 0, 0,
    1, 1, 1, 1
])
model = KNN(k=3) # init model
X_new = np.array([6.5, 6.5])
prediction = model.knn_predict(X, y, X_new)
print("=======================")
print("KNN Results:")
print(f"Test Data: {X_new}")
print(f"Predicted Class: {prediction}")

    
    
    