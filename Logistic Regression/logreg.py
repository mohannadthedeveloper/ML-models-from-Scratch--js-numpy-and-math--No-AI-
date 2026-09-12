import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split as tt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, recall_score
from sklearn.preprocessing import StandardScaler
#Predict whether someone will pass based on their study time

class LogisticRegression:
    def __init__(self, learning_rate=0.0001, epochs=250):
        self.lr = learning_rate
        self.weights = None
        self.bias = None
        self.epochs = epochs

    def _sigmoid(self, X):
        z = np.dot(X, self.weights) + self.bias
        return 1 / (1 + np.exp(-z))
    
    def fit(self, X, y):
        num_samples, num_features = X.shape
        self.bias = 0.0
        self.weights = np.zeros(num_features)
        for _ in range(self.epochs):
            y_pred = self._sigmoid(X)
            dw = (1 / num_samples) * np.dot(X.T, (y_pred - y))
            db = (1 / num_samples) * np.sum(y_pred - y)
            self.weights -= self.lr * dw
            self.bias -= self.lr * db

    def predict(self, X):
        return self._sigmoid(X)

data = pd.read_csv('Logistic Regression/logistic_regression_level_2.csv')
X = data[['study_hours', 'practice_tests']]
y = data['passed']
X_train, X_test, y_train, y_test = tt(X, y, test_size=0.2)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
model = LogisticRegression(learning_rate=0.01, epochs=1000)
model.fit(X_train, y_train)
preds = model.predict(X_test)
#evaluation
preds = (preds >= 0.5).astype(int)
plt.scatter(X_test[:, 0], preds)
plt.plot()
plt.show()

    
