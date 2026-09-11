import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split

class LinearRegression:
    def __init__(self, learning_rate=0.00001, epochs=200):
        self.lr = learning_rate
        self.epochs = epochs
        self.weights = None
        self.bias = None
    
    def fit(self, X, y):
        n, num_features = X.shape
        self.weights = np.zeros(num_features)
        for _ in range(self.epochs):
            y_pred = np.dot(self.weights, X) + self.bias
            dw = (1 / n) * np.dot(X.T, y_pred - y)
            db = (1 / n) * np.sum(y_pred - y)
            self.bias -= self.lr * db
            self.weights -= self.lr * dw

    def predict(self, X):
        return np.dot(self.weights, X) + self.bias

model = LinearRegression(learning_rate=0.0001, epochs=200)
data = pd.read_csv(r"C:\Users\HP\OneDrive\Desktop\ML models from Scratch (js numpy and math, No AI)\Linear Regression\ecommerce_sales_data.csv")
X = data['TotalAmount']
y = data['UnitPrice']
X_train, X_test, y_train, y_test = train_test_split()