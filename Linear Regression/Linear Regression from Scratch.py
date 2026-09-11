import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

class LinearRegression:
    def __init__(self, learning_rate=0.01, epochs=200):
        self.lr = learning_rate
        self.epochs = epochs
        self.weights = None
        self.bias = None
    
    def fit(self, X, y):
        num_samples, num_features = X.shape
        self.weights = np.zeros(num_features)
        self.bias = 0.0
        for _ in range(self.epochs):
            y_pred = np.dot(X, self.weights) + self.bias
            dw = (1 / num_samples) * np.dot(X.T, y_pred - y)
            db = (1 / num_samples) * np.sum(y_pred - y)
            self.bias -= self.lr * db
            self.weights -= self.lr * dw

    def predict(self, X):
        return np.dot(X, self.weights) + self.bias

model = LinearRegression(learning_rate=0.01, epochs=500)
scaler  = StandardScaler()
data = pd.read_csv(r"C:\Users\HP\OneDrive\Desktop\ML models from Scratch (js numpy and math, No AI)\Linear Regression\ecommerce_sales_data.csv")
X = data[['TotalAmount', 'Quantity']].to_numpy()
X_scaled = scaler.fit_transform(X)
y = data['UnitPrice'].to_numpy()
model.fit(X_scaled, y)
preds = model.predict(X_scaled)


plt.figure()
sort_idx = np.argsort(X_scaled[:, 0])
plt.scatter(X_scaled[:, 0], y, color='blue', label='Actual Data') 
plt.plot(X_scaled[sort_idx], preds[sort_idx], color='red', label='Model Line') 
plt.xlabel("Feature 1")
plt.ylabel("Target (y)")
plt.legend()
plt.show()

