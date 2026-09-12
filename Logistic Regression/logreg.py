import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split as tt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, recall_score
from sklearn.preprocessing import StandardScaler
#Predict whether someone will pass based on their study time

#load data
data = pd.read_csv("Logistic Regression/logistic_regression_level_2.csv")
X = data[['study_hours','practice_tests','attendance_pct','sleep_hours']].to_numpy()
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
y = data['passed'].to_numpy()
X_train, X_test, y_train, y_test = tt(X_scaled, y, test_size=0.2, random_state=42)
#model
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

b0 = 0
b = np.zeros(4)
alpha = 0.01
for i in range(1000): #Gradient Descent
    n = len(X_train)
    y_p = sigmoid(b0 + X_train @ b)
    db0 = (1/n)*np.sum(y_p - y_train)
    db = (1/n)*(X_train.T  @ (y_p - y_train))
    b0 -= alpha*db0
    b -= alpha*db
    error = -(1/len(y_train)) * np.sum(
    y_test * np.log(y_p)
    + (1-y_test) * np.log(1-y_p)
)
y_p = sigmoid(b0 + X_test @ b)
pred = (y_p >= 0.5).astype(int)
#BCE 
error = -(1/len(y_test)) * np.sum(
    y_test * np.log(y_p)
    + (1-y_test) * np.log(1-y_p)
)
recall = recall_score(y_test, pred)
fig, ax = plt.subplots(figsize=(6,6))
plt.scatter(X_test, y_test, color='blue')
plt.plot(y_p, color='orange', label='Model')
plt.legend()
plt.tight_layout()
plt.show()