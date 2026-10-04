import numpy as np
from sklearn.linear_model import LinearRegression

hours = np.array([[10], [14], [12], [6]])   # inputs must be 2D
marks = np.array([72, 85, 90, 64])          # what we want to predict

model = LinearRegression()
model.fit(hours, marks)

print("Slope:", model.coef_[0])
print("Intercept:", model.intercept_)
print("R^2 on training data:", model.score(hours, marks))
print("Predicted marks for 8 hours:", model.predict([[8]])[0])