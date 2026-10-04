from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

iris = load_iris()
X, y = iris.data, iris.target   # 150 flowers, 4 measurements each, 3 species

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

predictions = model.predict(X_test)

print("Training samples:", len(X_train))
print("Test samples:", len(X_test))
print("Accuracy on test data:", accuracy_score(y_test, predictions))
print("First 5 predicted:", predictions[:5])
print("First 5 actual:   ", y_test[:5])