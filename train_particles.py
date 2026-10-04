import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

df = pd.read_csv("data/particle_examples.csv", keep_default_na=False)
features = ["prev2", "prev1", "prev1_pos", "next1", "next1_pos", "next2"]

train = df[df["split"] == "train"]
test = df[df["split"] == "test"]

def to_dicts(frame):
    return frame[features].to_dict(orient="records")

vec = DictVectorizer()                      # one-hot encodes the word features
X_train = vec.fit_transform(to_dicts(train))
X_test = vec.transform(to_dicts(test))

model = LogisticRegression(max_iter=1000)
model.fit(X_train, train["label"])

pred = model.predict(X_test)
print("Features:", X_train.shape[1])
print("Test accuracy:", round(accuracy_score(test["label"], pred), 3))
print(classification_report(test["label"], pred, zero_division=0))