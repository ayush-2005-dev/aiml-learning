import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix

df = pd.read_csv("data/particle_examples.csv", keep_default_na=False)
features = ["prev2", "prev1", "prev1_pos", "next1", "next1_pos", "next2"]
train = df[df["split"] == "train"]
test = df[df["split"] == "test"]

vec = DictVectorizer()
X_train = vec.fit_transform(train[features].to_dict(orient="records"))
X_test = vec.transform(test[features].to_dict(orient="records"))

model = LogisticRegression(max_iter=1000)
model.fit(X_train, train["label"])
pred = model.predict(X_test)

labels = ["は", "が", "を", "に", "で", "へ"]
cm = confusion_matrix(test["label"], pred, labels=labels)
print(pd.DataFrame(cm,
                   index=[f"true {l}" for l in labels],
                   columns=[f"pred {l}" for l in labels]))

mask = (pred != test["label"].to_numpy())
errors = test[mask].copy()
errors["pred"] = pred[mask]

print("\n15 random mistakes (context [true -> model said] context):")
for _, r in errors.sample(15, random_state=1).iterrows():
    clean = lambda w: "" if w == "<none>" else w
    left = clean(r["prev2"]) + clean(r["prev1"])
    right = clean(r["next1"]) + clean(r["next2"])
    print(f"{left}[{r['label']} -> {r['pred']}]{right}")