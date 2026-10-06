import pickle

from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

PARTICLES = ["は", "が", "を", "に", "で", "へ"]

with open("data/tokens_base.pkl", "rb") as f:
    all_tokens = pickle.load(f)


def features(tokens, i, window=3):
    feats = {}
    for k in range(1, window + 1):
        w, p, _ = tokens[i - k] if i - k >= 0 else ("<none>", "<none>", "")
        feats[f"w-{k}"] = w
        feats[f"p-{k}"] = p
        w, p, _ = tokens[i + k] if i + k < len(tokens) else ("<none>", "<none>", "")
        feats[f"w+{k}"] = w
        feats[f"p+{k}"] = p
    verb = "<none>"
    for _, p, base in tokens[i + 1:]:
        if p == "動詞":
            verb = base
            break
    feats["next_verb"] = verb
    return feats


examples = []
for sid, tokens in enumerate(all_tokens):
    for i, (surface, pos, _) in enumerate(tokens):
        if surface in PARTICLES and pos == "助詞":
            examples.append((sid, surface, features(tokens, i)))

test = [(label, f) for sid, label, f in examples if sid % 10 == 0]
train_sids = [sid for sid in range(len(all_tokens)) if sid % 10 != 0]

print("n_sentences  n_examples  accuracy")
for n_sent in [1000, 2500, 5000, 10000, 18000]:
    keep = set(train_sids[:n_sent])
    train = [(label, f) for sid, label, f in examples if sid in keep]
    vec = DictVectorizer()
    X_train = vec.fit_transform([f for _, f in train])
    X_test = vec.transform([f for _, f in test])
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, [label for label, _ in train])
    pred = model.predict(X_test)
    acc = accuracy_score([label for label, _ in test], pred)
    print(f"{n_sent:>11}  {len(train):>10}  {acc:.3f}")