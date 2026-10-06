import bz2
import os
import pickle

import pandas as pd
from janome.tokenizer import Tokenizer
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, recall_score

PATH = "data/jpn_sentences.tsv.bz2"
CACHE = "data/tokens.pkl"
PARTICLES = ["は", "が", "を", "に", "で", "へ"]
N_SENTENCES = 20000


def load_tokens():
    # Tokenizing is slow, so the result is saved and reused on later runs
    if os.path.exists(CACHE):
        with open(CACHE, "rb") as f:
            return pickle.load(f)
    sentences = []
    with bz2.open(PATH, "rt", encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) >= 3:
                sentences.append(parts[2])
            if len(sentences) >= N_SENTENCES:
                break
    tokenizer = Tokenizer()
    all_tokens = [
        [(t.surface, t.part_of_speech.split(",")[0]) for t in tokenizer.tokenize(s)]
        for s in sentences
    ]
    with open(CACHE, "wb") as f:
        pickle.dump(all_tokens, f)
    return all_tokens


def build(all_tokens, before, after):
    rows = []
    for sid, tokens in enumerate(all_tokens):
        for i, (surface, pos) in enumerate(tokens):
            if surface in PARTICLES and pos == "助詞":
                feats = {}
                for k in range(1, before + 1):
                    w, p = tokens[i - k] if i - k >= 0 else ("<none>", "<none>")
                    feats[f"w-{k}"] = w
                    feats[f"p-{k}"] = p
                for k in range(1, after + 1):
                    w, p = tokens[i + k] if i + k < len(tokens) else ("<none>", "<none>")
                    feats[f"w+{k}"] = w
                    feats[f"p+{k}"] = p
                split = "test" if sid % 10 == 0 else "train"
                rows.append((split, surface, feats))
    return rows


def run(all_tokens, before, after):
    rows = build(all_tokens, before, after)
    train = [(label, f) for s, label, f in rows if s == "train"]
    test = [(label, f) for s, label, f in rows if s == "test"]
    vec = DictVectorizer()
    X_train = vec.fit_transform([f for _, f in train])
    X_test = vec.transform([f for _, f in test])
    y_train = [label for label, _ in train]
    y_test = [label for label, _ in test]
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    acc = accuracy_score(y_test, pred)
    recalls = recall_score(y_test, pred, labels=PARTICLES, average=None, zero_division=0)
    return acc, recalls


all_tokens = load_tokens()
results = {}
for window in [1, 2, 3, 5]:
    acc, recalls = run(all_tokens, window, window)
    results[f"{window} each side"] = [acc] + list(recalls)
    print("finished window", window)

table = pd.DataFrame(results, index=["accuracy"] + PARTICLES).T.round(3)
print()
print(table)