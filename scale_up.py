import bz2
import os
import pickle

from janome.tokenizer import Tokenizer
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

PATH = "data/jpn_sentences.tsv.bz2"
CACHE = "data/tokens_100k.pkl"
PARTICLES = ["は", "が", "を", "に", "で", "へ"]
N_SENTENCES = 100000


def load_tokens():
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
    all_tokens = []
    for n, s in enumerate(sentences):
        all_tokens.append(
            [(t.surface, t.part_of_speech.split(",")[0], t.base_form)
             for t in tokenizer.tokenize(s)]
        )
        if n % 10000 == 0:
            print("tokenized", n)
    with open(CACHE, "wb") as f:
        pickle.dump(all_tokens, f)
    return all_tokens


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


all_tokens = load_tokens()

examples = []
for sid, tokens in enumerate(all_tokens):
    for i, (surface, pos, _) in enumerate(tokens):
        if surface in PARTICLES and pos == "助詞":
            examples.append((sid, surface, features(tokens, i)))

# Same test sentences as before: the first 20,000 sentences, every 10th one
test = [(label, f) for sid, label, f in examples if sid < 20000 and sid % 10 == 0]

# Training pool: the earlier 18,000 first, then all newer sentences in order
train_sids = [sid for sid in range(20000) if sid % 10 != 0]
train_sids += list(range(20000, len(all_tokens)))

print("test examples:", len(test))
print("n_sentences  n_examples  accuracy")
for n_sent in [18000, 35000, 60000, len(train_sids)]:
    keep = set(train_sids[:n_sent])
    train = [(label, f) for sid, label, f in examples if sid in keep]
    vec = DictVectorizer()
    X_train = vec.fit_transform([f for _, f in train])
    X_test = vec.transform([f for _, f in test])
    model = LogisticRegression(max_iter=500)
    model.fit(X_train, [label for label, _ in train])
    pred = model.predict(X_test)
    acc = accuracy_score([label for label, _ in test], pred)
    print(f"{n_sent:>11}  {len(train):>10}  {acc:.3f}", flush=True)