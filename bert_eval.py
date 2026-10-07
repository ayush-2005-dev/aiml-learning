import csv
import pickle
import time

import torch
from sklearn.metrics import accuracy_score, recall_score
from transformers import AutoTokenizer, AutoModelForMaskedLM

NAME = "tohoku-nlp/bert-base-japanese-char-v3"
PARTICLES = ["は", "が", "を", "に", "で", "へ"]

tokenizer = AutoTokenizer.from_pretrained(NAME)
model = AutoModelForMaskedLM.from_pretrained(NAME)
model.eval()
particle_ids = tokenizer.convert_tokens_to_ids(PARTICLES)

with open("data/tokens_100k.pkl", "rb") as f:
    all_tokens = pickle.load(f)

# Same test examples as the logistic-regression experiments
test = []
for sid, tokens in enumerate(all_tokens):
    if sid >= 20000 or sid % 10 != 0:
        continue
    for i, (surface, pos, _) in enumerate(tokens):
        if surface in PARTICLES and pos == "助詞":
            left = "".join(t[0] for t in tokens[:i])[-60:]
            right = "".join(t[0] for t in tokens[i + 1:])[:60]
            test.append((surface, left, right))

print("test examples:", len(test), flush=True)


def predict(left, right):
    text = left + tokenizer.mask_token + right
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=256)
    mask_pos = (inputs["input_ids"][0] == tokenizer.mask_token_id).nonzero()[0].item()
    with torch.no_grad():
        logits = model(**inputs).logits[0, mask_pos]
    return PARTICLES[int(torch.argmax(logits[particle_ids]))]


truth, preds = [], []
start = time.time()
with open("data/bert_predictions.csv", "w", encoding="utf-8", newline="") as out:
    writer = csv.writer(out)
    writer.writerow(["label", "pred", "left", "right"])
    for n, (label, left, right) in enumerate(test, 1):
        pred = predict(left, right)
        truth.append(label)
        preds.append(pred)
        writer.writerow([label, pred, left, right])
        if n % 500 == 0:
            elapsed = time.time() - start
            print(f"{n} done, {elapsed:.0f}s elapsed", flush=True)

print()
print("BERT zero-shot accuracy:", round(accuracy_score(truth, preds), 3))
recalls = recall_score(truth, preds, labels=PARTICLES, average=None, zero_division=0)
for p, r in zip(PARTICLES, recalls):
    print(f"  {p}: recall {r:.3f}")