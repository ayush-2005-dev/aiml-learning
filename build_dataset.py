import bz2
import csv
from collections import Counter
from janome.tokenizer import Tokenizer

PATH = "data/jpn_sentences.tsv.bz2"
OUT = "data/particle_examples.csv"
PARTICLES = {"は", "が", "を", "に", "で", "へ"}
N_SENTENCES = 20000

sentences = []
with bz2.open(PATH, "rt", encoding="utf-8") as f:
    for line in f:
        parts = line.rstrip("\n").split("\t")
        if len(parts) >= 3:
            sentences.append(parts[2])
        if len(sentences) >= N_SENTENCES:
            break

tokenizer = Tokenizer()

def get(tokens, i):
    if 0 <= i < len(tokens):
        return tokens[i]
    return ("<none>", "<none>")

rows = []
for sid, sentence in enumerate(sentences):
    tokens = [(t.surface, t.part_of_speech.split(",")[0])
              for t in tokenizer.tokenize(sentence)]
    for i, (surface, pos) in enumerate(tokens):
        if surface in PARTICLES and pos == "助詞":
            p2, _ = get(tokens, i - 2)
            p1, p1pos = get(tokens, i - 1)
            n1, n1pos = get(tokens, i + 1)
            n2, _ = get(tokens, i + 2)
            split = "test" if sid % 10 == 0 else "train"
            rows.append([sid, split, surface, p2, p1, p1pos, n1, n1pos, n2])

with open(OUT, "w", encoding="utf-8", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["sid", "split", "label", "prev2", "prev1", "prev1_pos",
                     "next1", "next1_pos", "next2"])
    writer.writerows(rows)

train = [r for r in rows if r[1] == "train"]
test = [r for r in rows if r[1] == "test"]
print("Examples:", len(rows), "| train:", len(train), "| test:", len(test))

train_counts = Counter(r[2] for r in train)
print("Train label counts:", train_counts.most_common())

majority = train_counts.most_common(1)[0][0]
baseline = sum(1 for r in test if r[2] == majority) / len(test)
print(f"Baseline (always '{majority}'): {baseline:.3f}")