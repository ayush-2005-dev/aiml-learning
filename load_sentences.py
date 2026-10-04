import bz2
from collections import Counter
from janome.tokenizer import Tokenizer

PATH = "data/jpn_sentences.tsv.bz2"
PARTICLES = {"は", "が", "を", "に", "で", "へ"}

sentences = []
with bz2.open(PATH, "rt", encoding="utf-8") as f:
    for line in f:
        parts = line.rstrip("\n").split("\t")
        if len(parts) >= 3:               # id, language, text
            sentences.append(parts[2])

print("Total sentences:", len(sentences))
print("Sample:", sentences[:5])

tokenizer = Tokenizer()
counts = Counter()
for sentence in sentences[:2000]:
    for token in tokenizer.tokenize(sentence):
        if token.surface in PARTICLES and token.part_of_speech.startswith("助詞"):
            counts[token.surface] += 1

print("Particle counts in first 2000 sentences:")
print(counts.most_common())