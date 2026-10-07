import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM

NAME = "tohoku-nlp/bert-base-japanese-char-v3"
PARTICLES = ["は", "が", "を", "に", "で", "へ"]

tokenizer = AutoTokenizer.from_pretrained(NAME)
model = AutoModelForMaskedLM.from_pretrained(NAME)
model.eval()

particle_ids = tokenizer.convert_tokens_to_ids(PARTICLES)
print("Particle token ids:", particle_ids)
print("Unknown-token id:", tokenizer.unk_token_id, "(none of the above should match)")


def score(left, right):
    text = left + tokenizer.mask_token + right
    inputs = tokenizer(text, return_tensors="pt")
    mask_pos = (inputs["input_ids"][0] == tokenizer.mask_token_id).nonzero()[0].item()
    with torch.no_grad():
        logits = model(**inputs).logits[0, mask_pos]
    probs = torch.softmax(logits[particle_ids], dim=-1)   # only the six candidates
    return dict(zip(PARTICLES, probs.tolist()))


examples = [
    ("私", "学校に行きます", "は"),
    ("きみにちょっとしたもの", "もってきたよ。", "を"),
    ("一目", "見て取った", "で"),
    ("することに", "した", "に"),
]

for left, right, answer in examples:
    p = score(left, right)
    best = max(p, key=p.get)
    shown = "  ".join(f"{k}:{v:.2f}" for k, v in p.items())
    print(f"{left}[?]{right}   true={answer}  model={best}")
    print("   ", shown)