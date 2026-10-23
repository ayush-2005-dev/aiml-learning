import math
import os

import matplotlib.pyplot as plt
plt.rcParams["font.family"] = ["Yu Gothic", "Meiryo", "MS Gothic", "DejaVu Sans"]

N_TEST = 4327


def ci95(acc):
    # 95% range from test-set sampling noise only
    return 1.96 * math.sqrt(acc * (1 - acc) / N_TEST)


# Learning curve: logistic regression, window 3 + next verb
sentences = [1000, 2500, 5000, 10000, 18000, 35000, 60000, 98000]
accuracy = [0.677, 0.714, 0.743, 0.780, 0.805, 0.824, 0.826, 0.839]

# Model comparison on the same 4,327 test examples
models = [
    "Always guess は\n(baseline)",
    "Logistic regression\nwindow 2 (20k sent.)",
    "Logistic regression\nwindow 3 + verb (98k)",
    "Japanese BERT\n(no training on data)",
]
scores = [0.369, 0.789, 0.839, 0.926]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

# Left: learning curve
ax1.errorbar(sentences, accuracy, yerr=[ci95(a) for a in accuracy],
             marker="o", capsize=3, color="tab:blue")
ax1.set_xscale("log")
ax1.set_xlabel("Training sentences (log scale)")
ax1.set_ylabel("Test accuracy")
ax1.set_title("More data helps, with shrinking gains")
ax1.set_ylim(0.6, 0.95)
ax1.grid(alpha=0.3)

# Right: model comparison
bars = ax2.bar(range(len(models)), scores,
               yerr=[ci95(s) for s in scores], capsize=4,
               color=["gray", "tab:blue", "tab:blue", "tab:orange"])
ax2.set_xticks(range(len(models)))
ax2.set_xticklabels(models, fontsize=8)
ax2.set_ylabel("Test accuracy")
ax2.set_title("Particle prediction: model comparison")
ax2.set_ylim(0, 1.05)
for bar, s in zip(bars, scores):
    ax2.text(bar.get_x() + bar.get_width() / 2, s + 0.03, f"{s:.3f}",
             ha="center", fontsize=9)

fig.suptitle("Predicting Japanese particles (は が を に で へ) from Tatoeba sentences",
             fontsize=12)
fig.text(0.5, 0.01,
         "Test set: 4,327 particles. Error bars show 95% sampling uncertainty only (one train/test split).",
         ha="center", fontsize=8)
fig.tight_layout(rect=[0, 0.03, 1, 0.95])

os.makedirs("figures", exist_ok=True)
plt.savefig("figures/particle_results.png", dpi=150)
plt.show()