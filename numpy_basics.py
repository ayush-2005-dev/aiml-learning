import numpy as np

scores = np.array([72, 85, 90, 64])
print("Mean:", scores.mean())
print("Std dev:", scores.std())
print("Scores above 70:", scores[scores > 70])

# A dot product, the core operation inside neural networks
weights = np.array([0.5, 0.3, 0.2, 0.0])
print("Weighted sum:", np.dot(scores, weights))