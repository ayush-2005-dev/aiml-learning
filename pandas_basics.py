import pandas as pd
import matplotlib.pyplot as plt

data = {
    "subject": ["C++", "Statistics", "MySQL", "HTML/CSS"],
    "marks": [72, 85, 90, 64],
    "hours_studied": [10, 14, 12, 6],
}

df = pd.DataFrame(data)

print(df)
print()
print(df.describe())
print()
print("Best subject:", df.loc[df["marks"].idxmax(), "subject"])

df.plot(x="hours_studied", y="marks", kind="scatter", title="Hours vs marks")
plt.savefig("hours_vs_marks.png")
plt.show()