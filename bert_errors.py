import pandas as pd

df = pd.read_csv("data/bert_predictions.csv", keep_default_na=False)
labels = ["は", "が", "を", "に", "で", "へ"]

cm = pd.crosstab(df["label"], df["pred"]).reindex(
    index=labels, columns=labels, fill_value=0
)
cm.index = [f"true {l}" for l in labels]
cm.columns = [f"pred {l}" for l in labels]
print(cm)

errors = df[df["label"] != df["pred"]]
print("\nTotal mistakes:", len(errors), "of", len(df))


def show(rows):
    for _, r in rows.iterrows():
        left = r["left"][-12:]
        right = r["right"][:12]
        print(f"{left}[{r['label']} -> {r['pred']}]{right}")


print("\n20 random mistakes (left [true -> model said] right):")
show(errors.sample(20, random_state=1))

print("\nAll へ cases the model got wrong:")
show(errors[errors["label"] == "へ"])