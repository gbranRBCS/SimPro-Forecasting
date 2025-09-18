import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

train_path = Path(__file__).parent.parent / "train.json"
with open(train_path, "r") as f:
    data = json.load(f)["data"]

margins = [row["netMarginPct"] for row in data if "netMarginPct" in row and row["netMarginPct"] is not None]

if not margins:
    print("No netMarginPct values found in training data.")
    exit(1)

margins = np.array(margins)

# plot histogram
plt.figure(figsize=(8, 5))
plt.hist(margins, bins=20, color='skyblue', edgecolor='black')
plt.xlabel("Net Margin %")
plt.ylabel("Job Count")
plt.title("Distribution of netMarginPct in Training Data")


plt.axvline(0.05, color='orange', linestyle='--', label='5% threshold')
plt.axvline(0.20, color='red', linestyle='--', label='20% threshold')


q25, q50, q75 = np.percentile(margins, [25, 50, 75])
plt.axvline(q25, color='green', linestyle=':', label='25th percentile')
plt.axvline(q50, color='blue', linestyle=':', label='50th percentile (median)')
plt.axvline(q75, color='purple', linestyle=':', label='75th percentile')

plt.legend()
plt.tight_layout()
plt.show()

below_5 = np.sum(margins < 0.05)
between_5_20 = np.sum((margins >= 0.05) & (margins < 0.20))
above_20 = np.sum(margins >= 0.20)
total = len(margins)

print(f"Total jobs analyzed: {total}")
print(f"Jobs with netMarginPct < 5%: {below_5} ({below_5/total:.1%})")
print(f"Jobs with netMarginPct 5%–20%: {between_5_20} ({between_5_20/total:.1%})")
print(f"Jobs with netMarginPct >= 20%: {above_20} ({above_20/total:.1%})")
print()
print(f"25th percentile: {q25:.3f}")
print(f"50th percentile (median): {q50:.3f}")
print(f"75th percentile: {q75:.3f}")


min_class = min(below_5, between_5_20, above_20)
if min_class / total < 0.15:
    print("\nWarning: Class distribution is imbalanced.")
    print("Consider using quartiles as cut-offs for more balanced classes:")
    print(f"Suggested cut-offs: <{q25:.2f}, {q25:.2f}–{q75:.2f}, >{q75:.2f}")
else:
    print("\nClass distribution appears reasonably balanced.")