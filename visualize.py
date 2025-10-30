import json, csv, numpy as np, matplotlib.pyplot as plt
import seaborn as sns

# Load the main metrics JSON
with open("outputs/metrics.json") as f:
    metrics = json.load(f)

# 1. Accuracy vs Latency plot
plt.figure(figsize=(6,4))
pref = np.array([(x["ms_per_tok"]*1000, x["acc_pct"]) for x in metrics["prefix"]])
lora = np.array([(x["ms_per_tok"]*1000, x["acc_pct"]) for x in metrics["lora"]])
comp = np.array([[metrics["compiled_lora"]["ms_per_tok"]*1000,
                  metrics["compiled_lora"]["acc_pct"]]])

plt.plot(pref[:,0], pref[:,1], 'o-', label="Prefix-only", lw=2)
plt.plot(lora[:,0], lora[:,1], 's-', label="LoRA-only", lw=2)
plt.plot(comp[:,0], comp[:,1], '^-', label="Compiled LoRA", lw=2)
plt.xlabel("Latency (μs/token)")
plt.ylabel("Accuracy (%)")
plt.title("Accuracy vs Latency (SST-2, GPT-2-small)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("outputs/acc_vs_latency.png", dpi=300)

# 2. Heatmap of layerwise residuals
vals = np.loadtxt("outputs/heatmap.csv", delimiter=",")
layers = np.arange(1, len(vals)+1)
plt.figure(figsize=(6,3))
sns.heatmap(vals.reshape(1,-1), cmap="viridis", cbar=True,
            xticklabels=layers, yticklabels=["Residual"], annot=True, fmt=".2f")
plt.xlabel("Transformer Layer")
plt.title("Layerwise Prefix–LoRA Equivalence Residual (ℒ_eq)")
plt.tight_layout()
plt.savefig("outputs/equiv_heatmap.png", dpi=300)

# 3. Bar chart summary
plt.figure(figsize=(6,3.5))
names = [f"m={x['m']}" for x in metrics["prefix"]] \
      + [f"r={x['r']}" for x in metrics["lora"]] + ["compiled"]
accs = [x["acc_pct"] for x in metrics["prefix"]] \
     + [x["acc_pct"] for x in metrics["lora"]] \
     + [metrics["compiled_lora"]["acc_pct"]]
plt.bar(names, accs, color=["C0"]*3+["C1"]*3+["C2"])
plt.ylabel("Accuracy (%)")
plt.title("Prefix vs LoRA vs Compiled Performance")
plt.tight_layout()
plt.savefig("outputs/accuracy_summary.png", dpi=300)

print("✅ Saved plots to outputs/:")
print(" - acc_vs_latency.png")
print(" - equiv_heatmap.png")
print(" - accuracy_summary.png")