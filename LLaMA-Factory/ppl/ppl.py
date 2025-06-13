import pandas as pd
import matplotlib.pyplot as plt

# 模拟读取你提供的 DataFrame
df = pd.read_csv("llama_qwen_ppl_stats.csv")

# 提取四个子集
subsets = {
    "llama_answer": df[df["file"] == "llama_answer"],
    "llama_shuffled": df[df["file"] == "llama_shuffled"],
    "qwen_answer": df[df["file"] == "qwen_answer"],
    "qwen_shuffled": df[df["file"] == "qwen_shuffled"]
}

# 画图
for name, subset in subsets.items():
    plt.figure(figsize=(6, 4))
    subset = subset.sort_values("ckpt")
    plt.plot(subset["ckpt"], subset["ppl"], marker="o", linestyle="-")
    plt.xlabel("Checkpoint", fontsize=12)
    plt.ylabel("Perplexity", fontsize=12)
    plt.title(name.replace("_", " ").title(), fontsize=14)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(f"{name}.png")
    plt.close()

plt.figure(figsize=(8, 6))

for name, subset in subsets.items():
    subset = subset.sort_values("ckpt")
    plt.plot(subset["ckpt"], subset["ppl"], marker="o", linestyle="-", label=name.replace("_", " ").title())

plt.xlabel("Checkpoint", fontsize=14)
plt.ylabel("Perplexity", fontsize=14)
plt.title("Perplexity over Checkpoints", fontsize=16)
plt.legend(title="Model", fontsize=12)
plt.grid(True, linestyle="--", alpha=0.6)
plt.tight_layout()
plt.savefig("ppl_all_in_one.png")
plt.close()