import pandas as pd
import matplotlib.pyplot as plt

# 文件路径
base_path = "/data2/yihewang/psychdepth_v2/pipeline/results/Llama-3.1-8B-Instruct_sft/"
train_path = base_path + "gsm8k_answer-train_set.csv"
test_path = base_path + "gsm8k_answer-test_set.csv"

# 加载数据
train_df = pd.read_csv(train_path, dtype={'prediction': str, 'ground_truth': str})
test_df = pd.read_csv(test_path, dtype={'prediction': str, 'ground_truth': str})

# 按 ckpt 分组，计算准确率
train_acc = train_df.groupby("ckpt").apply(lambda x: (x["prediction"] == x["ground_truth"]).mean()).reset_index(name="accuracy")
test_acc = test_df.groupby("ckpt").apply(lambda x: (x["prediction"] == x["ground_truth"]).mean()).reset_index(name="accuracy")

# 排序
train_acc = train_acc.sort_values("ckpt")
test_acc = test_acc.sort_values("ckpt")

# 绘图
plt.figure(figsize=(8, 5))
plt.plot(train_acc["ckpt"], train_acc["accuracy"], marker='o', label="Train Accuracy")
plt.plot(test_acc["ckpt"], test_acc["accuracy"], marker='s', label="Test Accuracy")

plt.xlabel("Checkpoint")
plt.ylabel("Accuracy (prediction == ground_truth)")
plt.title("Accuracy over Checkpoints (Train vs Test)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("answer.png")
plt.show()
exit()

# 假设你已经有一个 DataFrame df
# df 中包含列：ckpt, prediction, shuffled_ground_truth, correct_ground_truth
df = pd.read_csv("/data2/yihewang/psychdepth_v2/pipeline/results/Llama-3.1-8B-Instruct_sft/gsm8k_shuffled-train_set.csv", dtype={'prediction': str, 'shuffled_ground_truth': str, 'correct_ground_truth': str })
# 分组计算准确率
grouped = df.groupby("ckpt").apply(
    lambda x: pd.Series({
        "shuffled_acc": (x["prediction"] == x["shuffled_ground_truth"]).mean(),
        "correct_acc": (x["prediction"] == x["correct_ground_truth"]).mean()
    })
).reset_index()

# 画图
plt.figure(figsize=(8, 5))
plt.plot(grouped["ckpt"], grouped["shuffled_acc"], marker='o', label="Shuffled Accuracy")
plt.plot(grouped["ckpt"], grouped["correct_acc"], marker='s', label="Correct Accuracy")

plt.xlabel("Checkpoint")
plt.ylabel("Accuracy")
plt.title("Prediction Accuracy vs. Checkpoint")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("huh.png")
plt.show()
