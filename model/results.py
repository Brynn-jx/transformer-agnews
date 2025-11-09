import pickle
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import argparse


def plot_curves_by_epoch(record_dict, save_dir="results"):
    """
    按 epoch 绘制并保存训练与验证的损失和准确率曲线（x轴范围固定为0~20）。
    """
    os.makedirs(save_dir, exist_ok=True)

    # 将数据转为 DataFrame
    train_df = pd.DataFrame(record_dict["train"])
    val_df = pd.DataFrame(record_dict["val"])

    # 按 epoch 聚合（求平均）
    train_epoch = train_df.groupby("epoch")[["loss", "acc"]].mean().reset_index()
    val_epoch = val_df.groupby("epoch")[["loss", "acc"]].mean().reset_index()

    epoch = max(train_epoch["epoch"].values)

    # ========== Loss 图 ==========
    plt.figure(figsize=(8, 6))
    plt.plot(train_epoch["epoch"], train_epoch["loss"],
             label="Train Loss", color="#1f77b4", linewidth=2)
    plt.plot(val_epoch["epoch"], val_epoch["loss"],
             label="Validation Loss", color="#ff7f0e", linewidth=2, marker='o')

    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Loss", fontsize=12)
    plt.title("Training vs Validation Loss", fontsize=14, fontweight="bold")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)

    # 固定 x 轴范围为 0~20，设置刻度间隔为 1
    plt.xlim(0, epoch)
    plt.xticks(np.arange(0, epoch+1, 1))

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "loss_curve_epoch.png"), dpi=300)
    plt.close()

    # ========== Accuracy 图 ==========
    plt.figure(figsize=(8, 6))
    plt.plot(train_epoch["epoch"], train_epoch["acc"],
             label="Train Accuracy", color="#2ca02c", linewidth=2)
    plt.plot(val_epoch["epoch"], val_epoch["acc"],
             label="Validation Accuracy", color="#d62728", linewidth=2, marker='o')

    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Accuracy", fontsize=12)
    plt.title("Training vs Validation Accuracy", fontsize=14, fontweight="bold")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)

    #  同样固定 x 轴范围
    plt.xlim(0, epoch)
    plt.xticks(np.arange(0, epoch + 1, 1))

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "accuracy_curve_epoch.png"), dpi=300)
    plt.close()

    print(f" 图像已保存到：{save_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_path", help="save path", type=str)
    config = parser.parse_args()
    with open(config.save_path + "/record_dict.pkl", "rb") as f:
        record_dict = pickle.load(f)
    plot_curves_by_epoch(record_dict, config.save_path)


