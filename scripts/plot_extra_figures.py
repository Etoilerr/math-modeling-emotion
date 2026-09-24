# -*- coding: utf-8 -*-
"""
补充 3 张图满足三类图覆盖规范：
1) raw_q2_modality_valid_length.png  三模态有效长度分布（Q2 缺失鲁棒性数据先验）
2) raw_q3_label_distribution.png     回归标签按极性分布（Q3 可解释性数据基础）
3) process_q3_training_curve.png      Q3 A-EMN 训练曲线
"""
import os, sys, pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paths import ALIGNED_50, FIGURES_DIR, Q3_DIR

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False
SPLITS = ["train", "valid", "test"]
MODS = ["text", "audio", "vision"]
CMAP = {"train": "#4c9be8", "valid": "#f2a93b", "test": "#5cb85c"}


def save(fig, name):
    p = os.path.join(FIGURES_DIR, name)
    fig.savefig(p, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("saved", p)


# ---------- 1. 三模态有效长度分布 ----------
def fig_valid_length(data):
    rows = []
    for sp in SPLITS:
        d = data[sp]
        for m in MODS:
            x = np.abs(d[m]).sum(-1) > 0          # (N,T) bool
            valid_len = x.sum(-1) / 50.0          # 有效长度比例
            for v in valid_len:
                rows.append({"split": sp, "modality": m, "valid_ratio": float(v)})
    df = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(10, 6))
    positions, colors, labels = [], [], []
    pos = 0
    xtick_pos, xtick_lab = [], []
    width = 0.25
    for mi, m in enumerate(MODS):
        for si, sp in enumerate(SPLITS):
            vals = df[(df.modality == m) & (df["split"] == sp)]["valid_ratio"].values
            bp = ax.boxplot([vals], positions=[pos + (si - 1) * width], widths=width * 0.9,
                            patch_artist=True, manage_ticks=False, showfliers=False)
            for b in bp["boxes"]:
                b.set_facecolor(CMAP[sp]); b.set_alpha(.7)
            med = np.median(vals); mn = vals.mean()
            ax.text(pos + (si - 1) * width, 1.02, f"μ={mn:.2f}", ha="center",
                    va="bottom", fontsize=7, color=CMAP[sp])
        xtick_pos.append(pos); xtick_lab.append(m)
        pos += 1.2
    ax.set_xticks(xtick_pos); ax.set_xticklabels(MODS, fontsize=12)
    ax.set_ylabel("有效时间步比例（非零 / 50）")
    ax.set_ylim(0, 1.15)
    ax.set_title("三模态有效长度分布（train/valid/test）—— Q2 缺失鲁棒性的数据先验")
    handles = [plt.Rectangle((0, 0), 1, 1, color=CMAP[s], alpha=.7) for s in SPLITS]
    ax.legend(handles, SPLITS, title="划分", loc="lower right")
    ax.grid(alpha=.3, axis="y")
    fig.tight_layout()
    save(fig, "raw_q2_modality_valid_length.png")


# ---------- 2. 回归标签按极性分布 ----------
def fig_label_dist(data):
    d = data["train"]
    reg = np.asarray(d["regression_labels"], dtype=float)
    cls = np.asarray(d["classification_labels"], dtype=int)
    names = {0: "Negative", 1: "Neutral", 2: "Positive"}
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.5), sharey=True)
    for k, ax in enumerate(axes):
        v = reg[cls == k]
        ax.hist(v, bins=25, color=["#d62728", "#7f7f7f", "#2ca02c"][k], alpha=.7, edgecolor="white")
        ax.axvline(v.mean(), color="black", linestyle="--", lw=1.5)
        ax.set_title(f"{names[k]} (n={len(v)})")
        ax.set_xlabel("回归强度标签 [-3,3]")
        ax.text(0.05, 0.95, f"μ={v.mean():.2f}\nσ={v.std():.2f}",
                transform=ax.transAxes, va="top", fontsize=9)
        ax.grid(alpha=.3)
    axes[0].set_ylabel("样本数")
    fig.suptitle("训练集回归标签分布按极性分类 —— Q3 可解释性的数据基础", y=1.02)
    fig.tight_layout()
    save(fig, "raw_q3_label_distribution.png")


# ---------- 3. Q3 训练曲线 ----------
def fig_q3_curve():
    df = pd.read_csv(os.path.join(Q3_DIR, "training_log.csv"))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    ax = axes[0]
    ax.plot(df["epoch"], df["train_loss"], "-", color="#333")
    ax.set_xlabel("Epoch"); ax.set_ylabel("训练损失"); ax.set_title("Q3 训练损失")
    ax.grid(alpha=.3)
    ax = axes[1]
    ax.plot(df["epoch"], df["val_accuracy"], "-o", ms=3, label="Accuracy")
    ax.plot(df["epoch"], df["val_macro_f1"], "-s", ms=3, label="macro-F1")
    ax.plot(df["epoch"], df["val_pearson"], "-^", ms=3, label="Pearson")
    ax.plot(df["epoch"], df["val_mae"] / 3, "-d", ms=3, label="MAE/3")
    ax.set_xlabel("Epoch"); ax.set_ylabel("验证集指标")
    ax.set_title("Q3 A-EMN 验证集指标")
    ax.grid(alpha=.3); ax.legend()
    fig.tight_layout()
    save(fig, "process_q3_training_curve.png")


def main():
    print("loading aligned_50.pkl ...")
    with open(ALIGNED_50, "rb") as f:
        data = pickle.load(f)
    fig_valid_length(data)
    fig_label_dist(data)
    fig_q3_curve()
    print("done.")


if __name__ == "__main__":
    main()
