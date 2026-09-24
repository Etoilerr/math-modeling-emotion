# -*- coding: utf-8 -*-
"""
生成全部 Q2/Q3 图 + 总体建模流程图（300 DPI PNG，输出到 figures/）。
- process_q2_training_curve.png      训练曲线
- process_q2_missing_augmentation.png 缺失增强示意
- result_q2_missing_ablation.png     缺失消融热力图
- result_q2_confusion_matrix.png      测试集混淆矩阵
- result_q3_modality_weights.png      α_m 箱线图
- result_q3_fragment_heatmap.png     β_{m,t} 热力图（典型样本）
- result_q3_occlusion_validation.png top-K vs bottom-K
- flow_overall_model.png             总体建模流程图
"""
import os, sys, json, pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import torch
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paths import (ALIGNED_50, FIGURES_DIR, Q2_DIR, Q3_DIR, Q2_MODEL, Q3_MODEL,
                   Q2_TRAIN_LOG, Q2_MISSING_ANALYSIS, Q3_OCCLUSION,
                   ATTACH4_PRED, ATTACH4_FRAGMENT)
from model_common import (set_seed, build_mask, apply_scaler, MAMFN, AEMN,
                          MOD_NAMES, LABEL_NAMES, T_MAX)

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False
set_seed(42)
DEV = torch.device("cpu")
C = {"text": "#1f77b4", "audio": "#ff7f0e", "vision": "#2ca02c"}


def save(fig, name):
    p = os.path.join(FIGURES_DIR, name)
    fig.savefig(p, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("saved", p)


# ---------- 1. Q2 训练曲线 ----------
def fig_training_curve():
    df = pd.read_csv(Q2_TRAIN_LOG)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    ax = axes[0]
    ax.plot(df["epoch"], df["train_loss"], "-", color="#333", label="train loss")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Loss"); ax.set_title("训练损失")
    ax.grid(alpha=.3); ax.legend()
    ax = axes[1]
    ax.plot(df["epoch"], df["val_accuracy"], "-o", ms=3, label="Accuracy")
    ax.plot(df["epoch"], df["val_macro_f1"], "-s", ms=3, label="macro-F1")
    ax.plot(df["epoch"], df["val_pearson"], "-^", ms=3, label="Pearson")
    ax.plot(df["epoch"], df["val_mae"] / 3, "-d", ms=3, label="MAE/3")
    ax.set_xlabel("Epoch"); ax.set_ylabel("指标")
    ax.set_title("验证集指标随 epoch 变化"); ax.grid(alpha=.3); ax.legend()
    fig.tight_layout(); save(fig, "process_q2_training_curve.png")


# ---------- 2. 缺失增强示意 ----------
def fig_missing_aug():
    rng = np.random.RandomState(42)
    T = T_MAX
    # 构造一个示意样本：text 全有效，audio/vision 尾部填充
    orig = np.ones((3, T))
    orig[1, 35:] = 0; orig[2, 40:] = 0
    aug = orig.copy()
    # 增强：audio 随机连续区间 [10,22)
    s, L = 10, 12
    aug[1, s:s+L] = 0
    fig, axes = plt.subplots(2, 1, figsize=(9, 2.6), sharex=True)
    for ax, mat, title in [(axes[0], orig, "原始有效性掩码 M"), (axes[1], aug, "缺失增强后掩码（p=0.7，随机连续区间）")]:
        ax.imshow(mat, aspect="auto", cmap="Greys", vmin=0, vmax=1)
        ax.set_yticks([0, 1, 2]); ax.set_yticklabels(["text", "audio", "vision"])
        ax.set_title(title, fontsize=10)
        ax.set_xlim(-0.5, T - .5)
    axes[1].set_xlabel("时间位置 t (0-49)")
    # 标注增强区间
    axes[1].add_patch(plt.Rectangle((s - .5, 1.5), L, 1, fill=False, edgecolor="red", lw=2))
    axes[1].annotate("随机缺失区间 [s,s+L)\nL~U(5,25)", xy=(s + L/2, 1), xytext=(25, 0.2),
                     textcoords="data", color="red", fontsize=9,
                     arrowprops=dict(arrowstyle="->", color="red"))
    fig.tight_layout(); save(fig, "process_q2_missing_augmentation.png")


# ---------- 3. 缺失消融热力图 ----------
def fig_missing_ablation():
    df = pd.read_csv(Q2_MISSING_ANALYSIS)
    pub = df[df["modality"] != "none"].groupby(["modality", "missing_rate"])["macro_f1"].mean().unstack()
    rates = sorted(df[df["modality"] != "none"]["missing_rate"].unique())
    fig, ax = plt.subplots(figsize=(7.5, 4))
    im = ax.imshow(pub.values, cmap="RdYlGn", aspect="auto")
    ax.set_xticks(range(len(rates))); ax.set_xticklabels([f"{int(r*100)}%" for r in rates])
    ax.set_yticks(range(len(pub.index))); ax.set_yticklabels(pub.index)
    ax.set_xlabel("缺失率"); ax.set_ylabel("缺失模态类型")
    ax.set_title("缺失消融：macro-F1 随缺失模态 × 缺失率变化（位置平均）")
    for i in range(pub.shape[0]):
        for j in range(pub.shape[1]):
            ax.text(j, i, f"{pub.values[i,j]:.3f}", ha="center", va="center", fontsize=9)
    fig.colorbar(im, ax=ax); fig.tight_layout()
    save(fig, "result_q2_missing_ablation.png")


# ---------- 4. 混淆矩阵 ----------
def fig_confusion():
    with open(ALIGNED_50, "rb") as f:
        data = pickle.load(f)["test"]
    ck = torch.load(Q2_MODEL, map_location=DEV, weights_only=True)
    sc = ck["scaler"]
    model = MAMFN().to(DEV); model.load_state_dict(ck["state_dict"]); model.eval()
    t = apply_scaler(torch.tensor(data["text"], dtype=torch.float32), sc["t_mean"], sc["t_std"])
    a = apply_scaler(torch.tensor(data["audio"], dtype=torch.float32), sc["a_mean"], sc["a_std"])
    v = apply_scaler(torch.tensor(data["vision"], dtype=torch.float32), sc["v_mean"], sc["v_std"])
    mt, ma, mv = build_mask(t), build_mask(a), build_mask(v)
    yt = data["classification_labels"].astype(int)
    with torch.no_grad():
        pred = model(t, a, v, mt, ma, mv)[0].argmax(-1).numpy()
    cm = confusion_matrix(yt, pred, labels=[0, 1, 2])
    fig, ax = plt.subplots(figsize=(5.5, 5))
    im = ax.imshow(cm, cmap="Blues")
    labs = ["Negative", "Neutral", "Positive"]
    ax.set_xticks([0, 1, 2]); ax.set_xticklabels(labs)
    ax.set_yticks([0, 1, 2]); ax.set_yticklabels(labs)
    ax.set_xlabel("预测"); ax.set_ylabel("真实"); ax.set_title("Q2 测试集分类混淆矩阵")
    thresh = cm.max() / 2
    for i in range(3):
        for j in range(3):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.colorbar(im, ax=ax, fraction=.046); fig.tight_layout()
    save(fig, "result_q2_confusion_matrix.png")


# ---------- Q3 数据收集 ----------
def collect_q3():
    with open(ALIGNED_50, "rb") as f:
        data = pickle.load(f)["valid"]
    ck = torch.load(Q3_MODEL, map_location=DEV, weights_only=True)
    sc = ck["scaler"]
    model = AEMN().to(DEV); model.load_state_dict(ck["state_dict"]); model.eval()
    t = apply_scaler(torch.tensor(data["text"], dtype=torch.float32), sc["t_mean"], sc["t_std"])
    a = apply_scaler(torch.tensor(data["audio"], dtype=torch.float32), sc["a_mean"], sc["a_std"])
    v = apply_scaler(torch.tensor(data["vision"], dtype=torch.float32), sc["v_mean"], sc["v_std"])
    mt, ma, mv = build_mask(t), build_mask(a), build_mask(v)
    alphas, betas = [], []
    with torch.no_grad():
        for i in range(len(t)):
            _, _, al, (bt, ba, bv) = model(t[i:i+1], a[i:i+1], v[i:i+1], mt[i:i+1], ma[i:i+1], mv[i:i+1])
            alphas.append(al[0].numpy())
            betas.append(np.stack([bt[0].numpy(), ba[0].numpy(), bv[0].numpy()], 0))
    return np.array(alphas), np.array(betas)


def fig_modality_weights(alphas):
    fig, ax = plt.subplots(figsize=(6, 4.5))
    bp = ax.boxplot([alphas[:, 0], alphas[:, 1], alphas[:, 2]],
                    tick_labels=["text", "audio", "vision"], patch_artist=True, widths=.6)
    for patch, m in zip(bp["boxes"], MOD_NAMES):
        patch.set_facecolor(C[m]); patch.set_alpha(.6)
    ax.set_ylabel("模态注意力权重 α_m"); ax.set_title(f"验证集 α_m 分布（n={len(alphas)}）")
    ax.grid(alpha=.3, axis="y"); fig.tight_layout()
    save(fig, "result_q3_modality_weights.png")


def fig_fragment_heatmap(alphas, betas):
    # 选 2 个典型样本：α_text 最大 / α_vision 最大
    idx_t = int(np.argmax(alphas[:, 0]))
    idx_v = int(np.argmax(alphas[:, 2]))
    picks = [("样本 %d（主：text）" % idx_t, betas[idx_t]),
             ("样本 %d（主：vision）" % idx_v, betas[idx_v])]
    fig, axes = plt.subplots(2, 1, figsize=(10, 4.2))
    for ax, (title, b) in zip(axes, picks):
        im = ax.imshow(b, aspect="auto", cmap="magma")
        ax.set_yticks([0, 1, 2]); ax.set_yticklabels(["text", "audio", "vision"])
        ax.set_xlabel("时间位置 t"); ax.set_title(title + "  片段级重要性 β_{m,t}")
        fig.colorbar(im, ax=ax, fraction=.025)
    fig.tight_layout(); save(fig, "result_q3_fragment_heatmap.png")


def fig_occlusion():
    df = pd.read_csv(Q3_OCCLUSION)
    K = df["K"].tolist()
    x = np.arange(len(K)); w = .35
    fig, ax = plt.subplots(figsize=(7, 4.5))
    b1 = ax.bar(x - w/2, df["mean_delta_top"], w, label="移除 top-K", color="#d62728", alpha=.8)
    b2 = ax.bar(x + w/2, df["mean_delta_bottom"], w, label="移除 bottom-K", color="#1f77b4", alpha=.8)
    ax.set_xticks(x); ax.set_xticklabels([f"K={k}" for k in K])
    ax.set_ylabel("预测扰动 Δ"); ax.set_title("遮挡法验证：top-K vs bottom-K 片段移除")
    ax.legend(); ax.grid(alpha=.3, axis="y")
    for b in list(b1) + list(b2):
        ax.text(b.get_x() + b.get_width()/2, b.get_height(), f"{b.get_height():.3f}",
                ha="center", va="bottom", fontsize=8)
    fig.tight_layout(); save(fig, "result_q3_occlusion_validation.png")


# ---------- 8. 总体流程图 ----------
def fig_flow():
    fig, ax = plt.subplots(figsize=(13, 7.5))
    ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.axis("off")

    def box(x, y, w, h, text, fc):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.4",
                                    fc=fc, ec="#333", lw=1.2))
        ax.text(x + w/2, y + h/2, text, ha="center", va="center", fontsize=9.5)

    def arrow(x1, y1, x2, y2):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                                     mutation_scale=14, lw=1.3, color="#444"))

    ax.text(50, 97, "复杂场景下多模态情感预测——总体建模流程", ha="center", fontsize=14, weight="bold")
    box(2, 84, 20, 9, "输入\n附件1原始视频 / 附件2特征\n附件3缺失样本 / 附件4可解释样本", "#dbe9f6")
    box(28, 84, 20, 9, "Q1 特征提取与对齐\n三模态特征 → 50 位置\naligned_50.pkl", "#dbe9f6")
    # Q2
    box(54, 84, 20, 9, "Q2 MA-MFN\n掩码感知编码+门控融合\n双任务(分类+回归)", "#fde6d4")
    box(80, 84, 18, 9, "附件3 预测\n30条无标签", "#e8e8e8")
    box(54, 68, 20, 9, "缺失增强训练\np=0.7 连续区间置零\n类别加权CE+MSE", "#fde6d4")
    box(80, 68, 18, 9, "缺失消融\n模态×缺失率×位置", "#e8e8e8")
    # Q3
    box(54, 52, 20, 9, "Q3 A-EMN\n片段级β+模态级α\n注意力加权池化", "#d9ecd0")
    box(80, 52, 18, 9, "附件4 预测+解释\nα/top-K片段/关键帧", "#e8e8e8")
    box(54, 36, 20, 9, "遮挡法验证\ntop-K vs bottom-K\nΔ_top>Δ_bottom", "#d9ecd0")
    # 输出
    box(28, 20, 20, 9, "模型与指标\nvalid/test\nAcc/F1/MAE/Pearson", "#fff3b0")
    box(54, 20, 20, 9, "可视化与结论\n训练曲线/热力图/解释卡", "#fff3b0")
    box(80, 20, 18, 9, "可复现性说明\n种子/哈希/参数/命令", "#fff3b0")

    arrow(22, 88.5, 28, 88.5)
    arrow(48, 88.5, 54, 88.5)
    arrow(74, 88.5, 80, 88.5)
    arrow(64, 84, 64, 77)
    arrow(74, 72.5, 80, 72.5)
    arrow(64, 68, 64, 61)
    arrow(74, 56.5, 80, 56.5)
    arrow(64, 52, 64, 45)
    arrow(64, 36, 64, 29)
    arrow(54, 24.5, 48, 24.5)
    arrow(74, 24.5, 80, 24.5)
    arrow(38, 84, 38, 29)
    fig.tight_layout(); save(fig, "flow_overall_model.png")


def main():
    fig_training_curve()
    fig_missing_aug()
    fig_missing_ablation()
    fig_confusion()
    alphas, betas = collect_q3()
    np.save(os.path.join(Q3_DIR, "_alphas.npy"), alphas)
    np.save(os.path.join(Q3_DIR, "_betas.npy"), betas)
    fig_modality_weights(alphas)
    fig_fragment_heatmap(alphas, betas)
    fig_occlusion()
    fig_flow()
    print("all figures done.")


if __name__ == "__main__":
    main()
