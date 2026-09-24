# -*- coding: utf-8 -*-
"""
Q1 四张正式图绘制（PNG 300 DPI, 10x6 inch, 中文字体）。
读 Q1_features/*.npz|csv 与 label-100.xlsx，输出到 figures/。
"""
import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# ---- 全局样式 ----
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 110
plt.rcParams["savefig.dpi"] = 300
plt.rcParams["font.size"] = 10

ROOT = Path(r"D:\Projects\math_modling")
FIG = ROOT / "figures"
FIG.mkdir(exist_ok=True)
Q1 = ROOT / "Q1_features"
LABEL = ROOT / r"E题数据\附件1-数据集原始多模态样本\MOSEI数据集部分原始视频-100条\label-100.xlsx"

OUT = dict(dpi=300, bbox_inches="tight")

# 色盲友好配色（Okabe-Ito）
C_POS, C_NEU, C_NEG = "#009E73", "#E69F00", "#D55E00"
C_TEXT, C_AUD, C_VIS = "#0072B2", "#CC79A7", "#E69F00"


# ============ 公共数据 ============
npz = np.load(Q1 / "features_aligned_50.npz")
summary = pd.read_csv(Q1 / "summary.csv")
mapping = pd.read_csv(Q1 / "mapping.csv")
label_df = pd.read_excel(LABEL)


# ============ 图1: label 分布直方图 + annotation 饼图 ============
def fig1():
    fig, axes = plt.subplots(1, 2, figsize=(10, 5.2),
                             gridspec_kw={"width_ratios": [1.4, 1]})
    ax = axes[0]
    bins = np.arange(-2.0, 3.0, 0.5)
    n, bins_out, patches = ax.hist(label_df["label"], bins=bins,
                                   edgecolor="white", color="#0072B2", alpha=0.85)
    # 按极性上色
    for b, p in zip(bins_out[:-1], patches):
        if b < -0.01:
            p.set_facecolor(C_NEG)
        elif b > 0.01:
            p.set_facecolor(C_POS)
        else:
            p.set_facecolor(C_NEU)
    mu = label_df["label"].mean()
    med = label_df["label"].median()
    ax.axvline(mu, color="black", ls="--", lw=1.2, label=f"均值={mu:.3f}")
    ax.axvline(med, color="gray", ls=":", lw=1.2, label=f"中位数={med:.3f}")
    ax.set_xlabel("情感强度 label（[-2, 2.667]）")
    ax.set_ylabel("样本数")
    ax.set_title("(a) 100 条样本情感强度分布")
    ax.legend(fontsize=9, loc="upper left")
    ax.grid(alpha=0.25)

    ax2 = axes[1]
    vc = label_df["annotation"].value_counts().reindex(["Positive", "Neutral", "Negative"])
    colors = [C_POS, C_NEU, C_NEG]
    wedges, texts, autotexts = ax2.pie(
        vc.values, labels=vc.index, colors=colors, autopct="%1.0f%%",
        startangle=90, wedgeprops=dict(edgecolor="white", linewidth=1.5),
        textprops=dict(fontsize=10))
    for at in autotexts:
        at.set_color("white"); at.set_fontweight("bold")
    ax2.set_title(f"(b) 极性分布（n=100）")
    ax2.legend(wedges, [f"{k}: {v} 条" for k, v in vc.items()],
               loc="lower center", bbox_to_anchor=(0.5, -0.12), ncol=3, fontsize=9)

    fig.suptitle("Q1 原始标签：情感强度直方图与极性构成", fontsize=12, y=1.02)
    fig.tight_layout()
    out = FIG / "raw_q1_label_distribution.png"
    fig.savefig(out, **OUT)
    plt.close(fig)
    print("[OK]", out)


# ============ 图2: 时长分布直方图 ============
def fig2():
    d = summary["duration_sec"].values
    fig, ax = plt.subplots(figsize=(10, 5.5))
    n, bins, patches = ax.hist(d, bins=18, edgecolor="white",
                               color="#0072B2", alpha=0.85)
    mu, med = d.mean(), np.median(d)
    ax.axvline(mu, color=C_POS, ls="--", lw=1.4, label=f"均值={mu:.2f}s")
    ax.axvline(med, color=C_NEG, ls=":", lw=1.6, label=f"中位数={med:.2f}s")
    # 标注范围
    ax.annotate(f"范围 [{d.min():.2f}, {d.max():.2f}]s",
                xy=(0.98, 0.95), xycoords="axes fraction", ha="right",
                fontsize=10,
                bbox=dict(boxstyle="round,pad=0.4", fc="#F0F0F0", ec="gray"))
    # 在最高柱上标数值
    imax = int(np.argmax(n))
    ax.text((bins[imax] + bins[imax + 1]) / 2, n[imax] + 0.4,
            f"{int(n[imax])}", ha="center", fontsize=9)
    ax.set_xlabel("视频时长 duration_sec（秒）")
    ax.set_ylabel("样本数")
    ax.set_title("Q1 100 条视频时长分布（mean=7.87s, median=6.72s, range=[2.26, 29.29]s）")
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    out = FIG / "raw_q1_duration_distribution.png"
    fig.savefig(out, **OUT)
    plt.close(fig)
    print("[OK]", out)


# ============ 图3: 单样本三模态对齐示例 ============
def fig3(sid=0):
    T = float(summary.loc[summary["sample_id"] == sid, "duration_sec"].iloc[0])
    m = mapping[mapping["sample_id"] == sid].sort_values("aligned_position").reset_index(drop=True)

    fig, axes = plt.subplots(3, 1, figsize=(10, 7.2), sharex=True,
                             gridspec_kw={"hspace": 0.12})

    # --- 上：文本词时间区间（单行甘特：所有词按时间从左到右排）---
    ax = axes[0]
    # 从 mapping 还原全量词序列（去重、按出现顺序）
    all_words = []
    for _, r in m.iterrows():
        seg = str(r["text_segment"])
        if seg and seg != "nan":
            for w in seg.split():
                if w not in all_words:
                    all_words.append(w)
    nw = len(all_words)
    for k, w in enumerate(all_words):
        ws = k * T / nw
        we = (k + 1) * T / nw if k < nw - 1 else T
        ax.barh(0, we - ws, left=ws, height=0.5,
                color=C_TEXT, alpha=0.85, edgecolor="white", lw=0.5)
        # 词标签放条上方，倾斜避免重叠
        ax.text((ws + we) / 2, 0.35, w, ha="center", va="bottom",
                fontsize=7, color="#003366", rotation=45, rotation_mode="anchor")
    ax.set_ylim(-0.6, 1.2)
    ax.set_yticks([0])
    ax.set_yticklabels([f"文本词序 (n={nw})"], fontsize=9)
    ax.set_title(f"样本 sample_id={sid}  T={T:.2f}s  三模态时间轴对齐（50 位置）",
                 fontsize=11)
    ax.grid(alpha=0.25)

    # --- 中：语音特征范数 ---
    ax = axes[1]
    centers = (m["time_start"] + m["time_end"]) / 2
    ax.fill_between(centers, m["feature_norm_audio"].values, step="mid",
                    alpha=0.25, color=C_AUD)
    ax.plot(centers, m["feature_norm_audio"].values, color=C_AUD, lw=1.6,
            label="audio 特征 L2 范数")
    ax.set_ylabel("音频特征\nL2 范数", fontsize=9)
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(alpha=0.25)

    # --- 下：视觉特征范数 ---
    ax = axes[2]
    ax.fill_between(centers, m["feature_norm_vision"].values, step="mid",
                    alpha=0.25, color=C_VIS)
    ax.plot(centers, m["feature_norm_vision"].values, color=C_VIS, lw=1.6,
            label="vision 特征 L2 范数")
    ax.set_ylabel("视觉特征\nL2 范数", fontsize=9)
    ax.set_xlabel("时间（秒）")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(alpha=0.25)

    # 垂直虚线标注 50 个位置边界（只在中、下图画）
    for ax in axes[1:]:
        for i in range(51):
            bx = i * T / 50
            ax.axvline(bx, color="gray", ls=":", lw=0.4, alpha=0.6)
    axes[2].set_xlim(0, T)

    fig.tight_layout()
    out = FIG / "process_q1_alignment_example.png"
    fig.savefig(out, **OUT)
    plt.close(fig)
    print("[OK]", out)


# ============ 图4: 三模态有效长度热力图 ============
def fig4():
    s = summary.sort_values("sample_id").reset_index(drop=True)
    mat = np.vstack([
        s["text_valid_len"].values / 50.0,
        s["audio_valid_len"].values / 50.0,
        s["vision_valid_len"].values / 50.0,
    ])  # (3,100)

    fig = plt.figure(figsize=(11, 4.6))
    gs = fig.add_gridspec(1, 2, width_ratios=[3.2, 1], wspace=0.08)
    ax = fig.add_subplot(gs[0, 0])
    im = ax.imshow(mat, aspect="auto", cmap="YlGnBu",
                   vmin=0, vmax=1.0, interpolation="nearest")
    ax.set_yticks([0, 1, 2])
    ax.set_yticklabels(["文本 text\n(768维)", "语音 audio\n(74维)", "视觉 vision\n(35维)"])
    ax.set_xlabel("样本 sample_id（0–99，按处理顺序）")
    ax.set_title("Q1 三模态有效对齐位置占比热力图（颜色 = valid_len/50）")
    ax.set_xticks(np.arange(0, 100, 10))
    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("有效位置比例")

    # 右侧统计摘要
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.axis("off")
    rows = [
        ("模态", "均值", "最小", "最大"),
        ("文本", *[f"{x:.2f}" for x in [s['text_valid_len'].mean(),
                                        s['text_valid_len'].min(),
                                        s['text_valid_len'].max()]]),
        ("语音", *[f"{x:.2f}" for x in [s['audio_valid_len'].mean(),
                                        s['audio_valid_len'].min(),
                                        s['audio_valid_len'].max()]]),
        ("视觉", *[f"{x:.2f}" for x in [s['vision_valid_len'].mean(),
                                        s['vision_valid_len'].min(),
                                        s['vision_valid_len'].max()]]),
    ]
    tbl = ax2.table(cellText=rows, loc="upper left", cellLoc="center")
    tbl.auto_set_font_size(False); tbl.set_fontsize(9); tbl.scale(1, 1.4)
    for (r, c), cell in tbl.get_celld().items():
        if r == 0:
            cell.set_facecolor("#4472C4"); cell.set_text_props(color="white")
    ax2.text(0.0, 0.05,
             "结论：文本/语音在 100 条样本上几乎全覆盖\n"
             "(均值≥49.6/50)，视觉受帧率与时长影响\n"
             "均值仅 34.3/50，是主要的零填充来源。",
             transform=ax2.transAxes, fontsize=9, va="bottom",
             bbox=dict(boxstyle="round,pad=0.4", fc="#FFF2CC", ec="#BF9000"))

    fig.tight_layout()
    out = FIG / "result_q1_feature_summary.png"
    fig.savefig(out, **OUT)
    plt.close(fig)
    print("[OK]", out)


if __name__ == "__main__":
    fig1()
    fig2()
    fig3(0)
    fig4()
    print("ALL DONE")
