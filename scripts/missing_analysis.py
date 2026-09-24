# -*- coding: utf-8 -*-
"""
Q2 缺失消融：在 valid 集上系统构造缺失（模态类型 × 缺失率 × 缺失位置）。
模态: text/audio/vision/随机；缺失率 0.1..0.5；位置 head/middle/tail。
输出 missing_analysis.csv（+ 快速图 missing_analysis.png）。
"""
import os, sys, pickle
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paths import ALIGNED_50, Q2_DIR, Q2_MODEL, Q2_MISSING_ANALYSIS
from model_common import (set_seed, build_mask, apply_scaler, MAMFN,
                          compute_metrics, T_MAX)

DEVICE = torch.device("cpu")
set_seed(42)
MODS = ["text", "audio", "vision", "随机"]
RATES = [0.1, 0.2, 0.3, 0.4, 0.5]
POSITIONS = ["head", "middle", "tail"]


@torch.no_grad()
def evaluate(model, t, a, v, mt, ma, mv, c, r):
    logits, reg = model(t, a, v, mt, ma, mv)
    pred = logits.argmax(-1).numpy()
    return compute_metrics(c, pred, r, reg.numpy())


def main():
    with open(ALIGNED_50, "rb") as f:
        data = pickle.load(f)["valid"]
    t = torch.tensor(data["text"], dtype=torch.float32)
    a = torch.tensor(data["audio"], dtype=torch.float32)
    v = torch.tensor(data["vision"], dtype=torch.float32)
    c = data["classification_labels"].astype(int)
    r = data["regression_labels"].astype(float)
    mt, ma, mv = build_mask(t), build_mask(a), build_mask(v)

    ckpt = torch.load(Q2_MODEL, map_location=DEVICE, weights_only=True)
    sc = ckpt["scaler"]
    model = MAMFN().to(DEVICE); model.load_state_dict(ckpt["state_dict"]); model.eval()

    t = apply_scaler(t, sc["t_mean"], sc["t_std"])
    a = apply_scaler(a, sc["a_mean"], sc["a_std"])
    v = apply_scaler(v, sc["v_mean"], sc["v_std"])

    base = evaluate(model, t, a, v, mt, ma, mv, c, r)
    print("baseline:", base)

    rng = np.random.RandomState(42)
    rows = [{"modality": "none", "missing_rate": 0.0, "position": "-", **base}]
    for mod in MODS:
        for rate in RATES:
            L = int(round(rate * T_MAX))
            for pos in POSITIONS:
                tt, aa, vv = t.clone(), a.clone(), v.clone()
                mt2, ma2, mv2 = mt.clone(), ma.clone(), mv.clone()
                if pos == "head":
                    s = 0
                elif pos == "middle":
                    s = (T_MAX - L) // 2
                else:
                    s = T_MAX - L
                e = s + L
                if mod == "随机":
                    choice = rng.randint(0, 3, size=len(c))
                    for i in range(len(c)):
                        m = choice[i]
                        if m == 0: tt[i, s:e] = 0; mt2[i, s:e] = 0
                        elif m == 1: aa[i, s:e] = 0; ma2[i, s:e] = 0
                        else: vv[i, s:e] = 0; mv2[i, s:e] = 0
                else:
                    idx = MODS.index(mod)
                    if idx == 0: tt[:, s:e] = 0; mt2[:, s:e] = 0
                    elif idx == 1: aa[:, s:e] = 0; ma2[:, s:e] = 0
                    else: vv[:, s:e] = 0; mv2[:, s:e] = 0
                m = evaluate(model, tt, aa, vv, mt2, ma2, mv2, c, r)
                rows.append({"modality": mod, "missing_rate": rate, "position": pos, **m})
                print(f"  {mod:6s} rate={rate:.1f} {pos:6s} -> f1={m['macro_f1']:.4f} acc={m['accuracy']:.4f}")

    df = pd.DataFrame(rows)
    df.to_csv(Q2_MISSING_ANALYSIS, index=False, encoding="utf-8-sig")
    print(f"saved {len(df)} rows -> {Q2_MISSING_ANALYSIS}")

    # 快速热力图（模态×缺失率，位置平均）
    pub = df[df["modality"] != "none"].groupby(["modality", "missing_rate"])["macro_f1"].mean().unstack()
    fig, ax = plt.subplots(figsize=(7, 4.5))
    im = ax.imshow(pub.values, cmap="RdYlGn", aspect="auto")
    ax.set_xticks(range(len(RATES))); ax.set_xticklabels([f"{int(x*100)}%" for x in RATES])
    ax.set_yticks(range(len(pub.index))); ax.set_yticklabels(pub.index)
    for i in range(len(pub.index)):
        for j in range(len(RATES)):
            ax.text(j, i, f"{pub.values[i,j]:.3f}", ha="center", va="center", fontsize=8)
    ax.set_xlabel("缺失率"); ax.set_ylabel("缺失模态")
    ax.set_title("缺失消融 macro-F1（位置平均）")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(os.path.join(Q2_DIR, "missing_analysis.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("quick png saved.")


if __name__ == "__main__":
    main()
