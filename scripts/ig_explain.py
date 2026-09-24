# -*- coding: utf-8 -*-
"""
Q3 积分梯度（Integrated Gradients, IG）解释。
对附件4 20 条样本，分别计算 text/audio/vision 每个时间步的 IG 重要性。
baseline = 0（全零输入），路径 M=20 步线性插值。
目标 = 预测类别的 logit（对分类）+ 回归输出绝对值（对强度）。
输出: attachment4_ig_importance.csv
"""
import os, sys, pickle, glob, json
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paths import (ALIGNED_50, ATTACH4_DIR, Q3_DIR, Q3_MODEL,
                   ATTACH4_FRAGMENT)
from model_common import (set_seed, build_mask, apply_scaler, AEMN,
                          LABEL_NAMES, MOD_NAMES, T_MAX)

DEVICE = torch.device("cpu")
set_seed(42)
M_STEPS = 20  # IG 路径步数


def load_model():
    ckpt = torch.load(Q3_MODEL, map_location=DEVICE, weights_only=True)
    m = AEMN(d_model=32).to(DEVICE); m.load_state_dict(ckpt["state_dict"]); m.eval()
    return m, ckpt["scaler"]


def integrated_gradients(model, t, a, v, mt, ma, mv, target_class=None):
    """对单个样本计算三模态 IG。
    返回: dict[mod] -> (T,) IG 重要性（按时间步求和后的绝对值）。
    """
    # baseline = 0
    bl_t = torch.zeros_like(t)
    bl_a = torch.zeros_like(a)
    bl_v = torch.zeros_like(v)

    # 先跑一次拿预测类别
    with torch.no_grad():
        logits, reg, _, _ = model(t, a, v, mt, ma, mv)
        if target_class is None:
            target_class = logits.argmax(dim=-1).item()

    # 累积梯度
    sum_grad_t = torch.zeros_like(t)
    sum_grad_a = torch.zeros_like(a)
    sum_grad_v = torch.zeros_like(v)

    for k in range(1, M_STEPS + 1):
        alpha = k / M_STEPS
        t_k = (bl_t + alpha * (t - bl_t)).clone().requires_grad_(True)
        a_k = (bl_a + alpha * (a - bl_a)).clone().requires_grad_(True)
        v_k = (bl_v + alpha * (v - bl_v)).clone().requires_grad_(True)
        logits_k, reg_k, _, _ = model(t_k, a_k, v_k, mt, ma, mv)
        # 目标 = 预测类别的 logit + 回归输出
        target = logits_k[0, target_class] + reg_k[0].abs()
        model.zero_grad()
        target.backward()
        sum_grad_t += t_k.grad
        sum_grad_a += a_k.grad
        sum_grad_v += v_k.grad

    # IG = (x - baseline) × 平均梯度
    ig_t = (t - bl_t) * sum_grad_t / M_STEPS
    ig_a = (a - bl_a) * sum_grad_a / M_STEPS
    ig_v = (v - bl_v) * sum_grad_v / M_STEPS

    # 按时间步取绝对值求和（跨特征维）
    return {
        "text": ig_t[0].abs().sum(dim=-1).detach().numpy(),      # (T,)
        "audio": ig_a[0].abs().sum(dim=-1).detach().numpy(),
        "vision": ig_v[0].abs().sum(dim=-1).detach().numpy(),
    }


def main():
    model, sc = load_model()
    files = sorted(glob.glob(os.path.join(ATTACH4_DIR, "*.pkl")))
    assert len(files) == 20
    rows = []
    for fp in files:
        sid = os.path.basename(fp).replace(".pkl", "")
        d = pickle.load(open(fp, "rb"))
        t = apply_scaler(torch.tensor(d["text"], dtype=torch.float32).unsqueeze(0),
                         sc["t_mean"], sc["t_std"])
        a = apply_scaler(torch.tensor(d["audio"], dtype=torch.float32).unsqueeze(0),
                         sc["a_mean"], sc["a_std"])
        v = apply_scaler(torch.tensor(d["vision"], dtype=torch.float32).unsqueeze(0),
                         sc["v_mean"], sc["v_std"])
        mt, ma, mv = build_mask(t), build_mask(a), build_mask(v)
        ig = integrated_gradients(model, t, a, v, mt, ma, mv)
        for mi, mname in enumerate(MOD_NAMES):
            m = [mt, ma, mv][mi][0].numpy()
            for ti in range(T_MAX):
                if m[ti] > 0:
                    rows.append({
                        "sample_id": sid, "modality": mname,
                        "position": ti,
                        "ig_importance": round(float(ig[mname][ti]), 6),
                    })
        print(f"  {sid}: IG done | text_top={ig['text'].argmax()} "
              f"audio_top={ig['audio'].argmax()} vision_top={ig['vision'].argmax()}")
    out = Q3_DIR / "attachment4_ig_importance.csv"
    pd.DataFrame(rows).to_csv(out, index=False, encoding="utf-8-sig")
    print(f"saved {len(rows)} IG rows -> {out}")


if __name__ == "__main__":
    main()
