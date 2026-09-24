# -*- coding: utf-8 -*-
"""
Q2 融合方式对比实验：MA-MFN（池化级门控融合） vs MAMFN_ACB（TACFN 自适应跨模态块）。
严格同口径：同一数据管线 / 同一超参 / 同一种子 / 同一缺失增强 / 同一早停规则。
输出：valid/test 指标 + 测试集"audio+vision 同步连续缺失"消融（缺失率 10%/30%/50%）。
结果写 Q2_results/fusion_comparison.json。
"""
import os, sys, json, pickle, time, random
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paths import ALIGNED_50, Q2_DIR
from model_common import (set_seed, build_mask, fit_scaler, apply_scaler,
                          MAMFN, MAMFN_ACB, apply_missing_augmentation,
                          compute_metrics, SEED, BATCH_SIZE, EPOCHS, LR,
                          WEIGHT_DECAY, LAMBDA_CE, LAMBDA_MSE, D_MODEL, T_MAX)

DEVICE = torch.device("cpu")
set_seed(SEED)


def to_tensors(d):
    t = torch.tensor(d["text"], dtype=torch.float32)
    a = torch.tensor(d["audio"], dtype=torch.float32)
    v = torch.tensor(d["vision"], dtype=torch.float32)
    c = torch.tensor(d["classification_labels"], dtype=torch.long)
    r = torch.tensor(d["regression_labels"], dtype=torch.float32)
    return t, a, v, c, r


class MultiModalDS(Dataset):
    def __init__(self, t, a, v, c, r, mt, ma, mv, augment=False):
        self.t, self.a, self.v, self.c, self.r = t, a, v, c, r
        self.mt, self.ma, self.mv = mt, ma, mv
        self.augment = augment
        self.rng = random.Random(SEED)

    def __len__(self):
        return len(self.c)

    def __getitem__(self, i):
        t = self.t[i].clone(); a = self.a[i].clone(); v = self.v[i].clone()
        mt = self.mt[i].clone(); ma = self.ma[i].clone(); mv = self.mv[i].clone()
        if self.augment:
            t, a, v, mt, ma, mv = apply_missing_augmentation(t, a, v, mt, ma, mv, self.rng)
        return t, a, v, mt, ma, mv, self.c[i], self.r[i]


@torch.no_grad()
def run_eval(model, loader):
    model.eval()
    pc, tc, pr, tr = [], [], [], []
    for t, a, v, mt, ma, mv, yc, yr in loader:
        logits, reg = model(t, a, v, mt, ma, mv)
        pc.extend(logits.argmax(-1).numpy().tolist())
        tc.extend(yc.numpy().tolist())
        pr.extend(reg.numpy().tolist())
        tr.extend(yr.numpy().tolist())
    return compute_metrics(tc, pc, tr, pr)


@torch.no_grad()
def corrupt_and_eval(model, te_t, te_a, te_v, te_c, te_r, te_mt, te_ma, te_mv,
                     rate, draws=3):
    """测试集缺失消融：每样本在 audio+vision 同步随机连续段置零（对齐附件3实测模式），
    多次随机取平均。"""
    L = max(1, int(round(rate * T_MAX)))
    accs, f1s, maes, pears = [], [], [], []
    for d in range(draws):
        rng = random.Random(SEED + 1000 + d)
        a2, v2 = te_a.clone(), te_v.clone()
        ma2, mv2 = te_ma.clone(), te_mv.clone()
        for i in range(len(te_c)):
            if rng.random() < 0.9:
                s = rng.randint(1, T_MAX - L)
                a2[i, s:s + L, :] = 0.0; ma2[i, s:s + L] = 0.0
                v2[i, s:s + L, :] = 0.0; mv2[i, s:s + L] = 0.0
        loader = DataLoader(list(zip(te_t, a2, v2, te_mt, ma2, mv2, te_c, te_r)),
                            batch_size=BATCH_SIZE, shuffle=False)
        m = run_eval(model, loader)
        accs.append(m["accuracy"]); f1s.append(m["macro_f1"])
        maes.append(m["mae"]); pears.append(m["pearson"])
    return {"accuracy": round(float(np.mean(accs)), 4),
            "macro_f1": round(float(np.mean(f1s)), 4),
            "mae": round(float(np.mean(maes)), 4),
            "pearson": round(float(np.mean(pears)), 4)}


def train_eval(model, train_loader, valid_loader, test_loader, cls_w, tag, mode="best"):
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    ce_loss = torch.nn.CrossEntropyLoss(weight=cls_w)
    huber_loss = torch.nn.HuberLoss(delta=1.0)
    best_f1 = -1.0
    best_state = None
    last_state = None
    bad_epochs = 0
    for epoch in range(1, EPOCHS + 1):
        model.train()
        tl = 0.0; nb = 0
        for t, a, v, mt, ma, mv, yc, yr in train_loader:
            logits, reg = model(t, a, v, mt, ma, mv)
            loss = LAMBDA_CE * ce_loss(logits, yc) + LAMBDA_MSE * huber_loss(reg, yr)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tl += loss.item(); nb += 1
        sched.step()
        vm = run_eval(model, valid_loader)
        print(f"[{tag}] Ep {epoch:02d}/{EPOCHS} loss={tl/nb:.4f} | "
              f"val acc={vm['accuracy']:.4f} f1={vm['macro_f1']:.4f} "
              f"mae={vm['mae']:.4f} pear={vm['pearson']:.4f}")
        if vm["macro_f1"] > best_f1:
            best_f1 = vm["macro_f1"]
            best_state = {k: w.clone() for k, w in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1
            if mode == "best" and bad_epochs >= 8:
                print(f"[{tag}] early stop at epoch {epoch}")
                break
        last_state = {k: w.clone() for k, w in model.state_dict().items()}
    if mode == "full":
        # 固定 30 轮，取末轮权重（诊断充分训练下的差异）
        model.load_state_dict(last_state)
        state, note = last_state, "last-epoch(30)"
    else:
        model.load_state_dict(best_state)
        state, note = best_state, "best-valid-macroF1"
    valid_m = run_eval(model, valid_loader)
    test_m = run_eval(model, test_loader)
    return {"n_params": n_params, "best_val_macro_f1": round(best_f1, 4),
            "selection": note, "valid": valid_m, "test": test_m,
            "best_state": state}


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "best"   # best / full
    t0 = time.time()
    print("Loading aligned_50.pkl ...")
    with open(ALIGNED_50, "rb") as f:
        data = pickle.load(f)
    tr_t, tr_a, tr_v, tr_c, tr_r = to_tensors(data["train"])
    va_t, va_a, va_v, va_c, va_r = to_tensors(data["valid"])
    te_t, te_a, te_v, te_c, te_r = to_tensors(data["test"])

    tr_mt, tr_ma, tr_mv = build_mask(tr_t), build_mask(tr_a), build_mask(tr_v)
    va_mt, va_ma, va_mv = build_mask(va_t), build_mask(va_a), build_mask(va_v)
    te_mt, te_ma, te_mv = build_mask(te_t), build_mask(te_a), build_mask(te_v)

    st_mu_t, st_sd_t = fit_scaler(tr_t, tr_mt)
    st_mu_a, st_sd_a = fit_scaler(tr_a, tr_ma)
    st_mu_v, st_sd_v = fit_scaler(tr_v, tr_mv)
    tr_t = apply_scaler(tr_t, st_mu_t, st_sd_t); va_t = apply_scaler(va_t, st_mu_t, st_sd_t); te_t = apply_scaler(te_t, st_mu_t, st_sd_t)
    tr_a = apply_scaler(tr_a, st_mu_a, st_sd_a); va_a = apply_scaler(va_a, st_mu_a, st_sd_a); te_a = apply_scaler(te_a, st_mu_a, st_sd_a)
    tr_v = apply_scaler(tr_v, st_mu_v, st_sd_v); va_v = apply_scaler(va_v, st_mu_v, st_sd_v); te_v = apply_scaler(te_v, st_mu_v, st_sd_v)

    counts = torch.bincount(tr_c, minlength=3).float()
    cls_w = counts.sum() / (3.0 * counts)
    print("class weights:", cls_w.numpy().round(4).tolist())

    train_loader = DataLoader(MultiModalDS(tr_t, tr_a, tr_v, tr_c, tr_r,
                                           tr_mt, tr_ma, tr_mv, augment=True),
                              batch_size=BATCH_SIZE, shuffle=True)
    valid_loader = DataLoader(MultiModalDS(va_t, va_a, va_v, va_c, va_r,
                                           va_mt, va_ma, va_mv, augment=False),
                              batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(MultiModalDS(te_t, te_a, te_v, te_c, te_r,
                                          te_mt, te_ma, te_mv, augment=False),
                             batch_size=BATCH_SIZE, shuffle=False)

    print("\n===== [1/2] 基线 MA-MFN（门控融合）=====")
    base = train_eval(MAMFN().to(DEVICE), train_loader, valid_loader, test_loader, cls_w, "baseline", mode)
    print("baseline valid:", base["valid"])
    print("baseline test :", base["test"])

    print("\n===== [2/2] MAMFN_ACB（TACFN 自适应跨模态块）=====")
    acb = train_eval(MAMFN_ACB().to(DEVICE), train_loader, valid_loader, test_loader, cls_w, "acb", mode)
    print("acb valid:", acb["valid"])
    print("acb test :", acb["test"])

    # 测试集缺失消融（用各自 best_state 重载评估）
    print("\n===== 测试集缺失消融（audio+vision 同步连续缺失）=====")
    base_model = MAMFN().to(DEVICE); base_model.load_state_dict(base["best_state"])
    acb_model = MAMFN_ACB().to(DEVICE); acb_model.load_state_dict(acb["best_state"])
    base["missing"] = {}
    acb["missing"] = {}
    for rate in (0.1, 0.3, 0.5):
        base["missing"][str(rate)] = corrupt_and_eval(
            base_model, te_t, te_a, te_v, te_c, te_r, te_mt, te_ma, te_mv, rate)
        acb["missing"][str(rate)] = corrupt_and_eval(
            acb_model, te_t, te_a, te_v, te_c, te_r, te_mt, te_ma, te_mv, rate)
        print(f"rate={rate}: baseline {base['missing'][str(rate)]} | acb {acb['missing'][str(rate)]}")

    base.pop("best_state", None)
    acb.pop("best_state", None)
    out = {
        "experiment": "Q2 fusion comparison: MA-MFN(gated) vs MAMFN_ACB(TACFN)",
        "mode": mode,
        "seed": SEED, "d_model": D_MODEL, "epochs": EPOCHS,
        "aug": "audio+vision 同步短缺失(0-4段×1-4位置)",
        "missing_ablation": "audio+vision 同步随机连续段置零，缺失率×50 位置",
        "baseline": base, "acb": acb,
        "runtime_min": round((time.time() - t0) / 60, 2),
    }
    fname = "fusion_comparison.json" if mode == "best" else "fusion_comparison_full30.json"
    with open(Q2_DIR / fname, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"\n saved -> {Q2_DIR / fname}  (runtime {out['runtime_min']} min)")


if __name__ == "__main__":
    main()
