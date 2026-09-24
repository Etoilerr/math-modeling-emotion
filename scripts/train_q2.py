# -*- coding: utf-8 -*-
"""
Q2 训练：MA-MFN（掩码感知多模态融合网络）。
数据加载 -> 掩码构造 -> train 统计量标准化 -> 缺失增强训练 -> 验证选模(macro-F1)
-> 测试评估 -> 保存模型/指标/配置/训练日志。
"""
import os, sys, json, pickle, time, random
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paths import (ALIGNED_50, Q2_DIR, MODELS_DIR, Q2_MODEL,
                   Q2_METRICS, Q2_CONFIG, Q2_TRAIN_LOG)
from model_common import (set_seed, build_mask, fit_scaler, apply_scaler,
                          MAMFN, apply_missing_augmentation, compute_metrics,
                          SEED, BATCH_SIZE, EPOCHS, LR, WEIGHT_DECAY,
                          LAMBDA_CE, LAMBDA_MSE, D_MODEL, T_MAX)

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


def main():
    t0 = time.time()
    print("Loading aligned_50.pkl ...")
    with open(ALIGNED_50, "rb") as f:
        data = pickle.load(f)
    print(f"loaded in {time.time()-t0:.1f}s")

    tr_t, tr_a, tr_v, tr_c, tr_r = to_tensors(data["train"])
    va_t, va_a, va_v, va_c, va_r = to_tensors(data["valid"])
    te_t, te_a, te_v, te_c, te_r = to_tensors(data["test"])

    # 掩码（基于原始特征范数）
    tr_mt, tr_ma, tr_mv = build_mask(tr_t), build_mask(tr_a), build_mask(tr_v)
    va_mt, va_ma, va_mv = build_mask(va_t), build_mask(va_a), build_mask(va_v)
    te_mt, te_ma, te_mv = build_mask(te_t), build_mask(te_a), build_mask(te_v)

    # 标准化（仅用 train 统计量，有效位置）
    st_mu_t, st_sd_t = fit_scaler(tr_t, tr_mt)
    st_mu_a, st_sd_a = fit_scaler(tr_a, tr_ma)
    st_mu_v, st_sd_v = fit_scaler(tr_v, tr_mv)
    tr_t, va_t, te_t = apply_scaler(tr_t, st_mu_t, st_sd_t), apply_scaler(va_t, st_mu_t, st_sd_t), apply_scaler(te_t, st_mu_t, st_sd_t)
    tr_a, va_a, te_a = apply_scaler(tr_a, st_mu_a, st_sd_a), apply_scaler(va_a, st_mu_a, st_sd_a), apply_scaler(te_a, st_mu_a, st_sd_a)
    tr_v, va_v, te_v = apply_scaler(tr_v, st_mu_v, st_sd_v), apply_scaler(va_v, st_mu_v, st_sd_v), apply_scaler(te_v, st_mu_v, st_sd_v)

    print(f"train {tuple(tr_t.shape)} | valid {tuple(va_t.shape)} | test {tuple(te_t.shape)}")

    # 类别加权（与 train 频率成反比）
    counts = torch.bincount(tr_c, minlength=3).float()
    cls_w = counts.sum() / (3.0 * counts)
    print("class weights:", cls_w.numpy().round(4).tolist())

    train_ds = MultiModalDS(tr_t, tr_a, tr_v, tr_c, tr_r, tr_mt, tr_ma, tr_mv, augment=True)
    valid_ds = MultiModalDS(va_t, va_a, va_v, va_c, va_r, va_mt, va_ma, va_mv, augment=False)
    test_ds = MultiModalDS(te_t, te_a, te_v, te_c, te_r, te_mt, te_ma, te_mv, augment=False)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    valid_loader = DataLoader(valid_ds, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

    model = MAMFN().to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"params: {n_params:,}")

    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    ce_loss = torch.nn.CrossEntropyLoss(weight=cls_w)
    huber_loss = torch.nn.HuberLoss(delta=1.0)

    rows = []
    best_f1 = -1.0
    best_state = None
    patience = 8
    bad_epochs = 0
    for epoch in range(1, EPOCHS + 1):
        model.train()
        tl = tce = tmse = 0.0
        nb = 0
        for t, a, v, mt, ma, mv, yc, yr in train_loader:
            logits, reg = model(t, a, v, mt, ma, mv)
            l_ce = ce_loss(logits, yc)
            l_mse = huber_loss(reg, yr)
            loss = LAMBDA_CE * l_ce + LAMBDA_MSE * l_mse
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tl += loss.item(); tce += l_ce.item(); tmse += l_mse.item(); nb += 1
        sched.step()
        vm = run_eval(model, valid_loader)
        rows.append({"epoch": epoch, "train_loss": round(tl/nb, 5),
                     "train_ce": round(tce/nb, 5), "train_mse": round(tmse/nb, 5),
                     **{f"val_{k}": v for k, v in vm.items()}})
        print(f"Ep {epoch:02d}/{EPOCHS} loss={tl/nb:.4f} | val acc={vm['accuracy']:.4f} "
              f"f1={vm['macro_f1']:.4f} mae={vm['mae']:.4f} pear={vm['pearson']:.4f}")
        if vm["macro_f1"] > best_f1:
            best_f1 = vm["macro_f1"]
            best_state = {k: w.clone() for k, w in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                print(f"early stop at epoch {epoch}, no improvement for {patience} epochs")
                break

    # 加载最佳模型
    model.load_state_dict(best_state)
    valid_m = run_eval(model, valid_loader)
    test_m = run_eval(model, test_loader)
    print("valid:", valid_m)
    print("test :", test_m)

    # 保存 checkpoint（state_dict + 标准化统计量）
    torch.save({
        "state_dict": best_state,
        "scaler": {"t_mean": st_mu_t, "t_std": st_sd_t,
                   "a_mean": st_mu_a, "a_std": st_sd_a,
                   "v_mean": st_mu_v, "v_std": st_sd_v},
    }, Q2_MODEL)

    metrics = {"valid": valid_m, "test": test_m}
    with open(Q2_METRICS, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    pd.DataFrame(rows).to_csv(Q2_TRAIN_LOG, index=False, encoding="utf-8-sig")

    config = {
        "seed": SEED, "batch_size": BATCH_SIZE, "epochs": EPOCHS, "lr": LR,
        "weight_decay": WEIGHT_DECAY, "d_model": D_MODEL, "t_max": T_MAX,
        "lambda_ce": LAMBDA_CE, "lambda_mse": LAMBDA_MSE,
        "aug_prob": 0.7, "miss_len_range": [5, 25],
        "class_weights": cls_w.numpy().round(4).tolist(),
        "n_params": n_params, "best_val_macro_f1": best_f1,
        "selection": "valid macro-F1",
        "label_mapping": {"0": "Negative", "1": "Neutral", "2": "Positive"},
        "mask_rule": "M_m[t]=1 if ||x_m[t]||_1>0 else 0",
        "standardize": "z-score using train valid-position statistics",
    }
    with open(Q2_CONFIG, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    print(f"done in {(time.time()-t0)/60:.1f} min -> {Q2_MODEL}")


if __name__ == "__main__":
    main()
