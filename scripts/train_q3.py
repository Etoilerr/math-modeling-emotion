# -*- coding: utf-8 -*-
"""
Q3 训练：A-EMN（注意力可解释多模态网络）。
与 Q2 同数据/同损失/同优化器，但无缺失增强（三模态完整），
增加片段级 β 与模态级 α 注意力，注意力加权池化替代均值池化。
验证集 macro-F1 选模，保存 state_dict + 标准化统计量。
"""
import os, sys, json, pickle, time
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paths import (ALIGNED_50, Q3_DIR, Q3_MODEL, Q3_METRICS, Q3_CONFIG, Q2_TRAIN_LOG)
from model_common import (set_seed, build_mask, fit_scaler, apply_scaler,
                          AEMN, compute_metrics,
                          SEED, BATCH_SIZE, EPOCHS, LR, WEIGHT_DECAY,
                          LAMBDA_CE, LAMBDA_MSE, D_MODEL, T_MAX)

DEVICE = torch.device("cpu")
set_seed(SEED)


def to_tensors(d):
    return (torch.tensor(d["text"], dtype=torch.float32),
            torch.tensor(d["audio"], dtype=torch.float32),
            torch.tensor(d["vision"], dtype=torch.float32),
            torch.tensor(d["classification_labels"], dtype=torch.long),
            torch.tensor(d["regression_labels"], dtype=torch.float32))


class DS(Dataset):
    def __init__(self, t, a, v, c, r, mt, ma, mv):
        self.t, self.a, self.v, self.c, self.r = t, a, v, c, r
        self.mt, self.ma, self.mv = mt, ma, mv

    def __len__(self):
        return len(self.c)

    def __getitem__(self, i):
        return self.t[i], self.a[i], self.v[i], self.mt[i], self.ma[i], self.mv[i], self.c[i], self.r[i]


@torch.no_grad()
def run_eval(model, loader):
    model.eval()
    pc, tc, pr, tr = [], [], [], []
    for t, a, v, mt, ma, mv, yc, yr in loader:
        logits, reg, alpha, beta = model(t, a, v, mt, ma, mv)
        pc.extend(logits.argmax(-1).numpy().tolist())
        tc.extend(yc.numpy().tolist())
        pr.extend(reg.numpy().tolist())
        tr.extend(yr.numpy().tolist())
    return compute_metrics(tc, pc, tr, pr)


def main():
    t0 = time.time()
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
    tr_t, va_t, te_t = apply_scaler(tr_t, st_mu_t, st_sd_t), apply_scaler(va_t, st_mu_t, st_sd_t), apply_scaler(te_t, st_mu_t, st_sd_t)
    tr_a, va_a, te_a = apply_scaler(tr_a, st_mu_a, st_sd_a), apply_scaler(va_a, st_mu_a, st_sd_a), apply_scaler(te_a, st_mu_a, st_sd_a)
    tr_v, va_v, te_v = apply_scaler(tr_v, st_mu_v, st_sd_v), apply_scaler(va_v, st_mu_v, st_sd_v), apply_scaler(te_v, st_mu_v, st_sd_v)

    counts = torch.bincount(tr_c, minlength=3).float()
    cls_w = counts.sum() / (3.0 * counts)

    train_loader = DataLoader(DS(tr_t, tr_a, tr_v, tr_c, tr_r, tr_mt, tr_ma, tr_mv), batch_size=BATCH_SIZE, shuffle=True)
    valid_loader = DataLoader(DS(va_t, va_a, va_v, va_c, va_r, va_mt, va_ma, va_mv), batch_size=BATCH_SIZE)
    test_loader = DataLoader(DS(te_t, te_a, te_v, te_c, te_r, te_mt, te_ma, te_mv), batch_size=BATCH_SIZE)

    model = AEMN().to(DEVICE)
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
        tl = 0.0; nb = 0
        for t, a, v, mt, ma, mv, yc, yr in train_loader:
            logits, reg, _, _ = model(t, a, v, mt, ma, mv)
            loss = LAMBDA_CE * ce_loss(logits, yc) + LAMBDA_MSE * huber_loss(reg, yr)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tl += loss.item(); nb += 1
        sched.step()
        vm = run_eval(model, valid_loader)
        rows.append({"epoch": epoch, "train_loss": round(tl/nb, 5),
                     **{f"val_{k}": v for k, v in vm.items()}})
        print(f"Ep {epoch:02d}/{EPOCHS} loss={tl/nb:.4f} | f1={vm['macro_f1']:.4f} acc={vm['accuracy']:.4f}")
        if vm["macro_f1"] > best_f1:
            best_f1 = vm["macro_f1"]
            best_state = {k: w.clone() for k, w in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                print(f"early stop at epoch {epoch}, no improvement for {patience} epochs")
                break

    model.load_state_dict(best_state)
    valid_m = run_eval(model, valid_loader)
    test_m = run_eval(model, test_loader)
    print("valid:", valid_m); print("test :", test_m)

    torch.save({"state_dict": best_state,
                "scaler": {"t_mean": st_mu_t, "t_std": st_sd_t,
                           "a_mean": st_mu_a, "a_std": st_sd_a,
                           "v_mean": st_mu_v, "v_std": st_sd_v}}, Q3_MODEL)

    with open(Q3_METRICS, "w", encoding="utf-8") as f:
        json.dump({"valid": valid_m, "test": test_m}, f, indent=2, ensure_ascii=False)
    pd.DataFrame(rows).to_csv(str(Q3_DIR / "training_log.csv"), index=False, encoding="utf-8-sig")
    config = {"seed": SEED, "batch_size": BATCH_SIZE, "epochs": EPOCHS, "lr": LR,
              "weight_decay": WEIGHT_DECAY, "d_model": D_MODEL, "t_max": T_MAX,
              "lambda_ce": LAMBDA_CE, "lambda_mse": LAMBDA_MSE,
              "missing_augmentation": False, "n_params": n_params,
              "best_val_macro_f1": best_f1, "selection": "valid macro-F1",
              "label_mapping": {"0": "Negative", "1": "Neutral", "2": "Positive"}}
    with open(Q3_CONFIG, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    print(f"done in {(time.time()-t0)/60:.1f} min -> {Q3_MODEL}")


if __name__ == "__main__":
    main()
