# -*- coding: utf-8 -*-
"""
Q3 解释：
1) 验证集遮挡验证（top-K vs bottom-K，K=5/10/15）
2) 附件4（20 条完整样本）预测 + 解释（α_m, β_{m,t}, 主要参考模态, top-K 片段）
3) 提取附件4 视觉关键帧 -> Q3_results/keyframes/
输出: occlusion_validation.csv, attachment4_predictions_explanations.csv,
      attachment4_fragment_importance.csv
"""
import os, sys, pickle, glob, json
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paths import (ALIGNED_50, ATTACH4_DIR, ATTACH4_VIDEOS, Q3_DIR, Q3_MODEL,
                   Q3_OCCLUSION, ATTACH4_PRED, ATTACH4_FRAGMENT)
from model_common import (set_seed, build_mask, apply_scaler, AEMN, LABEL_NAMES,
                          MOD_NAMES, T_MAX)

DEVICE = torch.device("cpu")
set_seed(42)
TOPK = 5


def load_model():
    ckpt = torch.load(Q3_MODEL, map_location=DEVICE, weights_only=True)
    # 学生模型 d=32（知识蒸馏）
    m = AEMN(d_model=32).to(DEVICE); m.load_state_dict(ckpt["state_dict"]); m.eval()
    return m, ckpt["scaler"]


def predict_one(model, t, a, v, mt, ma, mv):
    with torch.no_grad():
        logits, reg, alpha, (bt, ba, bv) = model(t, a, v, mt, ma, mv)
    p = F.softmax(logits, dim=-1)
    return logits, reg, alpha, (bt, ba, bv), p


@torch.no_grad()
def occlusion_validation(model, sc):
    """valid 集 top-K vs bottom-K 遮挡。"""
    with open(ALIGNED_50, "rb") as f:
        d = pickle.load(f)["valid"]
    t = apply_scaler(torch.tensor(d["text"], dtype=torch.float32), sc["t_mean"], sc["t_std"])
    a = apply_scaler(torch.tensor(d["audio"], dtype=torch.float32), sc["a_mean"], sc["a_std"])
    v = apply_scaler(torch.tensor(d["vision"], dtype=torch.float32), sc["v_mean"], sc["v_std"])
    mt, ma, mv = build_mask(t), build_mask(a), build_mask(v)
    c = d["classification_labels"].astype(int)
    n = len(c)

    rows = []
    from scipy.stats import ttest_rel
    for K in [5, 10, 15]:
        dt_list, db_list = [], []
        for i in range(n):
            tt, aa, vv = t[i:i+1], a[i:i+1], v[i:i+1]
            mtt, maa, mvv = mt[i:i+1], ma[i:i+1], mv[i:i+1]
            _, reg0, alpha, (bt, ba, bv), p0 = predict_one(model, tt, aa, vv, mtt, maa, mvv)
            # 重要性 = α_m * β_{m,t}，仅有效位置
            betas = torch.stack([bt[0], ba[0], bv[0]], 0)          # (3,T)
            msk = torch.stack([mtt[0], maa[0], mvv[0]], 0)         # (3,T)
            imp = alpha[0].unsqueeze(-1) * betas * msk            # (3,T)
            imp_flat = imp.reshape(-1)
            valid_idx = torch.nonzero(imp_flat > 0).squeeze(-1)
            if len(valid_idx) < 2 * K:
                continue
            order = valid_idx[torch.argsort(imp_flat[valid_idx], descending=True)]
            top = order[:K]; bottom = order[-K:]

            def occlude(idxs):
                tt2, aa2, vv2 = tt.clone(), aa.clone(), vv.clone()
                m2t, m2a, m2v = mtt.clone(), maa.clone(), mvv.clone()
                for idx in idxs.tolist():
                    mi, ti = idx // T_MAX, idx % T_MAX
                    if mi == 0: tt2[0, ti] = 0; m2t[0, ti] = 0
                    elif mi == 1: aa2[0, ti] = 0; m2a[0, ti] = 0
                    else: vv2[0, ti] = 0; m2v[0, ti] = 0
                _, reg1, _, _, p1 = predict_one(model, tt2, aa2, vv2, m2t, m2a, m2v)
                kl = F.kl_div(torch.log(p1 + 1e-12), p0, reduction="sum").item()
                return kl + abs(reg1.item() - reg0.item())

            dt_list.append(occlude(top))
            db_list.append(occlude(bottom))
        dt = np.array(dt_list); db = np.array(db_list)
        frac = float((dt > db).mean())
        stat, pval = ttest_rel(dt, db)
        rows.append({"K": K, "n": len(dt),
                     "mean_delta_top": round(float(dt.mean()), 5),
                     "mean_delta_bottom": round(float(db.mean()), 5),
                     "frac_top_gt_bottom": round(frac, 4),
                     "paired_t": round(float(stat), 4),
                     "p_value": float(pval)})
        print(f"K={K}: n={len(dt)} Δtop={dt.mean():.4f} Δbottom={db.mean():.4f} frac={frac:.3f}")
    df = pd.DataFrame(rows)
    df.to_csv(Q3_OCCLUSION, index=False, encoding="utf-8-sig")
    print("occlusion_validation.csv saved ->", Q3_OCCLUSION)
    return df


def extract_keyframe(video_path, pos, out_path):
    import cv2
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total = cap.get(cv2.CAP_PROP_FRAME_COUNT) or (T_MAX * fps)
    t_sec = (pos + 0.5) / T_MAX * (total / fps)
    cap.set(cv2.CAP_PROP_POS_MSEC, t_sec * 1000)
    ok, frame = cap.read()
    if not ok:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int((pos + 0.5) / T_MAX * total))
        ok, frame = cap.read()
    if ok:
        cv2.imwrite(out_path, frame)
    cap.release()
    return ok


def explain_attach4(model, sc):
    files = sorted(glob.glob(os.path.join(ATTACH4_DIR, "*.pkl")))
    assert len(files) == 20, f"expect 20, got {len(files)}"
    kf_dir = Q3_DIR / "keyframes"; kf_dir.mkdir(exist_ok=True)
    rows, frag_rows = [], []
    for fp in files:
        sid = os.path.basename(fp).replace(".pkl", "")           # 01
        d = pickle.load(open(fp, "rb"))
        t = apply_scaler(torch.tensor(d["text"], dtype=torch.float32).unsqueeze(0), sc["t_mean"], sc["t_std"])
        a = apply_scaler(torch.tensor(d["audio"], dtype=torch.float32).unsqueeze(0), sc["a_mean"], sc["a_std"])
        v = apply_scaler(torch.tensor(d["vision"], dtype=torch.float32).unsqueeze(0), sc["v_mean"], sc["v_std"])
        mt, ma, mv = build_mask(t), build_mask(a), build_mask(v)

        logits, reg, alpha, (bt, ba, bv), p = predict_one(model, t, a, v, mt, ma, mv)
        pred = int(logits.argmax(-1).item())
        al = alpha[0].numpy()
        main_mod = MOD_NAMES[int(al.argmax())]
        betas = {"text": bt[0], "audio": ba[0], "vision": bv[0]}
        masks = {"text": mt[0], "audio": ma[0], "vision": mv[0]}
        # 全局重要性 = α_m * β_{m,t}
        imp = {}
        for mi, mname in enumerate(MOD_NAMES):
            imp[mname] = (float(al[mi]) * betas[mname] * masks[mname]).numpy()
        imp_stack = np.stack([imp[m] for m in MOD_NAMES], 0).reshape(-1)
        valid_pos = np.nonzero(imp_stack > 0)[0]
        order = valid_pos[np.argsort(-imp_stack[valid_pos])][:TOPK]
        top_pos, top_mod = [], []
        for idx in order.tolist():
            mi, ti = idx // T_MAX, idx % T_MAX
            top_mod.append(MOD_NAMES[mi]); top_pos.append(int(ti))
            frag_rows.append({"sample_id": sid, "modality": MOD_NAMES[mi],
                              "position": int(ti), "importance": round(float(imp_stack[idx]), 6),
                              "alpha": round(float(al[mi]), 4),
                              "beta": round(float(betas[MOD_NAMES[mi]][ti]), 6)})
        # 长表：全部有效位置
        for mi, mname in enumerate(MOD_NAMES):
            for ti in range(T_MAX):
                if float(masks[mname][ti]) > 0:
                    frag_rows.append({"sample_id": sid, "modality": mname,
                                      "position": ti, "importance": round(float(imp[mname][ti]), 6),
                                      "alpha": round(float(al[mi]), 4),
                                      "beta": round(float(betas[mname][ti]), 6)})
        # 关键帧：取视觉模态 top-3 位置
        vids = os.path.join(ATTACH4_VIDEOS, f"{sid}.mp4")
        kf_count = 0
        if os.path.exists(vids):
            vimp = imp["vision"]
            vorder = np.argsort(-vimp)[:3]
            for k, ti in enumerate(vorder.tolist()):
                if vimp[ti] <= 0:
                    continue
                out = str(kf_dir / f"{sid}_top{k+1}_pos{ti:02d}.jpg")
                if extract_keyframe(vids, ti, out):
                    kf_count += 1
        rows.append({
            "sample_id": sid, "video_id": f"A4_{sid}",
            "polarity_pred": pred, "polarity_label": LABEL_NAMES[pred],
            "intensity_pred": round(float(reg.item()), 4),
            "alpha_text": round(float(al[0]), 4),
            "alpha_audio": round(float(al[1]), 4),
            "alpha_vision": round(float(al[2]), 4),
            "main_modality": main_mod,
            "topK_positions": ",".join(map(str, top_pos)),
            "topK_modalities": ",".join(top_mod),
            "keyframes_extracted": kf_count,
        })
        print(f"  {sid}: {LABEL_NAMES[pred]:9s} inten={float(reg.item()):+.3f} "
              f"main={main_mod} α={al.round(3).tolist()} kf={kf_count}")

    pd.DataFrame(rows).to_csv(ATTACH4_PRED, index=False, encoding="utf-8-sig")
    pd.DataFrame(frag_rows).to_csv(ATTACH4_FRAGMENT, index=False, encoding="utf-8-sig")
    print(f"saved {len(rows)} pred rows, {len(frag_rows)} fragment rows")


def main():
    model, sc = load_model()
    print("== occlusion validation ==")
    occlusion_validation(model, sc)
    print("== attach4 explain ==")
    explain_attach4(model, sc)


if __name__ == "__main__":
    main()
