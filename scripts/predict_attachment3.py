# -*- coding: utf-8 -*-
"""
附件3 推理（30 条无标签，模态局部连续全零缺失）。
- 加载 Q2 最佳模型 + train 标准化统计量
- 附件3 仅含 text_bert/audio/vision：用 bert-base-uncased 将 text_bert 编码为 text(1,50,768)
- 构造掩码 -> 标准化 -> 预测三分类 + 连续强度
- 输出 attachment3_predictions.csv
"""
import os, sys, pickle, glob
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paths import ATTACH3_DIR, Q2_DIR, Q2_MODEL, ATTACH3_PRED
from model_common import (set_seed, build_mask, apply_scaler, MAMFN,
                          LABEL_NAMES, T_MAX)

DEVICE = torch.device("cpu")
set_seed(42)


def main():
    from transformers import BertModel
    ckpt = torch.load(Q2_MODEL, map_location=DEVICE, weights_only=True)
    model = MAMFN().to(DEVICE)
    model.load_state_dict(ckpt["state_dict"]); model.eval()
    sc = ckpt["scaler"]
    print("Q2 model loaded.")

    print("Loading BERT ...")
    bert = BertModel.from_pretrained("bert-base-uncased").to(DEVICE)
    bert.eval(); bert.requires_grad_(False)

    files = sorted(glob.glob(os.path.join(ATTACH3_DIR, "附件3_*.pkl")))
    assert len(files) == 30, f"expect 30 attach3 files, got {len(files)}"
    rows = []
    for fp in files:
        sid = os.path.basename(fp).replace(".pkl", "")          # 附件3_01
        with open(fp, "rb") as f:
            d = pickle.load(f)["test"]
        tb = d["text_bert"]                                     # (1,3,50)
        ids = torch.tensor(tb[:, 0:1, :], dtype=torch.long).squeeze(1)
        am = torch.tensor(tb[:, 1:2, :], dtype=torch.long).squeeze(1)
        tt = torch.tensor(tb[:, 2:3, :], dtype=torch.long).squeeze(1)
        with torch.no_grad():
            text = bert(input_ids=ids, attention_mask=am, token_type=tt).last_hidden_state
        audio = torch.tensor(d["audio"], dtype=torch.float32)   # (1,50,74)
        vision = torch.tensor(d["vision"], dtype=torch.float32) # (1,50,35)

        # 坑7修正：text 掩码直接用 text_bert 自带的 attention_mask，
        # 不能用 build_mask(BERT输出)——BERT padding 位置输出非零会被误判为有效
        mt = am.float()
        ma = build_mask(audio)
        mv = build_mask(vision)
        text = apply_scaler(text, sc["t_mean"], sc["t_std"])
        audio = apply_scaler(audio, sc["a_mean"], sc["a_std"])
        vision = apply_scaler(vision, sc["v_mean"], sc["v_std"])

        with torch.no_grad():
            logits, reg = model(text, audio, vision, mt, ma, mv)
            pred = int(logits.argmax(-1).item())
            inten = float(reg.item())
        rows.append({
            "sample_id": sid,
            "video_id": sid.replace("附件3_", "A3_"),
            "polarity_pred": pred,
            "polarity_label": LABEL_NAMES[pred],
            "intensity_pred": round(inten, 4),
        })
        print(f"  {sid}: {LABEL_NAMES[pred]:9s} intensity={inten:+.3f}")

    df = pd.DataFrame(rows)
    df.to_csv(ATTACH3_PRED, index=False, encoding="utf-8-sig")
    print(f"\nsaved {len(df)} rows -> {ATTACH3_PRED}")


if __name__ == "__main__":
    main()
