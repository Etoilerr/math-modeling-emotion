# -*- coding: utf-8 -*-
"""
生成 results/复现清单.json：随机种子、输入文件 SHA-256、运行时、依赖版本、
关键参数、唯一复现命令。
"""
import os, sys, json, hashlib, platform, time
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paths import (PROJECT_ROOT, ALIGNED_50, ATTACH3_DIR, ATTACH4_DIR,
                   Q2_METRICS, Q3_METRICS, Q2_CONFIG, Q3_CONFIG, REPRO_MANIFEST)


def sha256(path, limit=None):
    h = hashlib.sha256()
    size = 0
    with open(path, "rb") as f:
        while True:
            b = f.read(1 << 20)
            if not b:
                break
            h.update(b); size += len(b)
            if limit and size >= limit:
                break
    return h.hexdigest(), os.path.getsize(path)


def main():
    t_start = time.time()
    # 输入文件哈希
    inputs = {}
    h, sz = sha256(str(ALIGNED_50))
    inputs[str(ALIGNED_50.relative_to(PROJECT_ROOT))] = {"sha256": h, "bytes": sz}
    for d, pat in [(ATTACH3_DIR, "附件3_*.pkl"), (ATTACH4_DIR, "*.pkl")]:
        fs = sorted(d.glob(pat))
        inputs[str(d.relative_to(PROJECT_ROOT)) + f"/({len(fs)} files)"] = \
            {"sha256_of_01": sha256(str(fs[0]))[0], "count": len(fs)}

    with open(Q2_METRICS, encoding="utf-8") as f:
        q2m = json.load(f)
    with open(Q3_METRICS, encoding="utf-8") as f:
        q3m = json.load(f)
    with open(Q2_CONFIG, encoding="utf-8") as f:
        q2c = json.load(f)
    with open(Q3_CONFIG, encoding="utf-8") as f:
        q3c = json.load(f)

    manifest = {
        "task": "E题 Q2/Q3 MA-MFN 与 A-EMN",
        "random_seeds": {"numpy": 42, "torch": 42, "python_random": 42, "PYTHONHASHSEED": 42},
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "torch": torch.__version__,
            "numpy": np.__version__,
        },
        "input_files_sha256": inputs,
        "key_params": {
            "Q2_MA-MFN": q2c, "Q3_A-EMN": q3c,
        },
        "metrics": {"Q2": q2m, "Q3": q3m},
        "reproduction_commands": [
            "cd D:\\Projects\\math_modling\\scripts",
            'python train_q2.py',
            'python predict_attachment3.py',
            'python missing_analysis.py',
            'python train_q3.py',
            'python explain_q3.py',
            'python make_figures.py',
            'python repro_manifest.py',
        ],
        "python_interpreter": r"D:\Program Files\computer language\python.exe",
        "notes": "标准化用 train 有效位置统计量；掩码 M_m[t]=1 if ||x_m[t]||_1>0；"
                 "模型存 state_dict+scaler；CPU 训练。",
    }
    manifest["manifest_build_seconds"] = round(time.time() - t_start, 2)
    with open(REPRO_MANIFEST, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    print("saved ->", REPRO_MANIFEST)


if __name__ == "__main__":
    main()
