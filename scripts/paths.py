# -*- coding: utf-8 -*-
"""
E 题项目统一路径配置（唯一来源）。
所有脚本只从本文件 import 路径，不再写死绝对路径。
"""
from pathlib import Path

# ===== 锚点 =====
SCRIPTS_DIR = Path(__file__).resolve().parent          # ...\math_modling\scripts
PROJECT_ROOT = SCRIPTS_DIR.parent                      # ...\math_modling

# ===== 原始数据 =====
DATA_ROOT = PROJECT_ROOT / "E题数据"
ATTACH1_DIR = DATA_ROOT / "附件1-数据集原始多模态样本"
VIDEO_ROOT = ATTACH1_DIR / "MOSEI数据集部分原始视频-100条"
LABEL_XLSX = VIDEO_ROOT / "label-100.xlsx"

# ===== 特征数据（附件2/3/4）=====
ATTACH2_DIR = DATA_ROOT / "附件2-数据集特征文件"
ALIGNED_50 = ATTACH2_DIR / "aligned_50.pkl"
UNALIGNED_50 = ATTACH2_DIR / "unaligned_50.pkl"
LABEL_XLSX_2 = ATTACH2_DIR / "label.xlsx"
ATTACH3_DIR = DATA_ROOT / "附件3-模态缺失特征样本" / "对齐版本"
ATTACH4_DIR = (DATA_ROOT / "附件4-可解释专项视频样本与特征文件"
               / "附件4-可解释专项视频样本与特征文件" / "对齐版本")
ATTACH4_VIDEOS = ATTACH4_DIR / "videos"

# ===== 输出目录 =====
Q1_DIR = PROJECT_ROOT / "Q1_features"
Q2_DIR = PROJECT_ROOT / "Q2_results"
Q3_DIR = PROJECT_ROOT / "Q3_results"
MODELS_DIR = PROJECT_ROOT / "models"
LOGS_DIR = PROJECT_ROOT / "logs"
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = PROJECT_ROOT / "figures"

# ===== 具体文件 =====
FEATURES_NPZ = Q1_DIR / "features_aligned_50.npz"
SUMMARY_CSV = Q1_DIR / "summary.csv"
MAPPING_CSV = Q1_DIR / "mapping.csv"
Q2_MODEL = MODELS_DIR / "q2_robust_model.pt"
Q2_METRICS = Q2_DIR / "metrics.json"
Q2_CONFIG = Q2_DIR / "best_config.json"
Q2_TRAIN_LOG = Q2_DIR / "training_log.csv"
Q2_MISSING_ANALYSIS = Q2_DIR / "missing_analysis.csv"
ATTACH3_PRED = Q2_DIR / "attachment3_predictions.csv"
Q3_MODEL = MODELS_DIR / "q3_explainable_model.pt"
Q3_METRICS = Q3_DIR / "metrics.json"
Q3_CONFIG = Q3_DIR / "best_config.json"
ATTACH4_PRED = Q3_DIR / "attachment4_predictions_explanations.csv"
ATTACH4_FRAGMENT = Q3_DIR / "attachment4_fragment_importance.csv"
Q3_OCCLUSION = Q3_DIR / "occlusion_validation.csv"
REPRO_MANIFEST = RESULTS_DIR / "复现清单.json"

# 确保目录存在
for d in [Q1_DIR, Q2_DIR, Q3_DIR, MODELS_DIR, LOGS_DIR, RESULTS_DIR, FIGURES_DIR]:
    d.mkdir(parents=True, exist_ok=True)


def check(verbose=True):
    """自检：打印关键路径存在性。返回缺失项名称列表。"""
    keys = [
        ("VIDEO_ROOT", VIDEO_ROOT), ("LABEL_XLSX", LABEL_XLSX),
        ("ALIGNED_50", ALIGNED_50), ("ATTACH3_DIR", ATTACH3_DIR),
        ("ATTACH4_DIR", ATTACH4_DIR),
    ]
    missing = []
    for name, p in keys:
        ok = p.exists()
        if verbose:
            print(("[OK]   " if ok else "[MISS] ") + f"{name} = {p}")
        if not ok:
            missing.append(name)
    return missing


if __name__ == "__main__":
    m = check()
    if m:
        print("\n缺失项:", m)
        raise SystemExit(1)
    print("\n全部路径就绪。")
