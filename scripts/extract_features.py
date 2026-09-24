# -*- coding: utf-8 -*-
"""
Q1：三模态（文本/语音/视觉）特征提取与时序对齐管线。

输入（只读）：
  VIDEO_ROOT 下 37 个 video_id 子文件夹，内含 {clip_id}.mp4
  LABEL_XLSX  label-100.xlsx: video_id, clip_id, text, label, annotation

输出（全部落到 Q1_DIR）：
  features_aligned_50.npz   全量对齐特征 (100, 50, 768/74/35)
  summary.csv               每样本一行汇总
  mapping.csv               每样本每对齐位置的原始素材对应关系

技术决策（与题目分析报告 §5.1 一致）：
  文本: bert-base-uncased, 词级嵌入, N 个词按句序均匀映射到 [0, T]
  语音: 16kHz 单声道, MFCC(20)+delta(20)+delta2(20)+chroma(12)+rms(1)+centroid(1)=74
  视觉: 5fps 抽帧, HSV 直方图(16h+8s+8v=32)+灰度统计(mean/std/contrast=3)=35
  对齐: 50 个位置, 位置 i 对应时间区间 [i*T/50, (i+1)*T/50]
        每个位置对落在该区间内的原始帧/词取均值; 无原始证据则零向量(填充)
"""
import os
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

import sys
import tempfile
import subprocess
import logging
import numpy as np
import pandas as pd
import cv2
import librosa
import imageio_ffmpeg
import torch
from transformers import AutoTokenizer, AutoModel

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paths import (VIDEO_ROOT, LABEL_XLSX, Q1_DIR, LOGS_DIR,
                   FEATURES_NPZ, SUMMARY_CSV, MAPPING_CSV)

ALIGN_POS = 50
SR = 16000
N_FFT = 2048
HOP = 512
N_MFCC = 20
VIS_FPS = 5
SEED = 20260923

os.makedirs(Q1_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(LOGS_DIR, "extract_features.log"), encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger("Q1")

np.random.seed(SEED)
torch.manual_seed(SEED)

FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()
log.info("ffmpeg exe: %s", FFMPEG)


# ---------------- 工具函数 ----------------
def probe_duration(video_path):
    """用 ffmpeg stderr 的 'Duration: HH:MM:SS.cc' 解析真实时长(秒)。"""
    r = subprocess.run([FFMPEG, "-i", video_path], capture_output=True, text=True,
                       encoding="utf-8", errors="ignore")
    for line in (r.stderr or "").splitlines():
        line = line.strip()
        if line.startswith("Duration:"):
            head = line.split(",")[0].replace("Duration:", "").strip()
            h, m, s = head.split(":")
            return int(h) * 3600 + int(m) * 60 + float(s)
    return None


def extract_audio_wav(video_path, wav_path):
    """用内置 ffmpeg 抽 16kHz 单声道 wav 到临时文件。"""
    cmd = [FFMPEG, "-y", "-i", video_path,
           "-ar", str(SR), "-ac", "1", "-f", "wav", wav_path]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def audio_features(wav_path):
    """返回 (frame_times[F,], feat[F,74])。"""
    y, sr = librosa.load(wav_path, sr=SR, mono=True)
    assert sr == SR
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC, n_fft=N_FFT, hop_length=HOP)
    d1 = librosa.feature.delta(mfcc)
    d2 = librosa.feature.delta(mfcc, order=2)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr, n_fft=N_FFT, hop_length=HOP)
    rms = librosa.feature.rms(y=y, hop_length=HOP)
    cent = librosa.feature.spectral_centroid(y=y, sr=sr, n_fft=N_FFT, hop_length=HOP)
    feat = np.vstack([mfcc, d1, d2, chroma, rms, cent])  # (74, F)
    times = librosa.frames_to_time(np.arange(feat.shape[1]), sr=sr, hop_length=HOP)
    return times.astype(np.float32), feat.T.astype(np.float32)


def _norm_hist(h):
    s = h.sum()
    return h / s if s > 0 else h


def vision_frame_feature(frame_bgr):
    """单帧 35 维: 16 hue + 8 sat + 8 val + 3 gray stats。"""
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    h_hist = cv2.calcHist([hsv], [0], None, [16], [0, 180]).flatten()
    s_hist = cv2.calcHist([hsv], [1], None, [8], [0, 256]).flatten()
    v_hist = cv2.calcHist([hsv], [2], None, [8], [0, 256]).flatten()
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    stats = np.array([gray.mean(), gray.std(), float(gray.max()) - float(gray.min())], dtype=np.float32)
    feat = np.concatenate([_norm_hist(h_hist), _norm_hist(s_hist), _norm_hist(v_hist), stats]).astype(np.float32)
    return feat


def vision_features(video_path):
    """按 VIS_FPS 抽帧, 返回 (frame_times[F,], feats[F,35], T_video)。"""
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    n_total = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    T_vid = (n_total / fps) if fps and fps > 0 else 0.0
    times, feats = [], []
    next_t = 0.0
    step = 1.0 / VIS_FPS
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        ts = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
        if ts >= next_t - 1e-6:
            feats.append(vision_frame_feature(frame))
            times.append(ts)
            next_t += step
    cap.release()
    if not times:
        return np.zeros((0,), np.float32), np.zeros((0, 35), np.float32), T_vid
    return np.asarray(times, np.float32), np.asarray(feats, np.float32), float(T_vid)


def pool_frames_to_pos(times, feats, T, n_pos=ALIGN_POS):
    """把帧级时序特征按 50 位置均值池化。返回 (n_pos,D), valid_len, 每位置帧索引串。"""
    D = feats.shape[1] if feats.ndim == 2 else 0
    out = np.zeros((n_pos, D), dtype=np.float32)
    ranges = []
    for i in range(n_pos):
        lo = i * T / n_pos
        hi = (i + 1) * T / n_pos
        if i == n_pos - 1:
            mask = (times >= lo) & (times <= hi + 1e-6)
        else:
            mask = (times >= lo) & (times < hi)
        idx = np.where(mask)[0]
        if len(idx) > 0:
            out[i] = feats[idx].mean(axis=0)
            ranges.append(f"{int(idx.min())}-{int(idx.max())}")
        else:
            ranges.append("")
    valid = int((np.abs(out).sum(axis=1) > 0).sum())
    return out, valid, ranges


def text_word_embeddings(model, tokenizer, text):
    """对文本做 BERT 前向, 按词聚合子word嵌入。返回 (words[list[str]], word_vecs[N,768])。"""
    enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        out = model(**enc)
    last = out.last_hidden_state[0]  # (seq,768)
    word_ids = enc.word_ids()
    words = text.split()
    from collections import defaultdict
    bucket = defaultdict(list)
    for tok_i, w in enumerate(word_ids):
        if w is None:
            continue
        bucket[w].append(tok_i)
    seen = sorted(bucket.keys())
    vecs = []
    if len(seen) != len(words):
        for w in seen:
            idxs = bucket[w]
            vecs.append(last[idxs].mean(dim=0).numpy())
        words = [words[min(w, len(words) - 1)] for w in seen]
    else:
        for w in seen:
            idxs = bucket[w]
            vecs.append(last[idxs].mean(dim=0).numpy())
    vecs = np.stack(vecs, axis=0).astype(np.float32) if vecs else np.zeros((1, 768), np.float32)
    return words, vecs


def pool_text_to_pos(word_intervals, word_vecs, T, n_pos=ALIGN_POS):
    """word_intervals: list of (w_start,w_end); 按位置池化。返回 (n_pos,768), valid, 每位置词串。"""
    D = word_vecs.shape[1]
    out = np.zeros((n_pos, D), dtype=np.float32)
    seg_texts = []
    for i in range(n_pos):
        lo = i * T / n_pos
        hi = (i + 1) * T / n_pos
        sel = []
        for wi, (ws, we) in enumerate(word_intervals):
            if ws < hi and we > lo:
                sel.append(wi)
        if sel:
            out[i] = word_vecs[sel].mean(axis=0)
            seg_texts.append(" ".join([f"<{wi}>" for wi in sel]))
        else:
            seg_texts.append("")
    valid = int((np.abs(out).sum(axis=1) > 0).sum())
    return out, valid, seg_texts


# ---------------- 主流程 ----------------
def main():
    log.info("加载标签表: %s", LABEL_XLSX)
    df = pd.read_excel(LABEL_XLSX)
    assert len(df) == 100, f"标签表应为 100 行, 实际 {len(df)}"
    log.info("样本数=%d, 列=%s", len(df), list(df.columns))

    log.info("加载 BERT bert-base-uncased (HF_ENDPOINT=%s)", os.environ["HF_ENDPOINT"])
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    model = AutoModel.from_pretrained("bert-base-uncased")
    model.eval()
    bert_dim = model.config.hidden_size
    log.info("BERT hidden_size=%d", bert_dim)

    n = len(df)
    text_arr = np.zeros((n, ALIGN_POS, bert_dim), np.float32)
    audio_arr = np.zeros((n, ALIGN_POS, 74), np.float32)
    vision_arr = np.zeros((n, ALIGN_POS, 35), np.float32)
    sample_id = np.arange(n, dtype=np.int32)
    video_ids, clip_ids = [], []
    durations = np.zeros(n, np.float32)
    t_valid = np.zeros(n, np.int32)
    a_valid = np.zeros(n, np.int32)
    v_valid = np.zeros(n, np.int32)

    mapping_rows = []
    tmpdir = tempfile.mkdtemp(prefix="q1audio_")
    log.info("临时音频目录: %s", tmpdir)

    for si, row in enumerate(df.itertuples(index=False)):
        vid = str(row.video_id)
        cid = row.clip_id
        text = str(row.text).strip()
        video_path = os.path.join(VIDEO_ROOT, vid, f"{cid}.mp4")
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"找不到视频: {video_path}")

        # 语音
        wav_path = os.path.join(tmpdir, f"{si}.wav")
        extract_audio_wav(video_path, wav_path)
        a_times, a_feat = audio_features(wav_path)
        # 视觉
        v_times, v_feat, T_vid = vision_features(video_path)
        # 以 ffmpeg 权威时长为统一时间轴
        T_probe = probe_duration(video_path)
        if T_probe and T_probe > 0:
            T = T_probe
        elif len(v_times):
            T = float(v_times[-1])
        else:
            T = float(a_times[-1]) if len(a_times) else 0.0
        durations[si] = T
        # 文本
        words, w_vecs = text_word_embeddings(model, tokenizer, text)
        nw = len(words)
        word_intervals = []
        for j in range(nw):
            ws = j * T / nw
            we = (j + 1) * T / nw if j < nw - 1 else T
            word_intervals.append((ws, we))

        # 池化到 50 位置
        t_pool, tv, t_seg = pool_text_to_pos(word_intervals, w_vecs, T)
        a_pool, av, a_rng = pool_frames_to_pos(a_times, a_feat, T)
        v_pool, vv, v_rng = pool_frames_to_pos(v_times, v_feat, T)

        text_arr[si] = t_pool
        audio_arr[si] = a_pool
        vision_arr[si] = v_pool
        t_valid[si], a_valid[si], v_valid[si] = tv, av, vv
        video_ids.append(vid)
        clip_ids.append(int(cid))

        # mapping 行
        for i in range(ALIGN_POS):
            lo = round(i * T / ALIGN_POS, 4)
            hi = round((i + 1) * T / ALIGN_POS, 4)
            seg_txt = ""
            if t_seg[i]:
                idxs = [int(x.strip("<>")) for x in t_seg[i].split()]
                seg_txt = " ".join(words[k] for k in idxs if 0 <= k < nw)
            mapping_rows.append({
                "sample_id": si,
                "video_id": vid,
                "clip_id": int(cid),
                "aligned_position": i,
                "time_start": lo,
                "time_end": hi,
                "text_segment": seg_txt,
                "audio_frame_range": a_rng[i],
                "video_frame_range": v_rng[i],
                "feature_norm_text": round(float(np.linalg.norm(t_pool[i])), 6),
                "feature_norm_audio": round(float(np.linalg.norm(a_pool[i])), 6),
                "feature_norm_vision": round(float(np.linalg.norm(v_pool[i])), 6),
            })

        if (si + 1) % 10 == 0 or si == 0:
            log.info("完成 %d/%d  vid=%s clip=%s T=%.2fs words=%d t/a/v_valid=%d/%d/%d",
                     si + 1, n, vid, cid, T, nw, tv, av, vv)

    # 保存 npz
    np.savez_compressed(
        FEATURES_NPZ,
        text=text_arr, audio=audio_arr, vision=vision_arr,
        sample_id=sample_id,
        video_id=np.array(video_ids),
        clip_id=np.array(clip_ids, dtype=np.int32),
        duration=durations,
        text_valid_len=t_valid,
        audio_valid_len=a_valid,
        vision_valid_len=v_valid,
        text_dim=np.array([bert_dim], dtype=np.int32),
        audio_dim=np.array([74], dtype=np.int32),
        vision_dim=np.array([35], dtype=np.int32),
        alignment_positions=np.array([ALIGN_POS], dtype=np.int32),
        seed=np.array([SEED], dtype=np.int32),
    )
    log.info("已保存 %s", FEATURES_NPZ)

    # summary.csv
    summary = pd.DataFrame({
        "sample_id": sample_id,
        "video_id": video_ids,
        "clip_id": clip_ids,
        "duration_sec": np.round(durations, 4),
        "text_valid_len": t_valid,
        "audio_valid_len": a_valid,
        "vision_valid_len": v_valid,
        "text_dim": bert_dim,
        "audio_dim": 74,
        "vision_dim": 35,
        "alignment_positions": ALIGN_POS,
        "padding_rule": "零向量填充; 位置[iT/50,(i+1)T/50]无原始证据则为零; 文本无逐词时间戳按句序均匀映射",
    })
    summary.to_csv(SUMMARY_CSV, index=False, encoding="utf-8-sig")
    log.info("已保存 %s", SUMMARY_CSV)

    # mapping.csv
    mdf = pd.DataFrame(mapping_rows)
    mdf.to_csv(MAPPING_CSV, index=False, encoding="utf-8-sig")
    log.info("已保存 %s  rows=%d", MAPPING_CSV, len(mdf))

    # 清理临时音频
    try:
        for f in os.listdir(tmpdir):
            os.remove(os.path.join(tmpdir, f))
        os.rmdir(tmpdir)
    except Exception as e:
        log.warning("清理临时目录失败: %s", e)

    log.info("全部完成. text=%s audio=%s vision=%s", text_arr.shape, audio_arr.shape, vision_arr.shape)


if __name__ == "__main__":
    main()
