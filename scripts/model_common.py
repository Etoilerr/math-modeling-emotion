# -*- coding: utf-8 -*-
"""
Q2/Q3 共享组件：随机种子、数据掩码与标准化、MA-MFN / A-EMN 模型、评估指标。
所有脚本从这里 import，保证训练/预测/消融/解释口径完全一致。
"""
import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ============ 全局超参（严格按题目分析报告 §5.2/§5.3）============
SEED = 42
T_MAX = 50
D_MODEL = 64
DROPOUT = 0.3
BATCH_SIZE = 64
EPOCHS = 30
LR = 1e-3
WEIGHT_DECAY = 1e-4
LAMBDA_CE = 1.0
LAMBDA_MSE = 0.5
AUG_PROB = 0.7
MISS_LEN_MIN = 5      # 连续缺失区间长度下限（位置数）= floor(0.1*50)
MISS_LEN_MAX = 25     # 连续缺失区间长度上限（位置数）= floor(0.5*50)
N_CLASSES = 3
LABEL_NAMES = {0: "Negative", 1: "Neutral", 2: "Positive"}
MOD_NAMES = ["text", "audio", "vision"]


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def build_mask(x):
    """x: (..., T, D) float -> (..., T) float; M_m[t]=1 if ||x[t]||_1>0 else 0"""
    return (x.abs().sum(dim=-1) > 0).float()


def fit_scaler(x, mask):
    """用 train 有效位置拟合逐特征 z-score。x:(N,T,D) mask:(N,T) -> mean/std (D,)"""
    m = mask.unsqueeze(-1).bool().expand_as(x)
    vals = x[m].reshape(-1, x.shape[-1])
    mean = vals.mean(dim=0)
    std = vals.std(dim=0).clamp(min=1e-6)
    return mean, std


def apply_scaler(x, mean, std):
    return (x - mean) / std


# ============ 自适应激活函数 ============
class AdaptiveSwish(nn.Module):
    """可学习 Swish：y = x · sigmoid(β · x)。
    β 逐通道可学习，初始 1.0（标准 Swish）。
    β→0 近似线性，β→∞ 近似 ReLU，模型自适应学每个通道该多强。"""
    def __init__(self, dim):
        super().__init__()
        self.beta = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return x * torch.sigmoid(self.beta * x)


# ============ 单模态编码器 ============
class ModalEncoder(nn.Module):
    """线性投影 -> LayerNorm -> GELU -> 1层 TransformerEncoderLayer(4头) residual。
    输出序列 (B,T,d)，池化由外部完成。
    消融实验：AdaptiveSwish 类保留在上方，主模型用固定 GELU（小数据集上泛化更稳）。"""
    def __init__(self, in_dim, d_model=D_MODEL, dropout=DROPOUT, nhead=4):
        super().__init__()
        self.proj = nn.Linear(in_dim, d_model)
        self.norm = nn.LayerNorm(d_model)
        # 可学习位置编码
        self.pos = nn.Parameter(torch.randn(1, T_MAX, d_model) * 0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model*4,
            dropout=dropout, activation="gelu", batch_first=True, norm_first=True)
        self.transformer = encoder_layer
        self.out_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        h = self.proj(x)            # (B,T,d)
        h = self.norm(h)
        h = F.gelu(h)
        h = h + self.pos[:, :h.size(1), :]
        # Transformer 需要 src_key_padding_mask: True 表示该位置被屏蔽
        if mask is not None:
            key_padding_mask = (mask < 0.5)  # (B,T)
            # 若某样本全部位置都被屏蔽，跳过注意力（输出 0，避免 NaN）
            all_masked = key_padding_mask.all(dim=-1)  # (B,)
            if all_masked.any():
                # 先对非全屏蔽样本跑 Transformer
                h_out = torch.zeros_like(h)
                if (~all_masked).any():
                    h_enc = self.transformer(h[~all_masked],
                                             src_key_padding_mask=key_padding_mask[~all_masked])
                    h_out[~all_masked] = h_enc
                h = h_out
            else:
                h = self.transformer(h, src_key_padding_mask=key_padding_mask)
        else:
            h = self.transformer(h)
        h = self.out_norm(h)
        h = self.dropout(h)
        return h                    # (B,T,d)


def masked_mean_pool(h, mask):
    """掩码感知均值池化。h:(B,T,d) mask:(B,T) -> (B,d)"""
    m = mask.unsqueeze(-1)
    s = (h * m).sum(dim=1)
    cnt = m.sum(dim=1).clamp(min=1.0)
    return s / cnt


# ============ Q2: MA-MFN ============
class MAMFN(nn.Module):
    """掩码感知多模态融合网络：掩码均值池化 + 门控跨模态融合 + 双任务头。"""
    def __init__(self, d_text=768, d_audio=74, d_vision=35,
                 d_model=D_MODEL, n_classes=N_CLASSES, dropout=DROPOUT):
        super().__init__()
        self.enc_t = ModalEncoder(d_text, d_model, dropout)
        self.enc_a = ModalEncoder(d_audio, d_model, dropout)
        self.enc_v = ModalEncoder(d_vision, d_model, dropout)
        # 门控：输入三模态池化拼接 (3d) -> d
        self.gate_t = nn.Linear(3 * d_model, d_model)
        self.gate_a = nn.Linear(3 * d_model, d_model)
        self.gate_v = nn.Linear(3 * d_model, d_model)
        # 融合
        self.fuse = nn.Linear(3 * d_model, d_model)
        self.fuse_norm = nn.LayerNorm(d_model)
        # 双任务头
        self.cls_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_model // 2, n_classes))
        self.reg_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1))

    def _encode_pool(self, t, a, v, mt, ma, mv):
        zt = masked_mean_pool(self.enc_t(t, mt), mt)
        za = masked_mean_pool(self.enc_a(a, ma), ma)
        zv = masked_mean_pool(self.enc_v(v, mv), mv)
        return zt, za, zv

    def _fuse_heads(self, zt, za, zv):
        cat = torch.cat([zt, za, zv], dim=-1)           # (B,3d)
        gt = torch.sigmoid(self.gate_t(cat))
        ga = torch.sigmoid(self.gate_a(cat))
        gv = torch.sigmoid(self.gate_v(cat))
        fused_in = torch.cat([gt * zt, ga * za, gv * zv], dim=-1)
        h = self.fuse_norm(F.gelu(self.fuse(fused_in)))
        logits = self.cls_head(h)
        reg = 3.0 * torch.tanh(self.reg_head(h).squeeze(-1))  # 约束到 [-3,3]
        return logits, reg

    def forward(self, t, a, v, mt, ma, mv):
        zt, za, zv = self._encode_pool(t, a, v, mt, ma, mv)
        return self._fuse_heads(zt, za, zv)


# ============ TACFN 式自适应跨模态块（序列版）============
class AdaptiveCrossModalBlock(nn.Module):
    """TACFN 自适应跨模态块在 50 位置序列上的移植：
    1) 源/目标模态线性投影相加 -> tanh -> softmax 生成权重向量；
    2) 权重向量调制目标模态实现增强，残差保留目标模态原始信息；
    3) 源模态缺失位置置零后其投影≈0，权重自动退化为仅依赖目标模态（缺失鲁棒）。
    对应论文公式：X_q=tanh(W_v X_V+b_v+W_a X_A)，X_o=(softmax(X_q)⊗X_V)⊕X_V。
    """
    def __init__(self, d_model, k=None, dropout=DROPOUT):
        super().__init__()
        k = k or max(8, d_model // 4)
        self.W_tgt = nn.Linear(d_model, k)
        self.W_src = nn.Linear(d_model, k)
        self.b = nn.Parameter(torch.zeros(k))
        self.W_out = nn.Linear(k, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, X_tgt, X_src, mask_src=None):
        # 缺失位置源模态贡献置零（防止污染权重生成）
        if mask_src is not None:
            X_src = X_src * mask_src.unsqueeze(-1)
        q = torch.tanh(self.W_tgt(X_tgt) + self.W_src(X_src) + self.b)  # (B,T,k)
        w = F.softmax(q, dim=-1)                                        # 权重向量
        out = self.norm(self.dropout(self.W_out(w) * X_tgt) + X_tgt)    # 增强 + 残差
        return out


class MAMFN_ACB(nn.Module):
    """TACFN 版 Q2 模型：模态内自注意力选择(ModalEncoder) + 双向自适应跨模态块 + 双任务头。
    融合层用 6 个有向 ACB（t←a, t←v, a←t, a←v, v←t, v←a）替代原门控融合，
    池化后在模态内整合原始/增强表示，再接原双任务头（与 MA-MFN 完全同口径）。"""
    def __init__(self, d_text=768, d_audio=74, d_vision=35,
                 d_model=D_MODEL, n_classes=N_CLASSES, dropout=DROPOUT, acb_k=None):
        super().__init__()
        self.enc_t = ModalEncoder(d_text, d_model, dropout)
        self.enc_a = ModalEncoder(d_audio, d_model, dropout)
        self.enc_v = ModalEncoder(d_vision, d_model, dropout)
        # 6 个有向自适应跨模态块
        self.acb_t_a = AdaptiveCrossModalBlock(d_model, acb_k, dropout)
        self.acb_t_v = AdaptiveCrossModalBlock(d_model, acb_k, dropout)
        self.acb_a_t = AdaptiveCrossModalBlock(d_model, acb_k, dropout)
        self.acb_a_v = AdaptiveCrossModalBlock(d_model, acb_k, dropout)
        self.acb_v_t = AdaptiveCrossModalBlock(d_model, acb_k, dropout)
        self.acb_v_a = AdaptiveCrossModalBlock(d_model, acb_k, dropout)
        # 模态内整合：原始 + 两个增强版本 -> d
        self.comb_t = nn.Linear(3 * d_model, d_model)
        self.comb_a = nn.Linear(3 * d_model, d_model)
        self.comb_v = nn.Linear(3 * d_model, d_model)
        # 融合 + 双任务头（同 MA-MFN）
        self.fuse = nn.Linear(3 * d_model, d_model)
        self.fuse_norm = nn.LayerNorm(d_model)
        self.cls_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_model // 2, n_classes))
        self.reg_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1))

    def forward(self, t, a, v, mt, ma, mv):
        zt = self.enc_t(t, mt) * mt.unsqueeze(-1)   # 缺失位置置零
        za = self.enc_a(a, ma) * ma.unsqueeze(-1)
        zv = self.enc_v(v, mv) * mv.unsqueeze(-1)
        # 双向增强 + 掩码感知池化
        zt_p = masked_mean_pool(zt, mt)
        za_p = masked_mean_pool(za, ma)
        zv_p = masked_mean_pool(zv, mv)
        zt_ra = masked_mean_pool(self.acb_t_a(zt, za, ma), mt)
        zt_rv = masked_mean_pool(self.acb_t_v(zt, zv, mv), mt)
        za_rt = masked_mean_pool(self.acb_a_t(za, zt, mt), ma)
        za_rv = masked_mean_pool(self.acb_a_v(za, zv, mv), ma)
        zv_rt = masked_mean_pool(self.acb_v_t(zv, zt, mt), mv)
        zv_ra = masked_mean_pool(self.acb_v_a(zv, za, ma), mv)
        zt_f = self.comb_t(torch.cat([zt_p, zt_ra, zt_rv], dim=-1))
        za_f = self.comb_a(torch.cat([za_p, za_rt, za_rv], dim=-1))
        zv_f = self.comb_v(torch.cat([zv_p, zv_rt, zv_ra], dim=-1))
        # 融合 + 双任务头（无额外门控，ACB 已承担自适应加权）
        h = self.fuse_norm(F.gelu(self.fuse(torch.cat([zt_f, za_f, zv_f], dim=-1))))
        logits = self.cls_head(h)
        reg = 3.0 * torch.tanh(self.reg_head(h).squeeze(-1))
        return logits, reg


# ============ Q3: A-EMN ============
class AEMN(nn.Module):
    """注意力可解释多模态网络：片段级注意力 β 加权池化 + 模态级注意力 α + 门控融合。"""
    def __init__(self, d_text=768, d_audio=74, d_vision=35,
                 d_model=D_MODEL, n_classes=N_CLASSES, dropout=DROPOUT):
        super().__init__()
        self.enc_t = ModalEncoder(d_text, d_model, dropout)
        self.enc_a = ModalEncoder(d_audio, d_model, dropout)
        self.enc_v = ModalEncoder(d_vision, d_model, dropout)
        # 片段级注意力打分器（每模态）
        self.score_t = nn.Linear(d_model, 1)
        self.score_a = nn.Linear(d_model, 1)
        self.score_v = nn.Linear(d_model, 1)
        # 模态级注意力打分器
        self.q = nn.Linear(d_model, 1)
        # 门控融合 + 双任务头（同 MA-MFN）
        self.gate_t = nn.Linear(3 * d_model, d_model)
        self.gate_a = nn.Linear(3 * d_model, d_model)
        self.gate_v = nn.Linear(3 * d_model, d_model)
        self.fuse = nn.Linear(3 * d_model, d_model)
        self.fuse_norm = nn.LayerNorm(d_model)
        self.cls_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_model // 2, n_classes))
        self.reg_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1))

    @staticmethod
    def _masked_softmax(scores, mask):
        scores = scores.masked_fill(mask < 0.5, -1e9)
        return F.softmax(scores, dim=-1)

    def forward(self, t, a, v, mt, ma, mv):
        # 序列编码（传入 mask 让 Transformer 屏蔽缺失位置）
        zt_seq = self.enc_t(t, mt)
        za_seq = self.enc_a(a, ma)
        zv_seq = self.enc_v(v, mv)
        # 片段级注意力 β
        beta_t = self._masked_softmax(self.score_t(zt_seq).squeeze(-1), mt)
        beta_a = self._masked_softmax(self.score_a(za_seq).squeeze(-1), ma)
        beta_v = self._masked_softmax(self.score_v(zv_seq).squeeze(-1), mv)
        # 注意力加权池化
        zt = (beta_t.unsqueeze(-1) * zt_seq * mt.unsqueeze(-1)).sum(dim=1)
        za = (beta_a.unsqueeze(-1) * za_seq * ma.unsqueeze(-1)).sum(dim=1)
        zv = (beta_v.unsqueeze(-1) * zv_seq * mv.unsqueeze(-1)).sum(dim=1)
        # 模态级注意力 α
        cat_z = torch.stack([zt, za, zv], dim=1)         # (B,3,d)
        alpha = F.softmax(self.q(cat_z).squeeze(-1), dim=-1)  # (B,3)
        # 门控融合 + 双任务头（与 MA-MFN 一致）
        cat = torch.cat([zt, za, zv], dim=-1)
        gt = torch.sigmoid(self.gate_t(cat))
        ga = torch.sigmoid(self.gate_a(cat))
        gv = torch.sigmoid(self.gate_v(cat))
        fused_in = torch.cat([gt * zt, ga * za, gv * zv], dim=-1)
        h = self.fuse_norm(F.gelu(self.fuse(fused_in)))
        logits = self.cls_head(h)
        reg = 3.0 * torch.tanh(self.reg_head(h).squeeze(-1))  # 约束到 [-3,3]
        return logits, reg, alpha, (beta_t, beta_a, beta_v)


# ============ 缺失增强 ============
# 附件3实测分布（2026-09-24 核对）：
#   - text 无中间缺失，只有尾部 padding（有效长度 8~50，均值 22）
#   - audio 与 vision 零段位置 IoU=0.995，几乎完全重合（相关缺失）
#   - 有效范围内的缺失段很短：长度 1~4，中位 1，每样本 0~8 段
#   - 第 0 步恒零是固定现象，不是缺失
# 因此增强策略：text 不做中间缺失；audio+vision 在相同位置同时做短缺失。
AUG_PROB = 0.7
MISS_RUNS_MAX = 4        # 每样本最多做几个短缺失段（对齐附件3实测 0~8）
MISS_LEN_MIN = 1         # 短缺失段长度下限（位置数）
MISS_LEN_MAX = 4         # 短缺失段长度上限（对齐附件3实测 max=4）


def apply_missing_augmentation(t, a, v, mt, ma, mv, rng=None):
    """模拟附件3实测的缺失模式：
    - text 不做中间缺失（仅保留尾部 padding）
    - audio+vision 在相同位置同时做连续短缺失（位置重合）
    输入均为一维 (T,...) 张量，原地修改。"""
    import random as _r
    r = rng or _r
    if r.random() >= AUG_PROB:
        return t, a, v, mt, ma, mv
    # 在有效范围内随机选若干短区间，audio/vision 同时置零
    n_runs = rng.randint(0, MISS_RUNS_MAX) if rng else _r.randint(0, MISS_RUNS_MAX)
    for _ in range(n_runs):
        L = rng.randint(MISS_LEN_MIN, MISS_LEN_MAX) if rng else _r.randint(MISS_LEN_MIN, MISS_LEN_MAX)
        # 避开第 0 步（恒零），在 [1, T_MAX-L] 随机起点
        s = rng.randint(1, T_MAX - L) if rng else _r.randint(1, T_MAX - L)
        e = s + L
        a[s:e, :] = 0.0; ma[s:e] = 0.0
        v[s:e, :] = 0.0; mv[s:e] = 0.0
    return t, a, v, mt, ma, mv


# ============ 评估 ============
def compute_metrics(y_true_cls, y_pred_cls, y_true_reg, y_pred_reg):
    from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error
    from scipy.stats import pearsonr
    acc = accuracy_score(y_true_cls, y_pred_cls)
    f1 = f1_score(y_true_cls, y_pred_cls, average="macro")
    mae = mean_absolute_error(y_true_reg, y_pred_reg)
    pr = pearsonr(y_true_reg, y_pred_reg)[0]
    return {"accuracy": round(float(acc), 4),
            "macro_f1": round(float(f1), 4),
            "mae": round(float(mae), 4),
            "pearson": round(float(pr), 4)}
