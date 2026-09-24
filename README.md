# 2026 Huawei Cup E题: 复杂场景下多模态情感预测

## 项目结构
- `scripts/` — 全部 Python 代码（Q1特征提取、Q2缺失鲁棒模型、Q3可解释模型、知识蒸馏、积分梯度）
- `Q1_features/` — 自提取特征汇总表与映射表
- `Q2_results/` — 缺失鲁棒模型指标 + 附件3预测CSV
- `Q3_results/` — 可解释模型指标 + 附件4预测解释 + 遮挡验证 + IG重要性
- `models/` — 模型权重（.pt，本地未上传）
- `figures/` — 论文图表（本地未上传）

## 核心结果
| 指标 | Q2 (MAMFN) | Q3 (AEMN蒸馏) |
|---|---|---|
| test Accuracy | 0.663 | 0.636 |
| test macro-F1 | 0.638 | 0.612 |
| test MAE | 0.711 | 0.653 |
| test Pearson | 0.642 | 0.645 |

## 方法
- Q1: wav2vec2 CTC forced alignment 真词级时间对齐
- Q2: Transformer(d=64) + Huber + 3·tanh + 掩码感知缺失增强
- Q3: 注意力α/β + 遮挡法 + 积分梯度IG 三路互证
- 创新: AdaptiveSwish自适应激活消融 + 知识蒸馏(教师d=64→学生d=32)
