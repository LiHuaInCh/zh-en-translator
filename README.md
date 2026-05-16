# 基于编码器-解码器架构的中英互译系统

> 软件工程专业 2026 届毕业设计

## 概述

本系统实现并对比了两种经典的编码器-解码器（Encoder-Decoder）架构在中文-英文翻译任务上的性能。

| 模型 | 架构 | 训练策略 | 中→英 BLEU |
|------|------|----------|------------|
| **Pro** | Transformer (Vaswani 2017) | opus-mt 预训练 + 27万句对 Fine-tune | **35.07** |
| **Flash** | LSTM + Bahdanau Attention | 从零训练 (5万 × 12轮) | ≈0 |

**核心发现**：预训练策略对神经机器翻译质量具有决定性作用。在无预训练条件下，即使 5 万句对也无法使 LSTM 模型收敛。

## 系统架构

```
┌────────────────────────────────┐
│         Gradio Web UI          │  ← 展示层
├────────────────────────────────┤
│    Pro (Transformer)  Flash    │  ← 模型层
│    opus-mt 77.9M    LSTM 157M  │
├────────────────────────────────┤
│    SentencePiece BPE (65K)     │  ← 数据层
│    OPUS 语料 (28.8万句对)       │
└────────────────────────────────┘
```

## 技术栈

### Pro 模型
- **架构**: Transformer Encoder-Decoder，6层/6层
- **注意力**: Multi-Head Self-Attention (8头) + Cross-Attention
- **分词**: SentencePiece BPE，词表 65K
- **解码**: Beam Search (束宽4)
- **训练**: opus-mt 预训练权重 + FP16 混合精度 Fine-tune

### Flash 模型
- **架构**: BiLSTM Encoder (2层) + LSTM Decoder (1层)
- **注意力**: Bahdanau Additive Attention
- **词向量**: 256维
- **解码**: Greedy Decoding
- **训练**: 从零训练，Adam 优化器 + Teacher Forcing

## 快速启动

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 下载模型权重（放在 models/ 目录下）
# 网盘链接: [待补充]

# 3. 启动
python app.py
# 浏览器打开 http://127.0.0.1:7860
```

> **注意**：存放模型和项目的**路径中不能包含中文**，否则 SentencePiece 分词器无法加载 `source.spm` 文件。
```

## 目录结构

```
├── app.py              # Gradio Web 界面
├── config.py           # 全局配置
├── lstm_model.py       # LSTM+Attention 模型定义
├── prepare_data.py     # OPUS 数据下载与预处理
├── train.py            # Transformer Fine-tune 训练
├── train_lstm.py       # LSTM 从零训练
└── requirements.txt    # Python 依赖
```

## 数据来源

| 语料库 | 句对数 | 领域 |
|--------|--------|------|
| News-Commentary v16 | 125,996 | 新闻评论 |
| TED2020 v1 | 16,382 | 演讲口语 |
| MultiUN v1 | 160,000 | 联合国文件 |

## 实验结果

### Pro (Transformer)
- **中→英**: BLEU 35.07 / chrF 60.70
- 翻译通顺自然，可实用

### Flash (LSTM+Attention)
- **中→英**: BLEU ≈ 0
- 输出为 subword 碎片，不可用
- 1 万→5 万样本扩展未能带来质变

## 许可证

MIT License
