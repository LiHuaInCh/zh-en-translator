# 基于编码器-解码器架构的中英互译系统

## 项目简介

本系统实现了两种编码器-解码器架构的中英翻译模型：

| 模型 | 架构 | 训练策略 | BLEU | chrF |
|------|------|----------|------|------|
| Pro | Transformer | opus-mt 预训练 + 27万 Fine-tune | 35.07 | 60.70 |
| Flash | LSTM+Attention | 从零训练 (5万 × 12轮) | ≈0 | ≈0 |

核心结论：预训练策略对翻译质量具有决定性作用。

## 环境

- Python 3.10+
- NVIDIA GPU + CUDA 12.4（或 CPU，速度较慢）
- Windows / Linux

## 快速启动

```bash
pip install -r requirements.txt
python app.py
# 浏览器 http://127.0.0.1:7860
```

## 项目结构

| 文件 | 说明 |
|------|------|
| `app.py` | Gradio Web 界面，支持 Pro/Flash 模型切换 |
| `config.py` | 全局配置（路径、超参数） |
| `lstm_model.py` | LSTM+Attention 模型定义（Encoder/Decoder/Attention） |
| `prepare_data.py` | OPUS 数据下载、清洗、划分 |
| `train.py` | Transformer Fine-tune 训练脚本 |
| `train_lstm.py` | LSTM 从零训练脚本 |
| `requirements.txt` | Python 依赖 |
| `README.md` | 本文件 |
| `毕设论文_*.docx` | 完整论文 |

## 模型权重

模型文件（~1.2GB）未包含在提交中，获取方式：
- 网盘：[待补充链接]
- 或自行训练：`python train.py zh2en` / `python train.py en2zh`

## 数据来源

OPUS 项目三个子语料库：
- News-Commentary v16（125,996 句对）
- TED2020 v1（16,382 句对）
- MultiUN v1（160,000 句对）
