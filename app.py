"""中英互译系统 — Gradio 演示界面。
支持 Transformer (opus-mt) 和 LSTM+Attention 两种模型切换。
用法:
    python app.py
"""
import re
import torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import gradio as gr

# ---------- 语言检测 ----------
def detect_lang(text: str) -> str:
    zh_chars = len(re.findall(r"[一-鿿]", text))
    return "zh" if zh_chars > len(text) * 0.15 else "en"


# ========== Transformer 模型 ==========
def load_transformer(model_path: str, device: str):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path).to(device)
    model.eval()
    return tokenizer, model


def init_transformer():
    from config import OUTPUT_DIR_ZH_EN, OUTPUT_DIR_EN_ZH, MODEL_ZH_EN, MODEL_EN_ZH
    device = "cuda" if torch.cuda.is_available() else "cpu"

    zh_en_path = str(OUTPUT_DIR_ZH_EN) if OUTPUT_DIR_ZH_EN.exists() else MODEL_ZH_EN
    en_zh_path = str(OUTPUT_DIR_EN_ZH) if OUTPUT_DIR_EN_ZH.exists() else MODEL_EN_ZH

    print(f"[Transformer] zh→en: {zh_en_path}")
    print(f"[Transformer] en→zh: {en_zh_path}")

    zh_en_tok, zh_en_model = load_transformer(zh_en_path, device)
    en_zh_tok, en_zh_model = load_transformer(en_zh_path, device)

    return {
        "zh2en": (zh_en_tok, zh_en_model, device),
        "en2zh": (en_zh_tok, en_zh_model, device),
    }


def translate_transformer(text, models):
    src_lang = detect_lang(text)
    direction = "zh2en" if src_lang == "zh" else "en2zh"
    tokenizer, model, device = models[direction]
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=256).to(device)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_length=256, num_beams=4, early_stopping=True)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


# ========== LSTM 模型 ==========
LSTM_MODEL = None

def load_lstm():
    global LSTM_MODEL
    if LSTM_MODEL is not None:
        return LSTM_MODEL

    from lstm_model import Seq2SeqLSTM
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 用 zh→en 的 tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        "C:/Users/26370/zh-en-translator/pretrained/opus-mt-zh-en", local_files_only=True)

    model = Seq2SeqLSTM(tokenizer.vocab_size, embed_dim=256, hidden_dim=512, num_layers=2)
    model_path = Path("models/lstm-zh2en/model.pt")
    if model_path.exists():
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
        print("[LSTM] 已加载训练好的模型")
    else:
        print("[LSTM] 使用未训练模型（随机权重），请先运行 train_lstm.py")

    model = model.to(device)
    model.eval()

    LSTM_MODEL = (tokenizer, model, device)
    return LSTM_MODEL


def translate_lstm(text):
    tokenizer, model, device = load_lstm()
    src_lang = detect_lang(text)
    if src_lang != "zh":
        return "(LSTM 模型仅支持中文→英文)"

    src_ids = tokenizer.encode(text)
    src_tensor = torch.tensor([src_ids], dtype=torch.long).to(device)
    with torch.no_grad():
        out_ids = model.translate(src_tensor, tokenizer, max_len=128, device=device)
    result = tokenizer.decode(out_ids[0], skip_special_tokens=True)
    if not result.strip():
        return "(LSTM 训练数据不足，翻译失败 — 请切换到 Pro 模型)"
    return result


# ========== 主翻译函数 ==========
TRANSFORMER_MODELS = None

def translate(text: str, model_choice: str) -> str:
    if not text.strip():
        return ""

    if model_choice == "Flash (LSTM + Attention)":
        return translate_lstm(text)
    else:
        global TRANSFORMER_MODELS
        if TRANSFORMER_MODELS is None:
            TRANSFORMER_MODELS = init_transformer()
        return translate_transformer(text, TRANSFORMER_MODELS)


# ========== UI ==========
HEADER = """
# 中英互译系统
基于 Encoder-Decoder 架构 | 支持 Pro (Transformer) / Flash (LSTM+Attention) 模型切换
"""

EXAMPLES = [
    ["今天天气真好，我们出去玩吧。"],
    ["Artificial intelligence is transforming every industry."],
    ["深度学习是机器学习的一个重要分支。"],
    ["The weather is beautiful today."],
    ["请帮我把这份文件翻译成英文。"],
]

if __name__ == "__main__":
    demo = gr.Interface(
        fn=translate,
        inputs=[
            gr.Textbox(label="输入文本", placeholder="输入中文或英文，系统会自动检测语言...", lines=4),
            gr.Radio(choices=["Pro (Transformer)", "Flash (LSTM + Attention)"],
                     label="模型选择", value="Pro (Transformer)"),
        ],
        outputs=gr.Textbox(label="翻译结果", lines=4),
        title="中英互译系统",
        description=HEADER,
        examples=[[e[0], "Pro (Transformer)"] for e in EXAMPLES],
    )
    demo.launch(server_name="127.0.0.1", share=False)
