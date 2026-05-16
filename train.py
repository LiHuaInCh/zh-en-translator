"""Fine-tune Helsinki-NLP opus-mt 模型，中英互译。
用法:
    python train.py zh2en   # 训练中→英
    python train.py en2zh   # 训练英→中
"""
import sys
import os
import json
import numpy as np
from pathlib import Path
from datasets import Dataset, DatasetDict
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback,
)
import evaluate

def load_saved_data(data_dir: Path) -> DatasetDict:
    """从 prepare_data.py 保存的 JSON 文件中加载数据。"""
    splits = {}
    for name in ["train", "val", "test"]:
        path = data_dir / f"{name}.json"
        if path.exists():
            splits[name] = Dataset.from_json(str(path))
    return DatasetDict(splits)


def preprocess(examples, tokenizer, max_length):
    """Tokenize 翻译句对。"""
    inputs = tokenizer(examples["src"], max_length=max_length, truncation=True, padding=False)
    labels = tokenizer(examples["tgt"], max_length=max_length, truncation=True, padding=False)
    inputs["labels"] = labels["input_ids"]
    return inputs


def compute_metrics(eval_preds, tokenizer, metric_bleu, metric_chrf):
    """计算 sacreBLEU 和 chrF。"""
    preds, labels = eval_preds
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # sacreBLEU 需要 list of references
    decoded_labels = [[ref] for ref in decoded_labels]

    bleu = metric_bleu.compute(predictions=decoded_preds, references=decoded_labels)
    chrf = metric_chrf.compute(predictions=decoded_preds, references=decoded_labels)

    return {"bleu": bleu["score"], "chrf": chrf["score"]}


def train(direction: str):
    from config import (
        MODEL_ZH_EN, MODEL_EN_ZH, DATA_DIR,
        OUTPUT_DIR_ZH_EN, OUTPUT_DIR_EN_ZH, LOG_DIR,
        MAX_SEQ_LENGTH, BATCH_SIZE, EVAL_BATCH_SIZE, GRADIENT_ACCUMULATION,
        LEARNING_RATE, NUM_EPOCHS, WARMUP_STEPS,
        LOGGING_STEPS, EVAL_STEPS, SAVE_STEPS,
    )

    # 选择模型和输出路径
    if direction == "zh2en":
        model_name = MODEL_ZH_EN
        output_dir = str(OUTPUT_DIR_ZH_EN)
        src_lang, tgt_lang = "zh", "en"
    else:
        model_name = MODEL_EN_ZH
        output_dir = str(OUTPUT_DIR_EN_ZH)
        src_lang, tgt_lang = "en", "zh"

    print(f"加载模型: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    # 加载数据
    dataset = load_saved_data(DATA_DIR)
    if len(dataset) == 0:
        print("未找到预处理数据，请先运行 prepare_data.py")
        sys.exit(1)

    # 把 translation 字段展成 src / tgt
    def flatten(example):
        return {"src": example["translation"][src_lang], "tgt": example["translation"][tgt_lang]}

    dataset = dataset.map(flatten, remove_columns=["translation"])

    # Tokenize
    tokenized = dataset.map(
        lambda x: preprocess(x, tokenizer, MAX_SEQ_LENGTH),
        batched=True,
        remove_columns=dataset["train"].column_names,
    )

    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model, padding=True)

    metric_bleu = evaluate.load("sacrebleu")
    metric_chrf = evaluate.load("chrf")

    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=EVAL_BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION,
        learning_rate=LEARNING_RATE,
        num_train_epochs=NUM_EPOCHS,
        warmup_steps=WARMUP_STEPS,
        logging_steps=LOGGING_STEPS,
        logging_dir=str(LOG_DIR / direction),
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="bleu",
        greater_is_better=True,
        fp16=True,
        predict_with_generate=True,
        generation_max_length=MAX_SEQ_LENGTH,
        report_to=["tensorboard"],
        dataloader_num_workers=2,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["val"],
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=lambda p: compute_metrics(p, tokenizer, metric_bleu, metric_chrf),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )

    print(f"开始训练 {direction}，日志: tensorboard --logdir {LOG_DIR}")
    trainer.train()

    # 保存最佳模型
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"模型已保存: {output_dir}")

    # 最终评估
    print("\n=== 最终评估 ===")
    eval_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        per_device_eval_batch_size=EVAL_BATCH_SIZE,
        fp16=True,
        predict_with_generate=True,
        generation_max_length=MAX_SEQ_LENGTH,
        report_to=[],
    )
    eval_trainer = Seq2SeqTrainer(
        model=model,
        args=eval_args,
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=lambda p: compute_metrics(p, tokenizer, metric_bleu, metric_chrf),
    )
    test_results = eval_trainer.evaluate(tokenized["test"])
    print(f"测试集 BLEU: {test_results.get('eval_bleu', 'N/A')}")
    print(f"测试集 chrF: {test_results.get('eval_chrf', 'N/A')}")


if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in ("zh2en", "en2zh"):
        print("用法: python train.py zh2en  或  python train.py en2zh")
        sys.exit(1)
    direction = sys.argv[1]

    # 检查数据是否已准备
    from config import DATA_DIR
    if not (DATA_DIR / "train.json").exists():
        # 尝试导入 (Windows 路径兼容)
        sys.path.insert(0, str(Path(__file__).parent))
        from prepare_data import prepare_data
        prepare_data()

    train(direction)
