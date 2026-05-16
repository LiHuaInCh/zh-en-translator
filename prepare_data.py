"""数据准备：从 OPUS 下载的平行语料中加载、清洗、切分，保存为 JSON。
要求先下载数据到 data/raw/ 目录。
"""
import json
import random
from pathlib import Path
from datasets import Dataset, DatasetDict


def load_opus_file(en_path: Path, zh_path: Path, max_lines: int = None):
    """从 OPUS Moses 格式文件加载平行句对。"""
    pairs = []
    with open(en_path, encoding="utf-8") as f_en, open(zh_path, encoding="utf-8") as f_zh:
        for i, (en, zh) in enumerate(zip(f_en, f_zh)):
            if max_lines and i >= max_lines:
                break
            pairs.append({"translation": {"en": en.strip(), "zh": zh.strip()}})
    return pairs


def prepare_data():
    from config import MAX_SEQ_LENGTH, DATA_DIR

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    raw_dir = Path(__file__).parent / "data" / "raw"

    # 加载三个语料，MultiUN 只取子集
    print("加载语料...")
    pairs = []
    pairs += load_opus_file(raw_dir / "nc" / "News-Commentary.en-zh.en",
                            raw_dir / "nc" / "News-Commentary.en-zh.zh")
    print(f"  News-Commentary: {len(pairs)} 句对")

    pairs += load_opus_file(raw_dir / "ted" / "TED2020.en-zh.en",
                            raw_dir / "ted" / "TED2020.en-zh.zh")
    print(f"  + TED2020: {len(pairs)} 句对")

    pairs += load_opus_file(raw_dir / "un" / "MultiUN.en-zh.en",
                            raw_dir / "un" / "MultiUN.en-zh.zh",
                            max_lines=160_000)
    print(f"  + MultiUN(16万): {len(pairs)} 句对")

    # 清洗
    print("清洗数据...")
    cleaned = []
    for p in pairs:
        zh = p["translation"]["zh"]
        en = p["translation"]["en"]
        if not zh or not en:
            continue
        if len(zh) < 3 or len(zh) > MAX_SEQ_LENGTH * 4:
            continue
        if len(en.split()) < 2 or len(en.split()) > MAX_SEQ_LENGTH:
            continue
        cleaned.append(p)

    print(f"清洗后: {len(cleaned)} 句对 (去除 {len(pairs) - len(cleaned)})")

    # 打乱 & 切分
    random.seed(42)
    random.shuffle(cleaned)

    n = len(cleaned)
    train_end = int(n * 0.94)
    val_end = int(n * 0.97)

    dataset = DatasetDict({
        "train": Dataset.from_list(cleaned[:train_end]),
        "val": Dataset.from_list(cleaned[train_end:val_end]),
        "test": Dataset.from_list(cleaned[val_end:]),
    })

    print(f"训练集: {len(dataset['train'])}  验证集: {len(dataset['val'])}  测试集: {len(dataset['test'])}")

    for name in ["train", "val", "test"]:
        path = DATA_DIR / f"{name}.json"
        dataset[name].to_json(str(path), force_ascii=False)
        print(f"已保存: {path}")

    print("数据准备完成。")


if __name__ == "__main__":
    prepare_data()
