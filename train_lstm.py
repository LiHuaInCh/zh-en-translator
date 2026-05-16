"""Train LSTM seq2seq model for zh-en translation (enhanced version)."""
import sys
import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from transformers import AutoTokenizer
from lstm_model import Seq2SeqLSTM


class TranslationDataset(Dataset):
    def __init__(self, data_file, src_lang='zh', tgt_lang='en', max_samples=100000):
        self.pairs = []
        print(f"加载数据 (最多 {max_samples} 句对)...")
        with open(data_file, encoding='utf-8') as f:
            for i, line in enumerate(f):
                if i >= max_samples:
                    break
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                self.pairs.append((
                    obj['translation'][src_lang],
                    obj['translation'][tgt_lang],
                ))
        print(f"已加载 {len(self.pairs)} 句对")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        return self.pairs[idx]


def collate_fn(batch, tokenizer, max_len=64):
    src_texts, tgt_texts = zip(*batch)
    src_enc = tokenizer(list(src_texts), padding=True, truncation=True,
                        max_length=max_len, return_tensors='pt')
    tgt_enc = tokenizer(list(tgt_texts), padding=True, truncation=True,
                        max_length=max_len, return_tensors='pt')
    return src_enc.input_ids, tgt_enc.input_ids


def train_lstm(direction='zh2en'):
    from config import DATA_DIR

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"设备: {device}")

    tokenizer_path = f'C:/Users/26370/zh-en-translator/pretrained/opus-mt-{"zh-en" if direction == "zh2en" else "en-zh"}'
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)
    vocab_size = tokenizer.vocab_size
    print(f"词典: {vocab_size}")

    # 10万句对，2/3训练 1/3验证
    ds = TranslationDataset(DATA_DIR / 'train.json', max_samples=50000)
    n_train = int(len(ds) * 0.8)
    n_val = len(ds) - n_train
    train_ds, val_ds = torch.utils.data.random_split(ds, [n_train, n_val])
    print(f"训练: {n_train}  验证: {n_val}")

    BATCH = 32
    train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True,
                              collate_fn=lambda b: collate_fn(b, tokenizer, max_len=64),
                              num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH,
                            collate_fn=lambda b: collate_fn(b, tokenizer, max_len=64),
                            num_workers=0, pin_memory=True)

    # 缩小模型加速训练: 1层 LSTM，256 隐层
    model = Seq2SeqLSTM(vocab_size, embed_dim=256, hidden_dim=512, num_layers=1, dropout=0.3)
    model = model.to(device)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"参数: {n_params:.1f}M")

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)

    num_epochs = 12
    best_val_loss = float('inf')
    save_dir = Path('models') / f'lstm-{direction}'
    save_dir.mkdir(parents=True, exist_ok=True)

    print(f"训练 LSTM，{num_epochs} 轮，每轮 {len(train_loader)} batch")
    print(f"预计时间: ~6-8 小时\n")

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for i, (src, tgt) in enumerate(train_loader):
            src, tgt = src.to(device), tgt.to(device)
            optimizer.zero_grad()

            output = model(src, tgt, teacher_forcing_ratio=0.5)
            output = output[:, :-1].reshape(-1, vocab_size)
            target = tgt[:, 1:].reshape(-1)

            loss = criterion(output, target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()

            if (i + 1) % 200 == 0:
                avg_loss = total_loss / 200
                print(f'  Epoch {epoch+1:2d}/{num_epochs} | Batch {i+1:4d}/{len(train_loader)} | Loss: {avg_loss:.4f}')
                total_loss = 0

        # 验证
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for src, tgt in val_loader:
                src, tgt = src.to(device), tgt.to(device)
                output = model(src, tgt, teacher_forcing_ratio=0)
                output = output[:, :-1].reshape(-1, vocab_size)
                target = tgt[:, 1:].reshape(-1)
                val_loss += criterion(output, target).item()
        val_loss /= len(val_loader)
        scheduler.step(val_loss)
        print(f'  >>> Epoch {epoch+1} 验证 Loss: {val_loss:.4f} | LR: {optimizer.param_groups[0]["lr"]:.6f}')

        # 抽样翻译
        with torch.no_grad():
            sample_src, sample_tgt = next(iter(val_loader))
            sample_src = sample_src[:2].to(device)
            out_ids = model.translate(sample_src, tokenizer, max_len=64, device=device)
            for j in range(min(2, sample_src.size(0))):
                src_decoded = tokenizer.decode(sample_src[j], skip_special_tokens=True)[:60]
                tgt_decoded = tokenizer.decode(out_ids[j], skip_special_tokens=True)[:60]
                print(f'  Sample: [{src_decoded}] → [{tgt_decoded}]')

        # 保存最佳
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), save_dir / 'model.pt')
            print(f'  已保存最佳模型 (val_loss={best_val_loss:.4f})')

    # 最终保存
    tokenizer.save_pretrained(str(save_dir))
    print(f"\n模型已保存: {save_dir}")
    print(f"最佳验证 Loss: {best_val_loss:.4f}")


if __name__ == '__main__':
    direction = sys.argv[1] if len(sys.argv) > 1 else 'zh2en'
    train_lstm(direction)
