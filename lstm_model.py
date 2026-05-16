"""LSTM Encoder-Decoder with Bahdanau Attention for translation."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_layers=2, dropout=0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers,
                            batch_first=True, bidirectional=True, dropout=dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src):
        # src: (batch, seq_len)
        embedded = self.dropout(self.embedding(src))
        outputs, (hidden, cell) = self.lstm(embedded)
        # outputs: (batch, seq_len, hidden_dim * 2)
        # hidden: (num_layers * 2, batch, hidden_dim)
        return outputs, hidden, cell


class BahdanauAttention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.attn = nn.Linear(hidden_dim * 3, hidden_dim)
        self.v = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, decoder_hidden, encoder_outputs, mask=None):
        # decoder_hidden: (batch, hidden_dim)
        # encoder_outputs: (batch, src_len, hidden_dim * 2)
        src_len = encoder_outputs.shape[1]
        hidden = decoder_hidden.unsqueeze(1).repeat(1, src_len, 1)
        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim=2)))
        attention = self.v(energy).squeeze(2)  # (batch, src_len)
        if mask is not None:
            attention = attention.masked_fill(mask == 0, -1e10)
        return F.softmax(attention, dim=1)


class Decoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_layers=2, dropout=0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.attention = BahdanauAttention(hidden_dim)
        self.lstm = nn.LSTM(embed_dim + hidden_dim * 2, hidden_dim, num_layers,
                            batch_first=True, dropout=dropout)
        self.fc_out = nn.Linear(hidden_dim * 3 + embed_dim, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, tgt, encoder_outputs, hidden, cell, src_mask=None):
        # tgt: (batch, 1) — single token at a time
        embedded = self.dropout(self.embedding(tgt))  # (batch, 1, embed_dim)
        attn_weights = self.attention(hidden[-1], encoder_outputs, src_mask)
        context = torch.bmm(attn_weights.unsqueeze(1), encoder_outputs)  # (batch, 1, hidden*2)
        lstm_input = torch.cat((embedded, context), dim=2)
        output, (hidden, cell) = self.lstm(lstm_input, (hidden, cell))
        # output: (batch, 1, hidden_dim)
        prediction = self.fc_out(torch.cat((output, context, embedded), dim=2))
        return prediction.squeeze(1), hidden, cell, attn_weights


class Seq2SeqLSTM(nn.Module):
    def __init__(self, vocab_size, embed_dim=256, hidden_dim=512, num_layers=2, dropout=0.3):
        super().__init__()
        self.encoder = Encoder(vocab_size, embed_dim, hidden_dim, num_layers, dropout)
        self.decoder = Decoder(vocab_size, embed_dim, hidden_dim, num_layers, dropout)
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers  # decoder layers (encoder has num_layers*2 because bidirectional)

    def encode(self, src):
        return self.encoder(src)

    def decode_step(self, tgt_token, encoder_outputs, hidden, cell, src_mask=None):
        return self.decoder(tgt_token, encoder_outputs, hidden, cell, src_mask)

    def forward(self, src, tgt, teacher_forcing_ratio=0.5):
        batch_size, tgt_len = tgt.shape
        encoder_outputs, hidden, cell = self.encode(src)

        # 取 decoder 方向的 hidden (双向 encoder 的 hidden 需要转换)
        # encoder hidden: (num_layers*2, batch, hidden_dim)
        # decoder hidden: (num_layers, batch, hidden_dim)
        # 把双向 hidden 相加合并
        hidden = hidden.view(self.num_layers, 2, batch_size, self.hidden_dim)
        hidden = hidden.sum(dim=1)  # (num_layers, batch, hidden_dim)
        cell = cell.view(self.num_layers, 2, batch_size, self.hidden_dim)
        cell = cell.sum(dim=1)

        outputs = torch.zeros(batch_size, tgt_len, self.vocab_size).to(src.device)
        input_token = tgt[:, 0].unsqueeze(1)  # <sos>

        for t in range(tgt_len):
            output, hidden, cell, _ = self.decoder(input_token, encoder_outputs, hidden, cell)
            outputs[:, t] = output
            teacher_force = torch.rand(1).item() < teacher_forcing_ratio
            top1 = output.argmax(1).unsqueeze(1)
            input_token = tgt[:, t].unsqueeze(1) if teacher_force and t < tgt_len - 1 else top1
        return outputs

    def translate(self, src, tokenizer, max_len=128, device='cuda'):
        """推理：给定源语言 token ids，生成目标语言 token ids。"""
        self.eval()
        src = src.to(device)
        encoder_outputs, hidden, cell = self.encode(src)
        hidden = hidden.view(self.num_layers, 2, src.size(0), self.hidden_dim).sum(dim=1)
        cell = cell.view(self.num_layers, 2, src.size(0), self.hidden_dim).sum(dim=1)

        sos_id = tokenizer.pad_token_id or 0
        eos_id = tokenizer.eos_token_id or 0

        input_token = torch.tensor([[sos_id]] * src.size(0)).to(device)
        outputs = []

        for _ in range(max_len):
            output, hidden, cell, _ = self.decoder(input_token, encoder_outputs, hidden, cell)
            top1 = output.argmax(1).unsqueeze(1)
            outputs.append(top1)
            if (top1 == eos_id).all():
                break
            input_token = top1

        return torch.cat(outputs, dim=1)
