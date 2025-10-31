"""
数据加载模块 - 作业要求：在小数据集上验证模型
支持语言建模和序列到序列任务
"""

import torch
from torch.utils.data import Dataset, DataLoader
from collections import Counter
import re
import xml.etree.ElementTree as ET

# ------------------------- 构建词汇表 -------------------------
def build_vocab(texts, vocab_size=10000):
    """构建词汇表"""
    counter = Counter()
    for text in texts:
        words = re.findall(r'\b\w+\b|[^\w\s]', text.lower())
        counter.update(words)

    most_common = counter.most_common(vocab_size - 4)
    vocab = {'<pad>': 0, '<unk>': 1, '<bos>': 2, '<eos>': 3}
    for word, _ in most_common:
        vocab[word] = len(vocab)
    return vocab

# ------------------------- 文本数据集 -------------------------
class TextDataset(Dataset):
    """文本数据集 - 语言建模任务"""
    def __init__(self, texts, vocab, seq_len=128):
        self.texts = texts
        self.vocab = vocab
        self.seq_len = seq_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        tokens = self.text_to_tokens(text)
        if len(tokens) > self.seq_len:
            tokens = tokens[:self.seq_len]
        else:
            tokens += [self.vocab['<pad>']] * (self.seq_len - len(tokens))
        input_ids = torch.tensor(tokens[:-1], dtype=torch.long)
        target_ids = torch.tensor(tokens[1:], dtype=torch.long)
        return input_ids, target_ids

    def text_to_tokens(self, text):
        words = re.findall(r'\b\w+\b|[^\w\s]', text.lower())
        return [self.vocab.get(word, self.vocab['<unk>']) for word in words]

# ------------------------- 序列到序列数据集 -------------------------
class Seq2SeqDataset(Dataset):
    """序列到序列数据集 - 机器翻译任务"""
    def __init__(self, source_texts, target_texts, src_vocab, tgt_vocab, seq_len=128):
        self.source_texts = source_texts
        self.target_texts = target_texts
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
        self.seq_len = seq_len

    def __len__(self):
        return len(self.source_texts)

    def __getitem__(self, idx):
        src_text = self.source_texts[idx]
        tgt_text = self.target_texts[idx]
        src_tokens = self.text_to_tokens(src_text, self.src_vocab)
        tgt_tokens = [self.tgt_vocab['<bos>']] + self.text_to_tokens(tgt_text, self.tgt_vocab) + [self.tgt_vocab['<eos>']]
        src_tokens = self._pad_or_truncate(src_tokens, self.seq_len, self.src_vocab['<pad>'])
        tgt_tokens = self._pad_or_truncate(tgt_tokens, self.seq_len, self.tgt_vocab['<pad>'])
        return torch.tensor(src_tokens, dtype=torch.long), torch.tensor(tgt_tokens, dtype=torch.long)

    def text_to_tokens(self, text, vocab):
        words = re.findall(r'\b\w+\b|[^\w\s]', text.lower())
        return [vocab.get(word, vocab['<unk>']) for word in words]

    def _pad_or_truncate(self, tokens, max_len, pad_token):
        if len(tokens) > max_len:
            return tokens[:max_len]
        else:
            return tokens + [pad_token] * (max_len - len(tokens))

# ------------------------- 解析 XML -------------------------
def parse_iwslt_xml(file_path):
    tree = ET.parse(file_path)
    root = tree.getroot()
    return [seg.text.strip() for seg in root.iter('seg') if seg.text]

# ------------------------- 加载 Tiny Shakespeare -------------------------
def load_tiny_shakespeare_local(file_path, seq_len=128, batch_size=32):
    """加载本地 Tiny Shakespeare 文件"""
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()
    lines = [line.strip() for line in text.split('\n') if line.strip()]
    # 划分训练/验证集 90% / 10%
    split_idx = int(len(lines) * 0.9)
    train_texts = lines[:split_idx]
    val_texts = lines[split_idx:]
    vocab = build_vocab(train_texts)
    train_loader = DataLoader(TextDataset(train_texts, vocab, seq_len), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TextDataset(val_texts, vocab, seq_len), batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, vocab

# ------------------------- 加载 IWSLT2017 本地 -------------------------
def load_iwslt2017_local(seq_len=128, batch_size=32):
    """加载本地 IWSLT2017 XML 文件"""
    # 请根据你的路径修改
    train_de = r"D:\pycode\transformer\data\train.tags.de-en.de"
    train_en = r"D:\pycode\transformer\data\train.tags.de-en.en"
    val_de = r"D:\pycode\transformer\data\IWSLT17.TED.dev2010.de-en.de.xml"
    val_en = r"D:\pycode\transformer\data\IWSLT17.TED.dev2010.de-en.en.xml"
    test_de = r"D:\pycode\transformer\data\IWSLT17.TED.tst2010.de-en.de.xml"
    test_en = r"D:\pycode\transformer\data\IWSLT17.TED.tst2010.de-en.en.xml"

    def load_text(file_path):
        if file_path.endswith(".xml"):
            return parse_iwslt_xml(file_path)
        else:
            with open(file_path, 'r', encoding='utf-8') as f:
                return [line.strip() for line in f if line.strip()]

    train_src = load_text(train_en)
    train_tgt = load_text(train_de)
    val_src = load_text(val_en)
    val_tgt = load_text(val_de)
    test_src = load_text(test_en)
    test_tgt = load_text(test_de)

    src_vocab = build_vocab(train_src, vocab_size=8000)
    tgt_vocab = build_vocab(train_tgt, vocab_size=8000)

    train_loader = DataLoader(Seq2SeqDataset(train_src, train_tgt, src_vocab, tgt_vocab, seq_len), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(Seq2SeqDataset(val_src, val_tgt, src_vocab, tgt_vocab, seq_len), batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(Seq2SeqDataset(test_src, test_tgt, src_vocab, tgt_vocab, seq_len), batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader, src_vocab, tgt_vocab
