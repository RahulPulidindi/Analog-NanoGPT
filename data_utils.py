import os
import zipfile
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import numpy as np
from src.data import CharTokenizer
from torch.nn.utils.rnn import pad_sequence
import random

def find_file(dirpath, basename):
    """
    Look for basename + ext in {'.txt', '.text'}; return first match or None.
    """
    for ext in (".txt", ".text"):
        p = os.path.join(dirpath, basename + ext)
        if os.path.isfile(p):
            return p
    return None

def load_lines(path, max_lines=None):
    """
    Load lines, skipping blank lines and Wikipedia headings (== Section ==).
    """
    lines = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("="):
                continue
            lines.append(line)
            if max_lines and len(lines) >= max_lines:
                break
    return lines

def get_text_examples(data_dir = "/content/data"):
    train_path = find_file(data_dir, "train")
    test_path  = find_file(data_dir, "test")

    if not train_path or not test_path:
        raise FileNotFoundError("Couldn't find train/test files in " + data_dir)
    
    train_lines = load_lines(train_path)
    test_lines  = load_lines(test_path)

    return train_lines, test_lines

def pad_collate(batch):
    # If your Dataset returns {"input_ids": LongTensor}, pull them out:
    input_ids = [b["input_ids"] for b in batch]
    # Pad to the max length in this batch:
    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=0)
    return {"input_ids": input_ids}


class WikiText2Dataset(Dataset):
    """Wraps a list of raw text lines for char-level tokenization."""
    def __init__(self, lines, tokenizer, max_length):
        self.lines = lines
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.lines)

    def __getitem__(self, idx):
        text = self.lines[idx]
        ids = self.tokenizer.encode(text)
        if len(ids) > self.max_length:
            ids = ids[: self.max_length]
        return {"input_ids": torch.tensor(ids, dtype=torch.long)}
    

def prepare_dataloaders(train_examples, test_examples, tokenizer, max_length, batch_size=32,
                        val_frac=0.1):
    """
    Split `examples` (list of {'text': ...}) into train/val/test and return 3 DataLoaders.
    """
    
    np.random.shuffle(train_examples)

    n = len(train_examples)
    n_val  = int(n * val_frac)

    val_examples   = train_examples[:n_val]
    train_examples = train_examples[n_val:]

    train_ds = WikiText2Dataset(train_examples, tokenizer, max_length)
    val_ds   = WikiText2Dataset(val_examples,   tokenizer, max_length)
    test_ds  = WikiText2Dataset(test_examples,  tokenizer, max_length)

    train_loader = DataLoader(train_ds, batch_size=batch_size, collate_fn=pad_collate,shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, collate_fn=pad_collate,shuffle=False)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, collate_fn=pad_collate,shuffle=False)

    return train_loader, val_loader, test_loader