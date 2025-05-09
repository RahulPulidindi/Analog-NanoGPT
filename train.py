import os
import requests
import json
import torch
import random
import time
import psutil
import math
import wandb
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from torch.profiler import profile, ProfilerActivity, schedule
from aihwkit.optim import AnalogSGD

from src.data import CharTokenizer


def fetch_and_parse_squad_v2(split="train", data_dir="data/squad_v2"):
    """
    Downloads and parses the official SQuAD v2 JSON from GitHub.
    Returns a flat list of {'context': str} dicts.
    """
    os.makedirs(data_dir, exist_ok=True)
    filename = f"{split}-v2.0.json"
    url = (
        "https://raw.githubusercontent.com/rajpurkar/SQuAD-explorer/"
        "master/dataset/" + filename
    )
    path = os.path.join(data_dir, filename)

    # Download if missing
    if not os.path.exists(path):
        print(f"Downloading {filename}…")
        r = requests.get(url)
        r.raise_for_status()
        with open(path, "w") as f:
            json.dump(r.json(), f)

    # Load and flatten
    with open(path, "r") as f:
        squad = json.load(f)

    examples = []
    for article in squad["data"]:
        for para in article["paragraphs"]:
            ctx = para["context"]
            examples.append({"context": ctx})

    return examples


class SimpleSquadContextDataset(Dataset):
    def __init__(self, examples, tokenizer, max_length):
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ctx = self.examples[idx]["context"]
        ids = self.tokenizer.encode(ctx)

        # truncate / pad
        if len(ids) > self.max_length:
            ids = ids[: self.max_length]

        ids = torch.tensor(ids, dtype=torch.long)
        return {"input_ids": ids}


def train(
    model_name,
    model,
    num_epochs=3,
    batch_size=4,
    max_samples=1000,
    learning_rate=5e-4,
    val_frac=0.2
):
    # 1) fetch & parse as before
    all_examples = fetch_and_parse_squad_v2("train")

    print(all_examples[:1])

    # 2) limit if desired
    if max_samples and max_samples < len(all_examples):
        all_examples = all_examples[:max_samples]
    print(f"Training on {len(all_examples)} contexts")

    # 3) split into train / validation
    random.shuffle(all_examples)
    n_val = int(len(all_examples) * val_frac)
    val_examples = all_examples[:n_val]
    train_examples = all_examples[n_val:]
    print(f"Train contexts: {len(train_examples)}, Val contexts: {len(val_examples)}")

    # 4) build tokenizer on full corpus
    all_text = "".join(ex["context"] for ex in train_examples + val_examples)
    tokenizer = CharTokenizer(corpus=all_text)

    # 5) datasets & loaders
    max_len = getattr(model, "context_size", 32)
    train_ds = SimpleSquadContextDataset(train_examples, tokenizer, max_len)
    val_ds = SimpleSquadContextDataset(val_examples, tokenizer, max_len)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    # optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        betas=(0.9, 0.95),
        weight_decay=0.1,
    )

    if model_name == "Analog_NanoGPT":
        optimizer = AnalogSGD(model.parameters(), lr=learning_rate)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # 7) wandb setup
    wandb.init(
        project="nano-gpt-squad",
        name=model_name,
        config={
            "epochs": num_epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "val_frac": val_frac,
        },
    )
    wandb.watch(model, log="parameters", log_freq=50)

    # 8) start profiler
    profiler = profile(
        activities=[ProfilerActivity.CPU],
        record_shapes=True,
        schedule=schedule(wait=1, warmup=1, active=3, repeat=1),    
    )
    profiler.start()

    loss_fn = torch.nn.CrossEntropyLoss(reduction="sum")
    best_val_loss = float("inf")

    for epoch in range(1, num_epochs + 1):
        # -- training --
        model.train()
        total_loss = 0.0
        total_correct = 0
        total_tokens = 0
        epoch_start = time.time()

        for batch in tqdm(train_loader, desc=f"Epoch {epoch}/{num_epochs} [Train]"):
            input_ids = batch["input_ids"].to(device)
            outputs = model(input_ids)
            logits = getattr(outputs, "logits", outputs)

            if input_ids.size(1) > 1:
                shift_logits = logits[:, :-1, :].contiguous().view(-1, logits.size(-1))
                shift_labels = input_ids[:, 1:].contiguous().view(-1)

                batch_loss = loss_fn(shift_logits, shift_labels)
                total_loss += batch_loss.item()

                preds = shift_logits.argmax(dim=-1)
                total_correct += (preds == shift_labels).sum().item()
                total_tokens += shift_labels.numel()

                optimizer.zero_grad()
                batch_loss.backward()
                optimizer.step()

            profiler.step()

        epoch_time = time.time() - epoch_start
        avg_loss = total_loss / total_tokens
        train_acc = total_correct / total_tokens * 100.0
        train_ppl = math.exp(avg_loss)
        train_throughput = total_tokens / epoch_time  # tokens per second
        cpu_pct = psutil.cpu_percent()
        ram_pct = psutil.virtual_memory().percent

        print(f"Epoch {epoch} Train — Loss: {avg_loss:.4f}, Acc: {train_acc:.2f}%, "
              f"PPL: {train_ppl:.2f}, {train_throughput:.0f} tok/s, CPU: {cpu_pct:.1f}%, RAM: {ram_pct:.1f}%")
        
        wandb.log({
            "train/loss": avg_loss,
            "train/accuracy": train_acc,
            "train/perplexity": train_ppl,
            "train/tokens_per_sec": train_throughput,
            "resource/cpu_percent": cpu_pct,
            "resource/ram_percent": ram_pct,
            "epoch": epoch
        })

        # -- validation --
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_tokens = 0

        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch}/{num_epochs} [Val]"):
                input_ids = batch["input_ids"].to(device)
                outputs = model(input_ids)
                logits = getattr(outputs, "logits", outputs)

                if input_ids.size(1) > 1:
                    shift_logits = logits[:, :-1, :].contiguous().view(-1, logits.size(-1))
                    shift_labels = input_ids[:, 1:].contiguous().view(-1)

                    batch_loss = loss_fn(shift_logits, shift_labels)
                    val_loss += batch_loss.item()
                    preds = shift_logits.argmax(dim=-1)
                    val_correct += (preds == shift_labels).sum().item()
                    val_tokens += shift_labels.numel()

        avg_val_loss = val_loss / val_tokens
        val_acc = val_correct / val_tokens * 100.0
        val_ppl = math.exp(avg_val_loss)

        print(f"Epoch {epoch} Val   — Loss: {avg_val_loss:.4f}, Acc: {val_acc:.2f}%, PPL: {val_ppl:.2f}")
        wandb.log({
            "val/loss": avg_val_loss,
            "val/accuracy": val_acc,
            "val/perplexity": val_ppl,
            "epoch": epoch
        })

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), f"{model_name}_best.pt")
            wandb.save(f"{model_name}_best.pt")
            print(f"Saved new best model (val loss {best_val_loss:.4f})")

    # 9) stop & report profiler
    profiler.stop()
    profiler.export_chrome_trace("trace.json")
    print(profiler.key_averages().table(sort_by="cpu_time_total", row_limit=5))

    return model
