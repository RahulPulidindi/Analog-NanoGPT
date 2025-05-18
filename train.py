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
import gc
from data_utils import prepare_dataloaders, get_text_examples

from src.data import CharTokenizer

def trainv2(
    model_name,
    model,
    max_length,
    num_epochs=3,
    batch_size=4,
    max_samples=1000,
    learning_rate=5e-4,
    val_frac=0.2
):
    # 1) fetch & parse as before
    train_examples, test_examples = get_text_examples()

    train_corpus = "\n".join(train_examples)
    test_corpus = "\n".join(test_examples)

    tokenizer = CharTokenizer(corpus=train_corpus)
    torch.save(tokenizer, "tokenizer.pt")
    train_loader, val_loader, _ = prepare_dataloaders(train_examples, test_examples, tokenizer, max_length=max_length)

    # 6) optimizer & hooks for analog
    is_analog = model_name.startswith("Analog")
    if is_analog:
        # clear CUDA cache and collect garbage to free memory
        torch.cuda.empty_cache()
        gc.collect()
        optimizer = AnalogSGD(model.parameters(), lr=learning_rate)
    else:
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            betas=(0.9, 0.95),
            weight_decay=0.1,
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # 7) wandb setup
    wandb.init(
        project="experiments",
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