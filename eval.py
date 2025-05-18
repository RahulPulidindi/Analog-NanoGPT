import os
import json
import requests
from torch.utils.data import Dataset, DataLoader
import torch
from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss
from src.model.gpt_language_model.gpt import GPTLanguageModel
from src.data import CharTokenizer
from aihwkit.nn.conversion import convert_to_analog
from aihwkit.simulator.configs import InferenceRPUConfig, FloatingPointRPUConfig, SoftBoundsReferenceDevice, SingleRPUConfig
from aihwkit.simulator.configs import InferenceRPUConfig
from aihwkit.inference.noise.pcm import PCMLikeNoiseModel  # or other noise models
from aihwkit.simulator.configs.devices import IdealDevice
from aihwkit.simulator.tiles.inference import InferenceTileWithPeriphery
from data_utils import get_text_examples, prepare_dataloaders
import math

    
def compute_perplexity(
    analog_model_check: bool,
    checkpoint_path: str,
    max_length: int,
    tiny_cfg: dict,
    batch_size: int = 32,
    device: str = "cuda"
) -> float:
    """
    Load a GPTLanguageModel from tiny_cfg, restore weights from checkpoint_path,
    evaluate on eval_dataset, and return the perplexity.
    """
    # Initialize model
    print(f"Initializing model with config: {tiny_cfg}")
    model = GPTLanguageModel(**tiny_cfg).to(device)

    if analog_model_check:
        # First load with the same config used during training
        inference_rpu_config = InferenceRPUConfig()

        model = convert_to_analog(model, rpu_config=inference_rpu_config)
        
        # Load checkpoint
        print(f"Loading checkpoint from: {checkpoint_path}")
        ckpt = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(ckpt)


        model.eval()
        model.program_analog_weights()

    else:
        # Load checkpoint for digital model
        print(f"Loading checkpoint from: {checkpoint_path}")
        ckpt = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(ckpt)
        model.eval()

    # model.eval()

    # Prepare DataLoader
     # 1) fetch & parse as before
    train_examples, test_examples = get_text_examples()

    tokenizer = torch.load("tokenizer.pt")
    _, _, eval_loader = prepare_dataloaders(train_examples, test_examples, tokenizer, batch_size=16, max_length=max_length)

    criterion = torch.nn.CrossEntropyLoss(reduction="sum")
    total_loss = 0.0
    total_correct = 0
    total_tokens = 0

    with torch.no_grad():
        for batch in eval_loader:
            input_ids = batch["input_ids"].to(device)          # (B, L)
            outputs = model(input_ids)
            logits = getattr(outputs, "logits", outputs)       # (B, L, V)

            # shift tokens for next‐token prediction
            B, L, V = logits.size()
            shift_logits = logits[:, :-1, :].reshape(-1, V)    # (B*(L-1), V)
            shift_labels = input_ids[:, 1:].reshape(-1)        # (B*(L-1),)

            # accumulate loss
            loss = criterion(shift_logits, shift_labels)
            total_loss += loss.item()

            # compute token‐level accuracy
            preds = shift_logits.argmax(dim=-1)
            total_correct += (preds == shift_labels).sum().item()
            total_tokens += shift_labels.numel()

    avg_loss = total_loss / total_tokens
    perplexity = math.exp(avg_loss)
    accuracy = (total_correct / total_tokens) * 100.0
    print(f"Avg Loss:           {avg_loss:.2f}")
    print(f"Perplexity:         {perplexity:.2f}")
    return perplexity
