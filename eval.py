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
import math

def fetch_and_parse_squad_v2(split="dev", data_dir="data/squad_v2"):
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
    
def compute_perplexity(
    analog_model_check: bool,
    checkpoint_path: str,
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
        
        # Now convert to inference mode
        # inference_rpu_config = InferenceRPUConfig(
        #     noise_model=PCMLikeNoiseModel(g_max=25.0)
        # )

        # inference_rpu_config = InferenceRPUConfig(
        #     noise_model=PCMLikeNoiseModel(
        #         prog_noise_scale=0.2,
        #         read_noise_scale=0.3,
        #         drift_noise_scale=0.1,
        #         drift_scale=0.2,
        #         drift_nu=0.4,
        #     )
        # )

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
    all_examples = fetch_and_parse_squad_v2("dev")


    all_text = "".join(ex["context"] for ex in all_examples)
    tokenizer = CharTokenizer(corpus=all_text)
    max_len = tiny_cfg["context_size"]
    val_ds = SimpleSquadContextDataset(all_examples, tokenizer, max_len)
    eval_loader = DataLoader(val_ds, batch_size=16, shuffle=False)

    loss_fn = torch.nn.CrossEntropyLoss(reduction="sum")

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

tiny_cfg = {
    "vocab_size":           50257,   
    "embeddings_size":      256,    
    "context_size":         128,     
    "head_size":            None,    
    "num_heads":            4,       
    "feed_forward_scaling": 2,       
    "num_layers":           4,       
    "bias":                 True,
    "dropout":              0.1,
}

compute_perplexity(True, "/content/Analog_NanoGPT_4_128_256_best.pt", tiny_cfg=tiny_cfg)
