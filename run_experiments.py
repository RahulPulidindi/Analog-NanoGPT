import os
import json
import torch
from itertools import product
from train import trainv2
from eval import compute_perplexity
from src.model.gpt_language_model.gpt import GPTLanguageModel
from aihwkit.nn.conversion import convert_to_analog
from aihwkit.simulator.configs import InferenceRPUConfig
import pandas as pd
from datetime import datetime
import wandb

# Define all configurations to test
CONFIGURATIONS = {
    "num_heads": [2, 4, 8],
    "context_size": [32, 64, 128],
    "embeddings_size": [128, 256, 512],
    "num_layers": [4, 8, 12],
    "bias": [True, False],
    "dropout": [0.1, 0.2, 0.3],
    "vocab_size": [50257]
}

# Training parameters
TRAIN_PARAMS = {
    "num_epochs": 200,
    "batch_size": 16,
    "learning_rate": 3e-4,
    "val_frac": 0.2
}

def create_model_config(num_heads, context_size, embeddings_size, num_layers, bias, dropout, vocab_size):
    """Create a model configuration dictionary."""
    return {
        "vocab_size": vocab_size,
        "embeddings_size": embeddings_size,
        "context_size": context_size,
        "head_size": None,
        "num_heads": num_heads,
        "feed_forward_scaling": 2,
        "num_layers": num_layers,
        "bias": bias,
        "dropout": dropout,
    }

def train_and_evaluate_config(config, device="cuda"):
    """Train and evaluate both digital and analog models for a given configuration."""
    num_heads = config["num_heads"]
    context_size = config["context_size"]
    embeddings_size = config["embeddings_size"]
    num_layers = config["num_layers"]
    bias = config["bias"]
    dropout = config["dropout"]
    vocab_size = config["vocab_size"]
    
    # Create unique model names based on configuration
    config_str = f"{num_heads}_{context_size}_{embeddings_size}_{num_layers}_{bias}_{dropout}_{vocab_size}"
    digital_model_name = f"Digital_NanoGPT_{config_str}"
    analog_model_name = f"Analog_NanoGPT_{config_str}"
    
    # Create model configuration
    model_config = create_model_config(num_heads, context_size, embeddings_size, num_layers, bias, dropout, vocab_size)
    
    # Initialize and train digital model
    print(f"\nTraining digital model with config: {config_str}")
    digital_model = GPTLanguageModel(**model_config).to(device)
    trainv2(digital_model_name, digital_model, **TRAIN_PARAMS)
    wandb.finish()
    
    # Convert to analog and train
    print(f"\nTraining analog model with config: {config_str}")
    analog_model = convert_to_analog(digital_model, rpu_config=InferenceRPUConfig())
    trainv2(analog_model_name, analog_model, **TRAIN_PARAMS)
    wandb.finish()
    
    # Evaluate both models
    print(f"\nEvaluating models with config: {config_str}")
    digital_perplexity = compute_perplexity(
        False, 
        f"{digital_model_name}_best.pt",
        model_config
    )
    
    analog_perplexity = compute_perplexity(
        True,
        f"{analog_model_name}_best.pt",
        model_config
    )
    
    return {
        "config": config_str,
        "num_heads": num_heads,
        "context_size": context_size,
        "embeddings_size": embeddings_size,
        "num_layers": num_layers,
        "bias": bias,
        "dropout": dropout,
        "vocab_size": vocab_size,
        "digital_perplexity": digital_perplexity,
        "analog_perplexity": analog_perplexity,
        "perplexity_ratio": analog_perplexity / digital_perplexity
    }

def main():
    # Create results directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = f"experiment_results_{timestamp}"
    os.makedirs(results_dir, exist_ok=True)
    
    # Generate all configurations
    configs = [
        {"num_heads": h, "context_size": c, "embeddings_size": e, "num_layers": l, "bias": b, "dropout": d, "vocab_size": v}
        for h, c, e, l, b, d, v in product(
            CONFIGURATIONS["num_heads"],
            CONFIGURATIONS["context_size"],
            CONFIGURATIONS["embeddings_size"],
            CONFIGURATIONS["num_layers"],
            CONFIGURATIONS["bias"],
            CONFIGURATIONS["dropout"],
            CONFIGURATIONS["vocab_size"]
        )
    ]
    
    # Run experiments
    results = []
    for config in configs:
        try:
            result = train_and_evaluate_config(config)
            results.append(result)
            
            # Save intermediate results
            df = pd.DataFrame(results)
            df.to_csv(f"{results_dir}/results.csv", index=False)
            
            # Save detailed results as JSON
            with open(f"{results_dir}/detailed_results.json", "w") as f:
                json.dump(results, f, indent=2)
                
        except Exception as e:
            print(f"Error in configuration {config}: {str(e)}")
            continue
    
    # Generate summary tables
    df = pd.DataFrame(results)
        
    print(f"\nExperiment results saved in {results_dir}/")


if __name__ == "__main__":
    main() 