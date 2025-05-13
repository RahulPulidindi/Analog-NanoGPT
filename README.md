# IBM Project: Layer-Wise Heterogeneous Mapping Using Mathematical Sensitivity Analysis

## Team Information

-   **Members**:
    -   Rahul Pulidindi (UNI: rp3254)
    -   Priya Deshpande (UNI: ppd2119)

---

## 1. Problem Statement

We aim to develop and evaluate a layer-wise sensitivity tool for transformer-based models on analog in-memory computing (AIMC) hardware. Our goals are:

-   Train a small GPT (NanoGPT) model on SQuAD v2.0 passages for next-token prediction.
-   Convert the digital model to analog using AIHWKit and fine-tune for a few epochs.
-   Measure per-layer vulnerability to device noise and quantization.
-   Extend AnalogNAS-Bench to include transformer-specific RPU hyperparameters, guided by sensitivity scores.
-   Compare digital vs. analog performance and simulate inference energy/latency on various RPU configurations.

---

## 2. Model Description

-   **Architecture**: NanoGPT (`GPTLanguageModel`) with configurable `tiny_cfg`:

    ```python
    tiny_cfg = {
        "vocab_size": 50257,
        "embeddings_size": 256,
        "context_size": 128,
        "head_size": None,
        "num_heads": 4,
        "feed_forward_scaling": 2,
        "num_layers": 4,
        "bias": True,
        "dropout": 0.1,
    }
    ```

    -   This default setup yields ~15M parameters and runs efficiently on CPU/GPU for rapid prototyping.

-   Framework: PyTorch 2.4.1
-   Analog Conversion: AIHWKit v0.9.2
-   Optimizer: AdamW for digital; AnalogSGD for analog
-   **Custom Components**:
    -   `SimpleSquadContextDataset` for SQuAD context slices

---

## 3. Final Results Summary

Example Table:

| Metric               | Value                           |
| -------------------- | ------------------------------- |
| Final Top-1 Accuracy | XX.XX%                          |
| Inference Latency    | XX.XX ms                        |
| Model Size           | XX MB                           |
| Peak Memory Use      | XX MB                           |
| Training Time/Epoch  | XX s                            |
| Device               | A100, Jetson Nano, M1 Pro, etc. |

---

## 4. Reproducibility Instructions

### A. Requirements

Install dependencies:

```bash
pip install -r requirements.txt
```

---

B. Wandb Dashboard

View training and evaluation metrics here: Wandb Dashboard Link
(Replace with actual link)

---

### C. Specify for Training or For Inference or if Both

To train the model from scratch:

```bash
python train.py --config configs/default.yaml
```

---

### D. Evaluation

To evaluate the trained model:

```bash
python eval.py --weights checkpoints/best_model.pth
```

---

### E. Quickstart: Minimum Reproducible Result

To reproduce our minimum reported result (e.g., XX.XX% accuracy), run:

```bash
# Step 1: Set up environment
pip install -r requirements.txt

# Step 2: Download dataset
bash scripts/download_dataset.sh  # if applicable

# Step 3: Run training (or skip if checkpoint is provided)
python train.py --config configs/default.yaml

# Step 4: Evaluate
python eval.py --weights checkpoints/best_model.pth
```

---

## 5. Notes (up to you)

-   All scripts are located in `scripts/`, `train.py`, `eval.py`, and `configs/`.
-   Trained Model are saved in `models/`.
-   Contact information
