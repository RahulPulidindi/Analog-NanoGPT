# IBM Project: Layer-Wise Heterogeneous Mapping Using Mathematical Sensitivity Analysis

## Team Information

-   **Members**:
    -   **Rahul Pulidindi (UNI: rp3254)**
    -   **Priya Deshpande (UNI: ppd2119)**
-   **Mentor: Dr. Hadjer Benmeziane, IBM Research**

---


## Setting up the environment 

- Ensure you're running Python 3.10
- Clone the following repo: https://github.com/Andrei-Aksionov/nanoGPTplus.git
- **Adjust dependencies**  for compatibility between aihwkit and NanoGPT

    In `pyproject.toml`, comment out the following under `[tool.poetry.dependencies]`:

    ```toml
    # numpy = "*"
    # pandas = "*"
    # python = "..."
    # requests = "*"
    # torch = "*"
    # transformers = "*"
    ```

    Run pip install -e . to install nanoGPT dependencies after config changes. 
- Install other required dependencies
    ``` pip install torch==2.4.1```,
    ```pip install numpy==2.2.5```,
    ```pip install wandb```,
    ```pip install pandas```

- Install awhwkit-gpu
- Download dataset (wikiText-2) from https://www.kaggle.com/datasets/vivekmettu/wikitext2-data?resource=download and unzip.
- Change configurations to be run in run_experiments.py
```
    CONFIGURATIONS = {
    "num_heads": [2, 4, 8],
    "context_size": [32, 64, 128],
    "embeddings_size": [128, 256, 512],
    "num_layers": [4, 8, 16],
    "bias": [True, False],
    "dropout": [0.1, 0.2, 0.3],
    "vocab_size": [50257]
    }
```

- Run run_experiments.py --> (```python run_experiments.py```)
- Results stored in /experiment_results_{date}_{time_stamp}/results.csv

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

### Table 1: Varying # Heads

| # Heads | Model Size (M) | Δ Loss | Δ Perplexity |
| :-----: | :------------: | :----: | :----------: |
|    2    |     14.97      |  0.79  |    301.47    |
|    4    |     14.97      |  0.46  |    170.71    |
|    8    |     14.97      |  0.83  |    393.83    |

---

### Table 2: Varying Context Size

| Context | Model Size (M) | Δ Loss | Δ Perplexity |
| :-----: | :------------: | :----: | :----------: |
|   32    |     14.97      |  0.30  |    30.80     |
|   64    |     14.97      |  0.13  |    49.01     |
|   128   |     14.97      |  0.46  |    170.71    |

---

### Table 3: Varying Embedding Size

| Embed | Model Size (M) | Δ Loss | Δ Perplexity |
| :---: | :------------: | :----: | :----------: |
|  128  |      6.96      |  0.37  |    90.50     |
|  256  |     14.97      |  0.46  |    170.71    |
|  512  |     34.14      |  0.01  |     5.80     |

---

### Table 4: Span Analysis

|           Sweep           | Span Analog Loss | Span Analog PPL |
| :-----------------------: | :--------------: | :-------------: |
|      Heads (2, 4, 8)      |       0.41       |     235.41      |
|   Context (32, 64, 128)   |       1.36       |     344.37      |
| Embedding (128, 256, 512) |       0.58       |     226.86      |

---

### Table 5: Analog / Digital Loss Ratio

| # Heads | Context | Embed | Analog / Digital Loss Ratio |
| :-----: | :-----: | :---: | :-------------------------: |
|    2    |   128   |  256  |            0.88             |
|    4    |   32    |  256  |            0.94             |
|    4    |   64    |  256  |            0.98             |
|    4    |   128   |  128  |            0.94             |
|    4    |   128   |  256  |            0.93             |
|    4    |   128   |  512  |            1.00             |
|    8    |   128   |  256  |            0.87             |

---

## 4. Reproducibility Instructions

### A. Train Digital & Analog Models

1. **Open the training notebook**  
   Launch `Experiments.ipynb` in your Jupyter environment.

2. **Prepare the codebase**  
   Copy the following into your working directory:

    - `main.py`
    - `model.py`
    - `train.py`
    - the entire `config/` folder

3. **Adjust dependencies**  
   In `pyproject.toml`, comment out the following under `[tool.poetry.dependencies]`:

    ```toml
    # numpy = "*"
    # pandas = "*"
    # python = "..."
    # requests = "*"
    # torch = "*"
    # transformers = "*"
    ```

4. **Proceed with installations**
5. **Configure model params in** `nanogpt_config.py`.
6. **Run the training script**
    - In a new notebook cell: `conda run -n py310 python -u main.py`
    - This will sequentially train both the digital and analog NanoGPT models using the settings defined in `main.py` and `train.py`.

### B. Evaluation

1. **Obtain model checkpoints**
    - Ensure that both the digital and analog model checkpoints (\*.pt files) have been saved in your working directory by the training step.
2. **Open the evaluation notebook**
    - Launch `Eval.ipynb`.
3. **Repeat dependency setup**
    - Use the same environment and installation steps as above.
4. **Prepare the codebase**
    - Copy `eval.py` into your working directory.
    - Place your digital and analog checkpoint files alongside `eval.py`.
5. **Specify model configuration**
    - In `eval.py`, update the `tiny_cfg` dictionary to match the exact architecture parameters of the checkpoint you wish to evaluate.
6. **Select evaluation mode**
    - **Call `compute_perplexity(analog_model_check, checkpoint_path, tiny_cfg, ...)` with**:
        - `analog_model_check = False` for the digital checkpoint
        - `analog_model_check = True` for the analog checkpoint
7. **Run the evaluation script**
    - In a new notebook cell: `conda run -n py310 python eval.py`
    - The script will output the average cross-entropy loss and perplexity for the selected model.

---

### E. Quickstart: Minimum Reproducible Result

To reproduce our minimum reported result (≈ 95–98% digital accuracy retention on analog), run:

```bash
# Step 1: Set up evaluation environment

# Step 2: Load analog and digital model checkpoints

# Step 3: Evaluate both models
conda run -n py310 python eval.py

```

---

## 5. Notes

-   **All code lives in the root of the repo**:
    -   `main.py`, `model.py`, `train.py`, `eval.py`
    -   Configuration files in `config/(nanogpt_config.py, rpu_config.py)`
-   **Jupyter notebooks for interactive analysis**:
    -   `Experiments.ipynb` (training workflow)
    -   `Eval.ipynb` (evaluation workflow)
-   **Checkpoints are saved within the notebook**:
    -   `Digital_NanoGPT_best.pt`
    -   `Analog_NanoGPT_best.pt`
-   **Contact & Support**:
    -   Rahul Pulidindi (rp3254@columbia.edu)
    -   Priya Deshpande (ppd2119@columbia.edu)
