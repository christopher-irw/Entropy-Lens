# EntropyLens
EntropyLens is a research framework for analyzing entropy profiles in transformer models. It enables model-agnostic extraction of entropy metrics from frozen, off-the-shelf transformers, providing insights into model computation patterns, prompt and task identification, and output correctness. The framework is designed for reproducible experiments and does not require fine-tuning or access to model internals.

The source code in this repository includes modules for entropy computation (`model_inspector`), model management (`model_manager.py`), and scripts for running experiments and analyses. Notebooks and scripts are provided for clustering models, task identification, and format classification. Example commands and usage instructions are included to help you reproduce the main experiments and explore entropy-based analysis on various transformer models.

## Installation

1. **Clone or Copy the Repository**  
    Copy the entire folder to your desired location.

2. **Create a Virtual Environment**  
    ```bash
    python3 -m venv .venv
    source .venv/bin/activate
    ```

3. **Install Dependencies**  
    ```bash
    pip install -r requirements.txt
    ```

## EntropyLens Overview
All the code necessary to calucalte entropy profiles on different models can be found in the `model_inspector` module. This module provides functions to compute entropy metrics on model outputs, which can then be used for further analysis. Moreover, the `model_manager.py` module handles model loading and management tasks.

## Experiments

### 1. Model family clustering
All the code necessary to analyse and generate the TSNE clustering can be found in the `get_model_entropy.ipynb` notebook. This notebook will guide you through the process of clustering different model families based on their entropy profiles.

### 2. Task identification experiments
For experiments with the TinyStories dataset, you can run the following command to generate entropy profiles for the `gemma-2-2b-it` model:

```bash
python3 run_experiments_ts.py "gemma-2-2b-it" tinystories --update_steps 1000 --max_len 8 --n_stories 100
python3 run_tests.py results/gemma-2-2b-it_tinystories_100.pkl --bs 8 --label "prompt"
```

### 3. Correct task execution experiments
For experiments with the MMLU dataset, you can run the following command to generate entropy profiles for the `gemma-2-2b-it` model:


```bash
python3 run_experiments.py "gemma-2-2b-it" mmlu --update_steps 1000 --max_len 8 --prompt_type base
python3 run_tests.py results/gemma-2-2b-it_mmlu_base.pkl --bs 8
```


### 4. Format classification experiments
For experiments with format classification, you can run the following command to generate the dataset and train a classifier:

```bash
python3 generate_format_dataset.py
python3 classify_formats.py -d data/gemma/formats.pkl -a 0.5 1 5 -k 5
```

---