# Qwen3 Fine-tuning for Token Classification

This project implements fine-tuning of Qwen3 models for token classification tasks using LoRA (Low-Rank Adaptation) and QLoRA (Quantized LoRA) techniques. The goal is to compare the performance of these two approaches in terms of precision, recall, and F1-score.

## Overview

Token classification is a fundamental NLP task that involves assigning labels to individual tokens in a sequence. This project explores how different fine-tuning techniques affect the performance of Qwen3 models on such tasks.

### Techniques Compared

- **LoRA (Low-Rank Adaptation)**: A parameter-efficient fine-tuning method that adds small rank decomposition matrices to existing weights
- **QLoRA (Quantized LoRA)**: An extension of LoRA that uses 4-bit quantization to reduce memory usage while maintaining performance

## Features

- Fine-tuning Qwen3 models for token classification
- Implementation of both LoRA and QLoRA techniques
- Performance comparison (Precision, Recall, F1-score)
- Memory-efficient training with gradient checkpointing
- Comprehensive evaluation metrics
- Support for various token classification datasets

## Requirements

- Python 3.8+ (Currently does not support Python 3.13)
- PyTorch 2.0+
- Transformers library
- PEFT (Parameter-Efficient Fine-tuning)
- Datasets library
- Evaluate library
- Accelerate library

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd qwen-fine-tuning

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Training with LoRA

```bash
python train_lora.py \
    --model_name "Qwen/Qwen2.5-7B" \
    --dataset_name "your_dataset" \
    --output_dir "./outputs/lora" \
    --lora_r 16 \
    --lora_alpha 32 \
    --learning_rate 2e-4
```

### Training with QLoRA

```bash
python train_qlora.py \
    --model_name "Qwen/Qwen2.5-7B" \
    --dataset_name "your_dataset" \
    --output_dir "./outputs/qlora" \
    --lora_r 16 \
    --lora_alpha 32 \
    --learning_rate 2e-4 \
    --bits 4
```

### Evaluation

```bash
python evaluate.py \
    --model_path "./outputs/lora" \
    --test_dataset "your_test_dataset" \
    --output_file "results_lora.json"
```

## Project Structure

```
qwen-fine-tuning/
├── README.md
├── requirements.txt
├── train_lora.py          # LoRA fine-tuning script
├── train_qlora.py         # QLoRA fine-tuning script
├── evaluate.py            # Evaluation script
├── utils/
│   ├── data_utils.py      # Data preprocessing utilities
│   ├── model_utils.py     # Model configuration utilities
│   └── metrics.py         # Evaluation metrics
├── configs/
│   ├── lora_config.py     # LoRA configuration
│   └── qlora_config.py    # QLoRA configuration
├── outputs/               # Training outputs and checkpoints
└── results/              # Evaluation results
```

## Configuration

### LoRA Parameters

- `lora_r`: Rank of the low-rank matrices (default: 16)
- `lora_alpha`: Scaling factor for LoRA weights (default: 32)
- `lora_dropout`: Dropout probability for LoRA layers (default: 0.1)

### QLoRA Parameters

- `bits`: Quantization bits (default: 4)
- `double_quant`: Use double quantization (default: True)
- `compute_dtype`: Computation dtype (default: torch.float16)

## Evaluation Metrics

The project evaluates models using standard token classification metrics:

- **Precision**: Ratio of correctly predicted positive observations to total predicted positives
- **Recall**: Ratio of correctly predicted positive observations to all actual positives
- **F1-Score**: Harmonic mean of precision and recall
- **Accuracy**: Overall accuracy across all tokens

## Results

Results will be stored in the `results/` directory with detailed comparisons between LoRA and QLoRA approaches, including:

- Performance metrics for each technique
- Memory usage comparison
- Training time analysis
- Model size comparison

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
