# Fine-Tune a Generative AI Model for Dialogue Summarization

This notebook demonstrates how to fine-tune a generative AI model (FLAN-T5) for dialogue summarization using both full fine-tuning and Parameter Efficient Fine-Tuning (PEFT) techniques.

## Overview

The notebook walks through the complete process of:
- Loading and preprocessing dialogue data
- Testing baseline model performance
- Performing full fine-tuning
- Implementing PEFT/LoRA for efficient training
- Evaluating model performance using ROUGE metrics

## Requirements

### System Requirements
- **Instance Type**: ml.m5.2xlarge (8 vCPUs, 32 GiB RAM)
- **Python**: 3.12+

### Dependencies
```bash
pip install tensorflow==2.18.0 keras==3.9.0
pip install torch==2.5.1 torchdata==0.6.0
pip install datasets==2.17.0 transformers==4.38.2 evaluate==0.4.0 rouge_score==0.1.2 peft==0.3.0
```

## Dataset

- **Source**: [DialogSum](https://huggingface.co/datasets/knkarthick/dialogsum)
- **Size**: 10,000+ dialogues with manual summaries
- **Splits**: Train (12,460), Validation (500), Test (1,500)

## Model

- **Base Model**: `google/flan-t5-base`
- **Parameters**: 247M trainable parameters
- **Task**: Sequence-to-sequence dialogue summarization

## Notebook Structure

### 1. Setup and Data Loading
- Environment verification and dependency installation
- Dataset and model loading
- Baseline model testing with zero-shot inference

### 2. Full Fine-Tuning
- Data preprocessing and tokenization
- Training configuration and execution
- Qualitative and quantitative evaluation

### 3. Parameter Efficient Fine-Tuning (PEFT)
- LoRA adapter setup and training
- Performance comparison with full fine-tuning
- ROUGE metric evaluation

## Key Features

- **Zero-shot baseline testing** for performance comparison
- **Full fine-tuning** with complete parameter updates
- **PEFT/LoRA** for memory-efficient training
- **ROUGE evaluation** for quantitative assessment
- **Human evaluation** for qualitative analysis

## Results

The notebook demonstrates:
- Improved summarization quality after fine-tuning
- Comparison between original and fine-tuned model outputs
- ROUGE score improvements across different metrics
- Efficiency gains with PEFT approaches

## File Structure

```
├── Lab_2_fine_tune_generative_ai_model.ipynb  # Main notebook
├── flan-dialogue-summary-checkpoint/           # Pre-trained checkpoint
└── dialogue-summary-training-*/                # Training outputs
```

## Notes

- Training is limited to 1 step for demonstration purposes
- Full training requires significant computational resources
- Pre-trained checkpoints are provided for faster experimentation
- ROUGE metrics provide quantitative evaluation of summarization quality

## Troubleshooting

- Ensure correct instance type (ml.m5.2xlarge) for memory requirements
- Check TOKENIZERS_PARALLELISM environment variable if encountering fork warnings
- Verify all dependencies are installed with correct versions"# Fine-tuning-Dialogue-Summarization-using-LoRa" 
