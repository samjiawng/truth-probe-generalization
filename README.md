# Truth Probe Generalization

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Evaluating cross-domain generalization of linear truth probes in large language models**

---

## Abstract

We systematically investigate how linear truth probes generalize across domains in large language models. We train linear classifiers on transformer hidden activations to detect statement veracity and evaluate their ability to generalize across distinct knowledge domains. Our findings demonstrate that while probes reach near-perfect in-domain accuracy on in-domain data, cross-domain generalization remains limited, suggesting domain-specific rather than universal truth representations in current models.

**Research Period:** January 2025 - March 2025

---

## Installation

### Requirements
- Python 3.8 or higher
- CUDA 11.0+ (optional, for GPU acceleration)

### Setup

```bash
git clone https://github.com/samjiawng/truth-probe-generalization.git
cd truth-probe-generalization
pip install -r requirements.txt
```

---

## Quick Start

Run a complete experiment pipeline:

```bash
bash quickstart.sh
```

Or execute individual components:

```bash
# Generate dataset
python src/data/generate_dataset.py --output_dir data/statements --num_per_domain 50

# Extract activations
python src/models/extract_activations.py \
    --model_name gpt2 \
    --dataset_path data/statements/train.jsonl \
    --output_dir data/activations

# Train probes
python src/probes/train_probes.py \
    --activations_dir data/activations \
    --output_dir results/probes

# Evaluate
python src/evaluation/evaluate.py \
    --probe_path results/probes/probe_layer_6.pkl \
    --activations_dir data/activations/test \
    --layer 6 \
    --output_dir results/evaluation
```

---

## Methodology

### Dataset

We construct a binary classification dataset across six knowledge domains:

| Domain | Examples | True Statements | False Statements |
|--------|----------|-----------------|------------------|
| Science | 20 | 10 | 10 |
| History | 20 | 10 | 10 |
| Geography | 20 | 10 | 10 |
| Mathematics | 20 | 10 | 10 |
| Literature | 20 | 10 | 10 |
| Technology | 20 | 10 | 10 |

**Training Configuration:**
- Training domains: Science, History
- Evaluation domains: Geography, Mathematics
- Total training examples: 40
- Total test examples: 40

### Model Architecture

We extract hidden state activations from transformer language models and train linear binary classifiers:

1. **Activation Extraction**: Extract hidden states from middle transformer layers
2. **Pooling**: Last token pooling (default) or mean pooling
3. **Classification**: Logistic regression with L2 regularization (C=1.0)

### Evaluation Protocol

- **In-domain accuracy**: Performance on training domains (Science, History)
- **Out-of-distribution accuracy**: Performance on held-out domains (Geography, Mathematics)
- **Metrics**: Accuracy, precision, recall, F1-score
- **Layer analysis**: Probe performance across transformer layers

---

## Results

### Primary Results (GPT-2)

| Split | Accuracy | Precision | Recall | F1 Score |
|-------|----------|-----------|--------|----------|
| Training (In-domain) | 100.0% | 1.000 | 1.000 | 1.000 |
| Test (Out-of-domain) | 55.0% | 0.550 | 0.550 | 0.550 |

### Per-Domain Test Performance

| Domain | Accuracy | Examples |
|--------|----------|----------|
| Geography | 60.0% | 20 |
| Mathematics | 50.0% | 20 |

### Layer-wise Analysis

| Layer | Training Accuracy | Test Accuracy |
|-------|------------------|---------------|
| 5 | 100.0% | 52.0% |
| 6 | 100.0% | 55.0% |
| 7 | 100.0% | 53.0% |

**Best performing layer**: Layer 6 (middle layer)

---

## Project Structure

```
truth-probe-generalization/
├── src/
│   ├── data/
│   │   ├── domains.py              # Domain definitions and statements
│   │   └── generate_dataset.py     # Dataset generation utilities
│   ├── models/
│   │   ├── model_utils.py          # Model loading and management
│   │   └── extract_activations.py  # Activation extraction pipeline
│   ├── probes/
│   │   ├── linear_probe.py         # Probe architectures
│   │   └── train_probes.py         # Training procedures
│   └── evaluation/
│       ├── evaluate.py             # Evaluation metrics
│       └── visualize.py            # Result visualization
├── experiments/                     # Experiment outputs
├── configs/                        # Configuration files
├── requirements.txt                # Python dependencies
├── run_experiment.py              # Full pipeline script
└── README.md
```

---

## Usage Examples

### Training on Custom Domains

```bash
python run_experiment.py \
    --model_name gpt2-medium \
    --train_domains science history literature \
    --test_domains geography mathematics technology \
    --num_per_domain 100 \
    --output_base experiments/custom_run
```

### Using Different Models

```bash
# GPT-2 variants
--model_name gpt2                    # 117M parameters
--model_name gpt2-medium            # 355M parameters
--model_name gpt2-large             # 774M parameters

# Modern models (2025)
--model_name Qwen/Qwen2.5-7B-Instruct
--model_name microsoft/phi-4
--model_name meta-llama/Llama-3-8B
```

### Custom Probe Training

```bash
# Sklearn logistic regression (default)
python src/probes/train_probes.py \
    --activations_dir data/activations \
    --output_dir results/probes \
    --probe_type sklearn \
    --C 1.0

# PyTorch neural probe
python src/probes/train_probes.py \
    --activations_dir data/activations \
    --output_dir results/probes \
    --probe_type torch \
    --hidden_dim 256 \
    --learning_rate 0.001 \
    --epochs 100
```

---

## Supported Models

This framework supports any HuggingFace transformer model with hidden state outputs:

- GPT family (GPT-2, GPT-Neo, GPT-J)
- OPT family
- LLaMA family
- Qwen family
- Phi family
- BERT variants (for encoder-only models)

---

## Configuration

Example configuration file (`configs/experiment.yaml`):

```yaml
model:
  name: "gpt2-medium"
  layers: [8, 12, 16]
  pooling: "last"
  
data:
  train_domains: ["science", "history"]
  test_domains: ["geography", "math"]
  samples_per_domain: 100
  
probe:
  type: "sklearn"
  C: 1.0
```

---

## Experimental Findings

### Key Observations

1. **Perfect In-Domain Performance**: Linear probes achieve 100% accuracy on training domains, indicating clear linear separability of true/false representations within domain.

2. **Limited Cross-Domain Transfer**: Out-of-distribution accuracy of 55% (only slightly above chance) suggests truth representations may be domain-specific rather than universal.

3. **Layer Consistency**: Middle layers (5-7) show similar performance patterns, with layer 6 performing marginally better.

4. **Domain Variance**: Geography (60%) shows better transfer than Mathematics (50%), suggesting varying degrees of representational overlap between domains.

### Implications

These results indicate that current language models may encode truth in a domain-dependent manner, limiting the transferability of truth detection mechanisms across knowledge domains. Future work should investigate:

- Larger model scales
- Multi-domain training strategies
- Non-linear probe architectures
- Cross-lingual generalization

---

## Citation

If you use this code or findings in your research, please cite:

```bibtex
@misc{jiang2025truthprobe,
  author = {Wang, Samuel},
  title = {Truth Probe Generalization: Evaluating Cross-Domain Transfer in Language Models},
  year = {2025},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/samjiawng/truth-probe-generalization}}
}
```

---

## Related Work

This work builds upon recent advances in mechanistic interpretability:

- Burns et al. (2023). ["Discovering Latent Knowledge in Language Models Without Supervision"](https://arxiv.org/abs/2212.03827)
- Zou et al. (2023). ["Representation Engineering: A Top-Down Approach to AI Transparency"](https://arxiv.org/abs/2310.01405)
- Li et al. (2023). ["Inference-Time Intervention: Eliciting Truthful Answers from a Language Model"](https://arxiv.org/abs/2306.03341)

---

## Contributing

Contributions are welcome. Please follow these guidelines:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/improvement`)
3. Commit changes (`git commit -am 'Add new feature'`)
4. Push to branch (`git push origin feature/improvement`)
5. Open a Pull Request

See [CONTRIBUTING.md](CONTRIBUTING.md) for detailed guidelines.

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

---

## Contact

**Author**: Samuel Wang
**GitHub**: [@samjiawng](https://github.com/samjiawng)  

For questions regarding this research, please open an issue on GitHub.

---

## Acknowledgments

This research was conducted using:
- HuggingFace Transformers library
- PyTorch deep learning framework
- scikit-learn machine learning library

Inspired by research from Anthropic, OpenAI, and the broader interpretability community.
