# Truth Probe Generalization

Evaluating how linear truth probes trained on language model activations generalize across different domains.

## Features

- 🔍 Dataset generation across 6+ domains
- 🧠 Activation extraction from any HuggingFace model
- 📊 Linear probe training with evaluation
- 🎯 Out-of-distribution generalization testing
- 📈 Visualization tools

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run quick experiment
bash quickstart.sh
```

## Usage

```bash
# Generate dataset
python src/data/generate_dataset.py --output_dir data/statements

# Extract activations
python src/models/extract_activations.py \
    --model_name gpt2 \
    --dataset_path data/statements/train.jsonl \
    --output_dir data/activations

# Train probes
python src/probes/train_probes.py \
    --activations_dir data/activations \
    --output_dir results/probes

# Full pipeline
python run_experiment.py --model_name gpt2
```

## Project Structure

```
truth-probe-generalization/
├── src/
│   ├── data/          # Dataset generation
│   ├── models/        # Activation extraction
│   ├── probes/        # Probe training
│   └── evaluation/    # Evaluation & visualization
├── quickstart.sh      # Quick start script
└── run_experiment.py  # Full pipeline
```

## Target

**89% accuracy** on out-of-distribution prompts across 6+ domains

## License

MIT License
