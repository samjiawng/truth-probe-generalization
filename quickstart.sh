#!/bin/bash
echo "=================================="
echo "Truth Probe Generalization"
echo "Quick Start"
echo "=================================="

pip install -r requirements.txt --quiet

python run_experiment.py \
    --model_name "gpt2" \
    --train_domains science history \
    --test_domains geography math \
    --num_per_domain 20 \
    --output_base "experiments/quickstart"
