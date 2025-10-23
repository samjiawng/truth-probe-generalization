"""Extract activations from models."""
import argparse
import json
import os
import numpy as np
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))
from src.models.model_utils import load_model, batch_process

def load_dataset(path):
    data = []
    with open(path, 'r') as f:
        for line in f:
            data.append(json.loads(line))
    return data

def extract_and_save(model_name, dataset_path, output_dir, layers=None, pooling="last", batch_size=32, max_examples=None):
    os.makedirs(output_dir, exist_ok=True)
    
    print("Loading model...")
    model = load_model(model_name)
    
    print(f"Loading dataset: {dataset_path}")
    data = load_dataset(dataset_path)
    if max_examples:
        data = data[:max_examples]
    
    texts = [item["statement"] for item in data]
    labels = [item["label"] for item in data]
    domains = [item["domain"] for item in data]
    
    if layers is None:
        layers = model.get_middle_layers(3)
    print(f"Extracting from layers: {layers}")
    
    activations = batch_process(model, texts, batch_size, layers, pooling)
    
    for layer_idx, acts in activations.items():
        layer_dir = os.path.join(output_dir, f"layer_{layer_idx}")
        os.makedirs(layer_dir, exist_ok=True)
        
        np.save(os.path.join(layer_dir, "activations.npy"), acts)
        np.save(os.path.join(layer_dir, "labels.npy"), np.array(labels))
        np.save(os.path.join(layer_dir, "domains.npy"), np.array(domains))
        
        with open(os.path.join(layer_dir, "metadata.json"), 'w') as f:
            json.dump({"model": model_name, "layer": layer_idx, "shape": list(acts.shape)}, f)
        
        print(f"Layer {layer_idx}: {acts.shape} -> {layer_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default="gpt2")
    parser.add_argument("--dataset_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--layers", type=int, nargs="+", default=None)
    parser.add_argument("--pooling", default="last")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_examples", type=int, default=None)
    args = parser.parse_args()
    
    extract_and_save(args.model_name, args.dataset_path, args.output_dir, args.layers, args.pooling, args.batch_size, args.max_examples)
