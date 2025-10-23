"""Train probes on activations."""
import argparse
import json
import os
import pickle
import numpy as np
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))
from src.probes.linear_probe import create_probe

def load_activations(activations_dir, layer):
    layer_dir = os.path.join(activations_dir, f"layer_{layer}")
    acts = np.load(os.path.join(layer_dir, "activations.npy"))
    labels = np.load(os.path.join(layer_dir, "labels.npy"))
    domains = np.load(os.path.join(layer_dir, "domains.npy"), allow_pickle=True)
    return acts, labels, domains

def filter_by_domains(acts, labels, domains, domain_list):
    if domain_list is None:
        return acts, labels, domains
    mask = np.isin(domains, domain_list)
    return acts[mask], labels[mask], domains[mask]

def train_and_evaluate_all_layers(activations_dir, output_dir, train_domains=None, test_domains=None, probe_type="sklearn", **kwargs):
    os.makedirs(output_dir, exist_ok=True)
    
    layer_dirs = [d for d in os.listdir(activations_dir) if d.startswith("layer_")]
    layers = sorted([int(d.split("_")[1]) for d in layer_dirs])
    
    print(f"Training on {len(layers)} layers: {layers}")
    print(f"Train domains: {train_domains or 'all'}")
    print(f"Test domains: {test_domains or 'all'}")
    
    results = {}
    for layer in layers:
        print(f"\n{'='*60}\nLAYER {layer}\n{'='*60}")
        
        acts, labels, domains = load_activations(activations_dir, layer)
        train_acts, train_labels, _ = filter_by_domains(acts, labels, domains, train_domains)
        test_acts, test_labels, _ = filter_by_domains(acts, labels, domains, test_domains)
        
        print(f"Training on {len(train_acts)} examples")
        probe = create_probe(train_acts.shape[1], probe_type, **kwargs)
        probe.fit(train_acts, train_labels)
        
        train_res = probe.evaluate(train_acts, train_labels)
        test_res = probe.evaluate(test_acts, test_labels)
        
        print(f"Train Acc: {train_res['accuracy']:.4f}, Test Acc: {test_res['accuracy']:.4f}")
        
        with open(os.path.join(output_dir, f"probe_layer_{layer}.pkl"), 'wb') as f:
            pickle.dump(probe, f)
        
        results[layer] = {
            "train": {k: float(v) for k, v in train_res.items() if k not in ['predictions', 'probabilities']},
            "test": {k: float(v) for k, v in test_res.items() if k not in ['predictions', 'probabilities']}
        }
    
    with open(os.path.join(output_dir, "results_summary.json"), 'w') as f:
        json.dump(results, f, indent=2)
    
    best_layer = max(results.keys(), key=lambda l: results[l]["test"]["accuracy"])
    print(f"\nBest layer: {best_layer} (Acc: {results[best_layer]['test']['accuracy']:.4f})")
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--activations_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--train_domains", nargs="+", default=None)
    parser.add_argument("--test_domains", nargs="+", default=None)
    parser.add_argument("--probe_type", default="sklearn")
    parser.add_argument("--C", type=float, default=1.0)
    args = parser.parse_args()
    
    train_and_evaluate_all_layers(args.activations_dir, args.output_dir, args.train_domains, args.test_domains, args.probe_type, C=args.C)
