"""Evaluate trained probes."""
import argparse
import json
import os
import pickle
import numpy as np
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

def load_probe(path):
    with open(path, 'rb') as f:
        return pickle.load(f)

def load_activations(activations_dir, layer):
    layer_dir = os.path.join(activations_dir, f"layer_{layer}")
    acts = np.load(os.path.join(layer_dir, "activations.npy"))
    labels = np.load(os.path.join(layer_dir, "labels.npy"))
    domains = np.load(os.path.join(layer_dir, "domains.npy"), allow_pickle=True)
    return acts, labels, domains

def evaluate_by_domain(probe, acts, labels, domains):
    results = {}
    for domain in np.unique(domains):
        mask = domains == domain
        res = probe.evaluate(acts[mask], labels[mask])
        results[domain] = {k: float(v) for k, v in res.items() if k not in ['predictions', 'probabilities']}
        results[domain]['num_examples'] = int(mask.sum())
    return results

def evaluate_comprehensive(probe_path, activations_dir, layer, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    
    probe = load_probe(probe_path)
    acts, labels, domains = load_activations(activations_dir, layer)
    
    overall = probe.evaluate(acts, labels)
    domain_results = evaluate_by_domain(probe, acts, labels, domains)
    
    print(f"\nLayer {layer} Results:")
    print(f"Overall Accuracy: {overall['accuracy']:.4f}")
    for domain, metrics in domain_results.items():
        print(f"  {domain}: {metrics['accuracy']:.4f}")
    
    results = {
        "layer": layer,
        "overall": {k: float(v) for k, v in overall.items() if k not in ['predictions', 'probabilities']},
        "per_domain": domain_results
    }
    
    with open(os.path.join(output_dir, f"evaluation_layer_{layer}.json"), 'w') as f:
        json.dump(results, f, indent=2)
    
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--probe_path", required=True)
    parser.add_argument("--activations_dir", required=True)
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()
    
    evaluate_comprehensive(args.probe_path, args.activations_dir, args.layer, args.output_dir)
