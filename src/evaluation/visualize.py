"""Visualization utilities."""
import json
import os
import matplotlib.pyplot as plt

def plot_layer_performance(results_file, output_dir):
    with open(results_file) as f:
        results = json.load(f)
    
    layers = sorted([int(k) for k in results.keys()])
    train_accs = [results[str(l)]["train"]["accuracy"] for l in layers]
    test_accs = [results[str(l)]["test"]["accuracy"] for l in layers]
    
    plt.figure(figsize=(10, 6))
    plt.plot(layers, train_accs, marker='o', label='Train', linewidth=2)
    plt.plot(layers, test_accs, marker='s', label='Test', linewidth=2)
    plt.xlabel('Layer')
    plt.ylabel('Accuracy')
    plt.title('Probe Performance Across Layers')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'layer_performance.png'), dpi=300)
    print(f"Saved to {output_dir}/layer_performance.png")
    plt.close()

def create_all_plots(results_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    summary = os.path.join(results_dir, "results_summary.json")
    if os.path.exists(summary):
        plot_layer_performance(summary, output_dir)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()
    create_all_plots(args.results_dir, args.output_dir)
