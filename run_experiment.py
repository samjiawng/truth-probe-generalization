import argparse
import os
import subprocess
import sys

def run_cmd(cmd, desc):
    print(f"\n{'='*60}\n{desc}\n{'='*60}")
    result = subprocess.run(cmd, capture_output=False)
    if result.returncode != 0:
        sys.exit(1)

def run_full_experiment(model_name="gpt2", train_domains=None, test_domains=None, num_per_domain=50, output_base="experiments/run1"):
    if train_domains is None:
        train_domains = ["science", "history"]
    if test_domains is None:
        test_domains = ["geography", "math"]
    
    os.makedirs(output_base, exist_ok=True)
    
    data_dir = os.path.join(output_base, "data")
    train_acts = os.path.join(output_base, "activations/train")
    test_acts = os.path.join(output_base, "activations/test")
    probes_dir = os.path.join(output_base, "probes")
    eval_dir = os.path.join(output_base, "evaluation")
    plots_dir = os.path.join(output_base, "plots")
    
    run_cmd(["python3", "src/data/generate_dataset.py", "--output_dir", data_dir, "--num_per_domain", str(num_per_domain), "--domain_split", "--train_domains"] + train_domains + ["--test_domains"] + test_domains, "Generating Dataset")
    
    run_cmd(["python3", "src/models/extract_activations.py", "--model_name", model_name, "--dataset_path", os.path.join(data_dir, "train_domain_split.jsonl"), "--output_dir", train_acts, "--batch_size", "16"], "Extracting Train Activations")
    
    run_cmd(["python3", "src/models/extract_activations.py", "--model_name", model_name, "--dataset_path", os.path.join(data_dir, "test_domain_split.jsonl"), "--output_dir", test_acts, "--batch_size", "16"], "Extracting Test Activations")
    
    run_cmd(["python3", "src/probes/train_probes.py", "--activations_dir", train_acts, "--output_dir", probes_dir], "Training Probes")
    
    for probe_file in os.listdir(probes_dir):
        if probe_file.startswith("probe_layer_"):
            layer = int(probe_file.split("_")[2].split(".")[0])
            run_cmd(["python3", "src/evaluation/evaluate.py", "--probe_path", os.path.join(probes_dir, probe_file), "--activations_dir", test_acts, "--layer", str(layer), "--output_dir", eval_dir], f"Evaluating Layer {layer}")
    
    run_cmd(["python3", "src/evaluation/visualize.py", "--results_dir", eval_dir, "--output_dir", plots_dir], "Creating Visualizations")
    
    print(f"\n{'='*60}\nEXPERIMENT COMPLETE!\n{'='*60}")
    print(f"Results: {output_base}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default="gpt2")
    parser.add_argument("--train_domains", nargs="+", default=["science", "history"])
    parser.add_argument("--test_domains", nargs="+", default=["geography", "math"])
    parser.add_argument("--num_per_domain", type=int, default=50)
    parser.add_argument("--output_base", default="experiments/run1")
    args = parser.parse_args()
    
    run_full_experiment(args.model_name, args.train_domains, args.test_domains, args.num_per_domain, args.output_base)
