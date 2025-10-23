"""Model utilities for activation extraction."""
import torch
from transformers import AutoModel, AutoTokenizer, AutoModelForCausalLM
import numpy as np

class ModelWrapper:
    def __init__(self, model_name, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Loading {model_name} on {self.device}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        causal_models = ["gpt", "opt", "bloom", "llama"]
        use_causal = any(m in model_name.lower() for m in causal_models)
        
        if use_causal:
            self.model = AutoModelForCausalLM.from_pretrained(model_name, output_hidden_states=True).to(self.device)
        else:
            self.model = AutoModel.from_pretrained(model_name, output_hidden_states=True).to(self.device)
        
        self.model.eval()
        self.num_layers = self.model.config.num_hidden_layers
        self.hidden_size = self.model.config.hidden_size
        print(f"Loaded! Layers: {self.num_layers}, Hidden size: {self.hidden_size}")
    
    def get_activations(self, texts, layers=None, pooling="last"):
        if isinstance(texts, str):
            texts = [texts]
        
        inputs = self.tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=512).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            hidden_states = outputs.hidden_states
        
        if layers is None:
            layers = list(range(self.num_layers))
        
        activations = {}
        for layer_idx in layers:
            if layer_idx >= len(hidden_states):
                continue
            layer_acts = hidden_states[layer_idx]
            
            if pooling == "last":
                seq_lengths = inputs["attention_mask"].sum(dim=1) - 1
                batch_indices = torch.arange(layer_acts.size(0))
                pooled = layer_acts[batch_indices, seq_lengths]
            elif pooling == "mean":
                mask = inputs["attention_mask"].unsqueeze(-1)
                pooled = (layer_acts * mask).sum(dim=1) / mask.sum(dim=1)
            else:
                pooled = layer_acts[:, 0, :]
            
            activations[layer_idx] = pooled.cpu().numpy()
        
        return activations
    
    def get_middle_layers(self, n=3):
        start = self.num_layers // 2 - n // 2
        return list(range(start, min(start + n, self.num_layers)))

def load_model(model_name="gpt2", device=None):
    return ModelWrapper(model_name, device)

def batch_process(model, texts, batch_size=32, layers=None, pooling="last"):
    from tqdm import tqdm
    all_acts = {l: [] for l in (layers or model.get_middle_layers(3))}
    
    for i in tqdm(range(0, len(texts), batch_size), desc="Extracting"):
        batch = texts[i:i+batch_size]
        acts = model.get_activations(batch, layers, pooling)
        for l, a in acts.items():
            all_acts[l].append(a)
    
    return {l: np.concatenate(all_acts[l]) for l in all_acts}
