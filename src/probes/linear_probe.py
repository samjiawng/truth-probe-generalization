import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

class SklearnProbe:
    def __init__(self, input_dim, C=1.0, max_iter=1000, random_state=42):
        self.model = LogisticRegression(C=C, max_iter=max_iter, random_state=random_state)
    
    def fit(self, X, y):
        self.model.fit(X, y)
        return self
    
    def predict(self, X):
        return self.model.predict(X)
    
    def predict_proba(self, X):
        return self.model.predict_proba(X)
    
    def score(self, X, y):
        return self.model.score(X, y)
    
    def evaluate(self, X, y):
        y_pred = self.predict(X)
        y_proba = self.predict_proba(X)[:, 1]
        acc = accuracy_score(y, y_pred)
        p, r, f1, _ = precision_recall_fscore_support(y, y_pred, average='binary')
        return {"accuracy": acc, "precision": p, "recall": r, "f1": f1, "predictions": y_pred, "probabilities": y_proba}

def create_probe(input_dim, probe_type="sklearn", **kwargs):
    if probe_type == "sklearn":
        return SklearnProbe(input_dim, **kwargs)
    raise ValueError(f"Unknown probe type: {probe_type}")
