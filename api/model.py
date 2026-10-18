"""Loads the trained weights into the from-scratch network and runs inference."""
import json
from pathlib import Path

import numpy as np

from Network import Container, Layer_Creation, RELU_Activation, SoftMax_Activation

MODEL_DIR = Path(__file__).resolve().parent.parent / "model"
WEIGHTS_PATH = MODEL_DIR / "spiral_model.npz"
METRICS_PATH = MODEL_DIR / "metrics.json"


class SpiralModel:
    def __init__(self, weights_path=WEIGHTS_PATH, metrics_path=METRICS_PATH):
        params = np.load(weights_path)
        layer_count = len([key for key in params.files if key.startswith("W")])

        # Same layers as training, minus the loss: inference only needs the forward pass.
        self.container = Container()
        for i in range(layer_count):
            weights = params[f"W{i}"]
            layer = Layer_Creation(*weights.shape)
            layer.weights = weights
            layer.biases = params[f"b{i}"]
            self.container.add(layer)
            self.container.add(RELU_Activation() if i < layer_count - 1 else SoftMax_Activation())

        self.metrics = json.loads(Path(metrics_path).read_text())
        self.num_classes = int(params[f"W{layer_count - 1}"].shape[1])

    def predict_proba(self, points):
        out = np.asarray(points, dtype=float)
        for instance in self.container.instances:
            out = instance.forward_pass(out)
        return out
