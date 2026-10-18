"""
Train the spiral network once and save the learned weights to model/.

The API loads these saved weights at startup, so the model is trained
once here and only served (not retrained) in production.

Usage:
    python train.py
    python train.py --epochs 2000 --lr 0.001 --seed 42
"""
import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from Network import (
    Categorical_Loss,
    Container,
    Layer_Creation,
    RELU_Activation,
    SoftMax_Activation,
    generate_spiral_data,
)

MODEL_DIR = Path(__file__).parent / "model"
WEIGHTS_PATH = MODEL_DIR / "spiral_model.npz"
METRICS_PATH = MODEL_DIR / "metrics.json"

LAYER_SIZES = [2, 64, 64, 3]
POINTS_PER_CLASS = 100
CLASSES = 3


def build_network():
    container = Container()
    container.add(Layer_Creation(2, 64))
    container.add(RELU_Activation())
    container.add(Layer_Creation(64, 64))
    container.add(RELU_Activation())
    container.add(Layer_Creation(64, 3))
    container.add(SoftMax_Activation())
    container.add(Categorical_Loss())
    return container


def accuracy(probabilities, one_hot):
    return float(np.mean(np.argmax(probabilities, axis=1) == np.argmax(one_hot, axis=1)))


def main():
    parser = argparse.ArgumentParser(description="Train the spiral network and save its weights.")
    parser.add_argument("--epochs", type=int, default=2000)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    np.random.seed(args.seed)
    X_train, y_train = generate_spiral_data(POINTS_PER_CLASS, CLASSES)
    # A second, independently sampled spiral acts as the held-out test set.
    X_test, y_test = generate_spiral_data(POINTS_PER_CLASS, CLASSES)

    container = build_network()
    for epoch in range(args.epochs):
        loss = container.forward(X_train, y_train)
        container.backward(container.ypreds, y_train, args.lr)
        if epoch % 500 == 0:
            print(f"Epoch {epoch}, Loss: {loss:.4f}")

    train_loss = container.forward(X_train, y_train)
    train_acc = accuracy(container.ypreds, y_train)
    test_loss = container.forward(X_test, y_test)
    test_acc = accuracy(container.ypreds, y_test)
    print(f"Train loss {train_loss:.4f}, accuracy {train_acc:.4f}")
    print(f"Test  loss {test_loss:.4f}, accuracy {test_acc:.4f}")

    layers = [inst for inst in container.instances if isinstance(inst, Layer_Creation)]
    weights = {}
    for i, layer in enumerate(layers):
        weights[f"W{i}"] = layer.weights
        weights[f"b{i}"] = layer.biases

    MODEL_DIR.mkdir(exist_ok=True)
    np.savez(WEIGHTS_PATH, **weights)

    metrics = {
        "architecture": "Input(2) -> Dense(64, ReLU) -> Dense(64, ReLU) -> Dense(3, Softmax)",
        "layer_sizes": LAYER_SIZES,
        "classes": CLASSES,
        "epochs": args.epochs,
        "learning_rate": args.lr,
        "seed": args.seed,
        "train_loss": round(float(train_loss), 4),
        "train_accuracy": round(train_acc, 4),
        "test_loss": round(float(test_loss), 4),
        "test_accuracy": round(test_acc, 4),
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    METRICS_PATH.write_text(json.dumps(metrics, indent=2) + "\n")
    print(f"Saved weights to {WEIGHTS_PATH} and metrics to {METRICS_PATH}")


if __name__ == "__main__":
    main()
