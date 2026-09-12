"""
Module: src.quantum.vqc
Implements Variational Quantum Classifier (VQC) with:
  - Angle Feature Encoding
  - Rotational Unitary Gates (Rx, Ry, Rz)
  - Circular CNOT Entangling Rings
  - Noise-robust SPSA & Adam Hybrid Optimizers
  - Pauli-Z Ground Expectation Value & Sigmoid Probability Mapping
"""

import time
import logging
from pathlib import Path
from typing import Optional, List, Dict

import numpy as np
import pennylane as qml
from pennylane import numpy as pnp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .feature_maps import normalize_for_encoding

logger = logging.getLogger(__name__)


class VQC:
    """
    Variational Quantum Classifier for MASLD Early Detection.
    Executes parameterized quantum circuits on near-term simulators.
    """

    def __init__(
        self,
        n_qubits: int = 8,
        n_layers: int = 3,
        lr: float = 0.05,
        max_iter: int = 50,
        batch_size: int = 16,
        optimizer: str = "spsa",
        random_state: int = 42
    ):
        """
        Args:
            n_qubits (int): Number of quantum wires (features).
            n_layers (int): Depth of variational ansatz layers.
            lr (float): Learning rate / step size.
            max_iter (int): Maximum optimization iterations / epochs.
            batch_size (int): Mini-batch size.
            optimizer (str): 'spsa' or 'adam'.
            random_state (int): Reproducibility seed.
        """
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.lr = lr
        self.max_iter = max_iter
        self.batch_size = batch_size
        self.optimizer = optimizer.lower()
        self.random_state = random_state

        self.dev = qml.device("default.qubit", wires=self.n_qubits)
        self.weights_: Optional[np.ndarray] = None
        self.bias_: float = 0.0
        self.loss_history_: List[float] = []
        self.scaler = None
        self.training_time: float = 0.0

        self._build_circuit()

    def _build_circuit(self):
        """Builds PennyLane parameterized ansatz."""
        @qml.qnode(self.dev, interface="autograd")
        def circuit(weights, x):
            # 1. State Encoding
            qml.AngleEmbedding(x, wires=range(self.n_qubits), rotation="Y")

            # 2. Variational Layers
            for l in range(self.n_layers):
                # 3-parameter general rotation on each qubit
                for i in range(self.n_qubits):
                    qml.Rot(weights[l, i, 0], weights[l, i, 1], weights[l, i, 2], wires=i)

                # Circular CNOT Entanglement
                for i in range(self.n_qubits - 1):
                    qml.CNOT(wires=[i, i + 1])
                qml.CNOT(wires=[self.n_qubits - 1, 0])

            # 3. Readout: Pauli-Z expectation value on wire 0
            return qml.expval(qml.PauliZ(0))

        self.qnode = circuit

    def _cost(self, weights, bias, X_batch, y_batch):
        """Mean Squared Error with target labels in {-1, +1}."""
        loss = 0.0
        for x, y in zip(X_batch, y_batch):
            expval = self.qnode(weights, x)
            pred = expval + bias
            loss = loss + (pred - y) ** 2
        return loss / len(X_batch)

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> "VQC":
        """
        Trains the VQC ansatz using SPSA or Adam optimization.
        Target labels mapped: 0 -> -1.0, 1 -> +1.0.
        """
        start_t = time.time()
        np.random.seed(self.random_state)

        # Scale features to [0, 2π]
        X_norm, self.scaler = normalize_for_encoding(X_train)
        # Convert {0, 1} to {-1, +1}
        y_encoded = np.where(y_train == 1, 1.0, -1.0)

        # Initialize weights uniformly in [-π, π]
        weights = pnp.array(
            np.random.uniform(-np.pi, np.pi, (self.n_layers, self.n_qubits, 3)),
            requires_grad=True
        )
        bias = pnp.array(0.0, requires_grad=True)

        n_samples = len(X_train)
        self.loss_history_ = []
        logger.info("Starting VQC optimization (%s) on %d qubits, %d layers...",
                    self.optimizer.upper(), self.n_qubits, self.n_layers)

        if self.optimizer == "spsa":
            # Simultaneous Perturbation Stochastic Approximation (SPSA)
            a = self.lr
            c = 0.1
            alpha = 0.602
            gamma = 0.101

            for it in range(self.max_iter):
                a_k = a / ((it + 1 + 10) ** alpha)
                c_k = c / ((it + 1) ** gamma)

                # Random batch
                batch_idx = np.random.choice(n_samples, size=min(self.batch_size, n_samples), replace=False)
                X_b, y_b = X_norm[batch_idx], y_encoded[batch_idx]

                # Bernoulli random perturbation vector delta in {-1, +1}
                delta_w = np.random.choice([-1.0, 1.0], size=weights.shape)
                delta_b = np.random.choice([-1.0, 1.0])

                w_plus = weights + c_k * delta_w
                b_plus = bias + c_k * delta_b
                loss_plus = float(self._cost(w_plus, b_plus, X_b, y_b))

                w_minus = weights - c_k * delta_w
                b_minus = bias - c_k * delta_b
                loss_minus = float(self._cost(w_minus, b_minus, X_b, y_b))

                grad_w = (loss_plus - loss_minus) / (2.0 * c_k * delta_w)
                grad_b = (loss_plus - loss_minus) / (2.0 * c_k * delta_b)

                weights = weights - a_k * grad_w
                bias = bias - a_k * grad_b

                current_loss = 0.5 * (loss_plus + loss_minus)
                self.loss_history_.append(float(current_loss))

                if (it + 1) % 10 == 0 or it == 0:
                    logger.info("Iteration %3d/%d | SPSA Loss: %.4f", it + 1, self.max_iter, current_loss)

        else:
            # Adam optimizer
            opt = qml.AdamOptimizer(stepsize=self.lr)
            for it in range(self.max_iter):
                indices = np.random.permutation(n_samples)
                X_shuffled = X_norm[indices]
                y_shuffled = y_encoded[indices]
                epoch_loss = 0.0
                n_b = 0

                for i in range(0, n_samples, self.batch_size):
                    X_b = X_shuffled[i:i + self.batch_size]
                    y_b = y_shuffled[i:i + self.batch_size]
                    (weights, bias), b_loss = opt.step_and_cost(
                        self._cost, weights, bias, X_batch=X_b, y_batch=y_b
                    )
                    epoch_loss += float(b_loss)
                    n_b += 1

                avg_loss = epoch_loss / max(n_b, 1)
                self.loss_history_.append(float(avg_loss))
                if (it + 1) % 10 == 0 or it == 0:
                    logger.info("Iteration %3d/%d | Adam Loss: %.4f", it + 1, self.max_iter, avg_loss)

        self.weights_ = np.array(weights)
        self.bias_ = float(bias)
        self.training_time = time.time() - start_t
        logger.info("VQC training complete in %.2f seconds.", self.training_time)
        return self

    def predict_raw(self, X: np.ndarray) -> np.ndarray:
        """Returns raw expectation values <Z_0> + bias."""
        X_norm, _ = normalize_for_encoding(X, scaler=self.scaler)
        raw_vals = []
        for x in X_norm:
            expval = float(self.qnode(self.weights_, x))
            raw_vals.append(expval + self.bias_)
        return np.array(raw_vals)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predicts class probabilities using sigmoid on expectation value."""
        raw_vals = self.predict_raw(X)
        prob_1 = 1.0 / (1.0 + np.exp(-raw_vals * 2.0))
        prob_1 = np.clip(prob_1, 1e-6, 1.0 - 1e-6)
        prob_0 = 1.0 - prob_1
        return np.vstack([prob_0, prob_1]).T

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """Predicts binary labels: 1 = MASLD, 0 = Healthy."""
        proba = self.predict_proba(X)
        return (proba[:, 1] >= threshold).astype(int)

    def plot_loss(self, save_path: str = "reports/vqc_loss_curve.png"):
        """Plots and saves the optimization loss curve."""
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.figure(figsize=(7, 4))
        plt.plot(range(1, len(self.loss_history_) + 1), self.loss_history_, marker="o", color="#00f2fe", linewidth=2)
        plt.title(f"VQC Optimization Loss Curve ({self.optimizer.upper()})", fontweight="bold")
        plt.xlabel("Iteration / Epoch", fontweight="bold")
        plt.ylabel("MSE Cost Function", fontweight="bold")
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        plt.close()
        logger.info("Saved VQC loss plot to %s", save_path)
