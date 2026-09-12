"""
Module: src.quantum.feature_maps
Implements 2nd-order Pauli ZZ-FeatureMap quantum encoding for biomedical vectors.
Scales features to [0, 2π] and builds entangling circuits for PennyLane and Qiskit.
"""

from typing import Tuple, Optional
import numpy as np
import pennylane as qml
from sklearn.preprocessing import MinMaxScaler


def normalize_for_encoding(X: np.ndarray, scaler: Optional[MinMaxScaler] = None) -> Tuple[np.ndarray, MinMaxScaler]:
    """
    Min-Max normalizes each feature into the quantum angle domain [0, 2π].
    Formula: x_norm = (x - min(x)) / (max(x) - min(x)) * 2 * np.pi

    Args:
        X (np.ndarray): Feature matrix (samples x features).
        scaler (Optional[MinMaxScaler]): Fitted scaler from training data if transforming test/val.

    Returns:
        Tuple[np.ndarray, MinMaxScaler]: Normalized feature matrix and fitted scaler.
    """
    if scaler is None:
        scaler = MinMaxScaler(feature_range=(0.0, 2.0 * np.pi))
        X_norm = scaler.fit_transform(X)
    else:
        X_norm = scaler.transform(X)

    # Ensure numerical bounds in [0, 2π]
    X_norm = np.clip(X_norm, 0.0, 2.0 * np.pi)
    return X_norm, scaler


class ZZFeatureMapEncoder:
    """
    2nd-order Pauli ZZ-FeatureMap Quantum Encoder.
    Applies Hadamard superposition, RZ rotations, and pairwise ZZ entangling phase gates:
      φ_{j,k}(x) = 2 * (π - x_j) * (π - x_k)
    """

    def __init__(self, n_qubits: int = 8, reps: int = 2, backend: str = "pennylane"):
        """
        Args:
            n_qubits (int): Number of qubits matching selected feature dimensions.
            reps (int): Circuit repetitions / depth.
            backend (str): 'pennylane' or 'qiskit'.
        """
        self.n_qubits = n_qubits
        self.reps = reps
        self.backend = backend.lower()

        if self.backend == "pennylane":
            self.dev = qml.device("default.qubit", wires=self.n_qubits)
            self._build_pennylane_circuit()
        else:
            self._build_qiskit_circuit()

    def _build_pennylane_circuit(self):
        @qml.qnode(self.dev)
        def circuit(x):
            for _ in range(self.reps):
                # Hadamard superposition + single-qubit rotations
                for i in range(self.n_qubits):
                    qml.Hadamard(wires=i)
                    qml.RZ(2.0 * x[i], wires=i)

                # Pairwise 2nd-order ZZ-entanglement
                # Linear nearest-neighbor or all-to-all pairs
                for i in range(self.n_qubits - 1):
                    j = i + 1
                    phi_ij = 2.0 * (np.pi - x[i]) * (np.pi - x[j])
                    qml.CNOT(wires=[i, j])
                    qml.RZ(phi_ij, wires=j)
                    qml.CNOT(wires=[i, j])

            return qml.state()

        self.qnode = circuit

    def _build_qiskit_circuit(self):
        try:
            from qiskit.circuit.library import ZZFeatureMap
            self.feature_map = ZZFeatureMap(feature_dimension=self.n_qubits, reps=self.reps, entanglement="linear")
        except Exception:
            self.feature_map = None

    def encode(self, x: np.ndarray) -> np.ndarray:
        """Encodes single sample x into quantum state vector."""
        if self.backend == "pennylane":
            return self.qnode(x)
        else:
            raise NotImplementedError("Qiskit state evaluation requires simulator backend.")

    def draw(self, sample_x: Optional[np.ndarray] = None) -> str:
        """Returns ASCII circuit diagram."""
        if sample_x is None:
            sample_x = np.full(self.n_qubits, np.pi)

        if self.backend == "pennylane":
            try:
                drawer = qml.draw(self.qnode, expansion_strategy="device")
                return drawer(sample_x)
            except Exception:
                return f"PennyLane ZZFeatureMap: {self.n_qubits} qubits, {self.reps} repetitions."
        elif self.feature_map is not None:
            return str(self.feature_map.draw())
        return "ZZFeatureMap"


if __name__ == "__main__":
    # Unit test with 8 random samples, 8 features
    np.random.seed(42)
    X_test = np.random.randn(8, 8)
    X_norm, scaler = normalize_for_encoding(X_test)
    assert X_norm.shape == (8, 8)
    assert np.all(X_norm >= 0.0) and np.all(X_norm <= 2.0 * np.pi + 1e-5)

    encoder = ZZFeatureMapEncoder(n_qubits=8, reps=2, backend="pennylane")
    state = encoder.encode(X_norm[0])
    assert len(state) == 2 ** 8  # 256 dimensional Hilbert space
    print("Feature Map Unit Test Passed! State vector norm:", np.linalg.norm(state))
    print(encoder.draw(X_norm[0]))
