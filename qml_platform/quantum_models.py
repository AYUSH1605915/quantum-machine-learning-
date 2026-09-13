import time
import numpy as np
import pennylane as qml
from pennylane import numpy as pnp
from sklearn.svm import SVC

class VariationalQuantumClassifier:
    """
    Variational Quantum Classifier (VQC) with Parameterized Quantum Circuit (PQC).
    Employs Angle Feature Encoding, Strongly Entangling Layers, and Hybrid Classical Optimization.
    """
    def __init__(self, n_qubits=4, n_layers=2, learning_rate=0.06, epochs=30, batch_size=16, random_state=42):
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.random_state = random_state

        self.dev = qml.device("default.qubit", wires=self.n_qubits)
        self.weights = None
        self.bias = 0.0
        self.training_history = []
        self.training_time = 0.0

        # Build QNode
        self._build_qnode()

    def _build_qnode(self):
        @qml.qnode(self.dev, interface="autograd")
        def circuit(weights, x):
            # 1. Quantum State Preparation (Angle Embedding)
            for i in range(self.n_qubits):
                qml.Hadamard(wires=i)
                qml.RY(x[i], wires=i)

            # 2. Parameterized Quantum Circuit Layers (Rotations + Entanglement)
            for l in range(self.n_layers):
                for i in range(self.n_qubits):
                    qml.RY(weights[l, i, 0], wires=i)
                    qml.RZ(weights[l, i, 1], wires=i)

                # Entangling CNOT ring topology
                for i in range(self.n_qubits):
                    qml.CNOT(wires=[i, (i + 1) % self.n_qubits])

            # 3. Measurement: Pauli-Z expectation value on wire 0
            return qml.expval(qml.PauliZ(0))

        self.qnode = circuit

    def _cost(self, weights, bias, X_batch=None, y_batch=None):
        loss = 0.0
        for x, y in zip(X_batch, y_batch):
            target = 1.0 if y == 1 else -1.0
            expval = self.qnode(weights, x)
            pred = expval + bias
            loss = loss + (pred - target) ** 2
        return loss / len(X_batch)

    def fit(self, X, y):
        """Train the VQC model using PennyLane Adam Optimizer."""
        start_time = time.time()
        np.random.seed(self.random_state)

        # Initialize trainable variational parameters (weights and bias)
        weights = pnp.array(0.05 * np.random.randn(self.n_layers, self.n_qubits, 2), requires_grad=True)
        bias = pnp.array(0.0, requires_grad=True)

        opt = qml.AdamOptimizer(stepsize=self.learning_rate)
        n_samples = len(X)
        self.training_history = []

        for epoch in range(self.epochs):
            indices = np.random.permutation(n_samples)
            X_shuffled = X[indices]
            y_shuffled = y[indices]

            epoch_loss = 0.0
            n_batches = 0

            for i in range(0, n_samples, self.batch_size):
                X_batch = X_shuffled[i:i + self.batch_size]
                y_batch = y_shuffled[i:i + self.batch_size]
                (weights, bias), batch_loss = opt.step_and_cost(
                    self._cost, weights, bias, X_batch=X_batch, y_batch=y_batch
                )
                epoch_loss += float(batch_loss)
                n_batches += 1

            avg_loss = epoch_loss / max(n_batches, 1)
            self.training_history.append({
                "epoch": epoch + 1,
                "loss": float(avg_loss)
            })

        self.weights = weights
        self.bias = float(bias)
        self.training_time = time.time() - start_time
        return self

    def predict_proba(self, X):
        """Predict disease probability for samples."""
        probs = []
        for x in X:
            expval = float(self.qnode(self.weights, x))
            pred = expval + self.bias
            prob_1 = float(1.0 / (1.0 + np.exp(-pred * 2.5)))
            prob_0 = 1.0 - prob_1
            probs.append([prob_0, prob_1])
        return np.array(probs)

    def predict(self, X, threshold=0.5):
        """Predict binary disease label (0 or 1)."""
        proba = self.predict_proba(X)
        return (proba[:, 1] >= threshold).astype(int)

    def get_circuit_ascii(self, sample_x=None):
        """Draws ASCII quantum circuit diagram."""
        if sample_x is None:
            sample_x = np.zeros(self.n_qubits)
        if self.weights is None:
            sample_weights = np.zeros((self.n_layers, self.n_qubits, 2))
        else:
            sample_weights = self.weights

        try:
            drawer = qml.draw(self.qnode, expansion_strategy="device")
            return drawer(sample_weights, sample_x)
        except Exception:
            return f"PennyLane VQC Circuit with {self.n_qubits} qubits, {self.n_layers} layers."


class QuantumSVM:
    """
    Quantum Support Vector Machine (QSVM) utilizing Quantum Kernel Estimation.
    Computes fidelity matrix in quantum Hilbert space and applies dual SVM optimization.
    """
    def __init__(self, n_qubits=4, C=1.0, random_state=42):
        self.n_qubits = n_qubits
        self.C = C
        self.random_state = random_state
        self.dev = qml.device("default.qubit", wires=self.n_qubits)
        self.svm = SVC(kernel="precomputed", C=self.C, random_state=self.random_state)
        self.X_train = None
        self.training_time = 0.0

        self._build_kernel_qnode()

    def _build_kernel_qnode(self):
        @qml.qnode(self.dev)
        def kernel_circuit(x1, x2):
            # Encode x1: Hadamard + RY + Entanglement
            for i in range(self.n_qubits):
                qml.Hadamard(wires=i)
                qml.RY(x1[i], wires=i)
            for i in range(self.n_qubits - 1):
                qml.CZ(wires=[i, i + 1])

            # Invert encoding for x2 (Adjoint / Inverse circuit)
            for i in reversed(range(self.n_qubits - 1)):
                qml.CZ(wires=[i, i + 1])
            for i in reversed(range(self.n_qubits)):
                qml.RY(-x2[i], wires=i)
                qml.Hadamard(wires=i)

            # Measurement of projection onto |0...0> ground state
            return qml.probs(wires=range(self.n_qubits))

        self.kernel_qnode = kernel_circuit

    def compute_kernel_matrix(self, X1, X2):
        """
        Computes the Quantum Kernel Gram Matrix between dataset X1 and X2.
        Leverages vectorized state-overlap fidelity: |<psi(x1)|psi(x2)>|^2 = prod(cos^2((x1_i - x2_i)/2)).
        Identical to PennyLane kernel circuit to machine precision.
        """
        X1 = np.asarray(X1)
        X2 = np.asarray(X2)
        diff = (X1[:, np.newaxis, :] - X2[np.newaxis, :, :]) / 2.0
        return np.prod(np.cos(diff) ** 2, axis=-1)

    def fit(self, X, y):
        """Trains the QSVM by computing the quantum training kernel matrix."""
        start_time = time.time()
        self.X_train = X
        K_train = self.compute_kernel_matrix(X, X)
        self.svm.fit(K_train, y)
        self.training_time = time.time() - start_time
        return self

    def predict(self, X):
        """Predict binary disease label for new test samples."""
        K_test = self.compute_kernel_matrix(X, self.X_train)
        return self.svm.predict(K_test)

    def predict_proba(self, X):
        """Predict probability scores for new test samples."""
        K_test = self.compute_kernel_matrix(X, self.X_train)
        dec = self.svm.decision_function(K_test)
        prob_1 = 1.0 / (1.0 + np.exp(-dec))
        prob_0 = 1.0 - prob_1
        return np.vstack([prob_0, prob_1]).T
