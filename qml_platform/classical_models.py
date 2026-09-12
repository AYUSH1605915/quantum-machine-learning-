import time
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

class ClassicalBaselines:
    """
    Classical Machine Learning baseline models for benchmark comparison.
    Includes Logistic Regression, Support Vector Machine (RBF), Random Forest, and MLP.
    """
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.models = {
            "Logistic Regression": LogisticRegression(max_iter=1000, random_state=self.random_state),
            "Classical SVM (RBF)": SVC(kernel="rbf", probability=True, random_state=self.random_state),
            "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=5, random_state=self.random_state),
            "Multilayer Perceptron (MLP)": MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500, random_state=self.random_state)
        }
        self.training_times = {}

    def fit_all(self, X_train, y_train):
        """Train all classical models on classical preprocessed data."""
        for name, model in self.models.items():
            start_t = time.time()
            model.fit(X_train, y_train)
            self.training_times[name] = time.time() - start_t
        return self

    def get_model(self, name):
        return self.models.get(name)

    def get_all_models(self):
        return self.models
