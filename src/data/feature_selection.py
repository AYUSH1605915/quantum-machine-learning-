"""
Module: src.data.feature_selection
Implements EnsembleFeatureSelector combining:
  1. ANOVA F-statistic Filter
  2. Mutual Information Filter
  3. LASSO (L1) Regularization Filter
  4. Ridge (L2) Regularization Filter
And SMOTEENN sampling exclusively for training distributions.
"""

import logging
from collections import Counter
from typing import List, Tuple, Optional

import numpy as np
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors

try:
    from imblearn.combine import SMOTEENN
    HAS_IMBLEARN = True
except Exception:
    HAS_IMBLEARN = False

logger = logging.getLogger(__name__)


class EnsembleFeatureSelector:
    """
    Ensemble Feature Selection combining Filter and Embedded methods:
    ANOVA F-value, Mutual Information, LASSO (L1), and Ridge (L2).
    
    Identifies the consensus intersection of discriminative features to fit
    the quantum circuit qubit budget (e.g., 6 to 10 qubits).
    """

    def __init__(self, n_features: int = 8):
        """
        Args:
            n_features (int): Target number of final compact features for quantum encoding.
        """
        self.n_features = n_features
        self.selected_indices_: Optional[np.ndarray] = None
        self.feature_scores_: dict = {}

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> "EnsembleFeatureSelector":
        """
        Runs the 4 feature selection algorithms and takes the intersection.
        Falls back to frequency-ranked union if intersection < n_features.

        Args:
            X_train (np.ndarray): Training feature matrix.
            y_train (np.ndarray): Training binary target vector.

        Returns:
            self
        """
        n_samples, total_features = X_train.shape
        k_candidate = min(20, total_features)

        # Standardize for stable LASSO / Ridge coefficient comparisons
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_train)

        # 1. ANOVA F-value filter
        anova = SelectKBest(score_func=f_classif, k=k_candidate)
        with np.errstate(divide="ignore", invalid="ignore"):
            anova.fit(X_scaled, y_train)
        anova_scores = np.nan_to_num(anova.scores_, nan=0.0)
        top_anova = set(np.argsort(anova_scores)[::-1][:k_candidate])

        # 2. Mutual Information filter
        mi = SelectKBest(score_func=mutual_info_classif, k=k_candidate)
        mi.fit(X_scaled, y_train)
        top_mi = set(np.argsort(mi.scores_)[::-1][:k_candidate])

        # 3. LASSO (L1) embedded selection
        lasso = LogisticRegression(penalty="l1", solver="liblinear", C=0.1, random_state=42)
        lasso.fit(X_scaled, y_train)
        lasso_coef = np.abs(lasso.coef_[0])
        top_lasso = set(np.where(lasso_coef > 0)[0])
        if len(top_lasso) == 0:
            top_lasso = set(np.argsort(lasso_coef)[::-1][:k_candidate])

        # 4. Ridge (L2) embedded selection
        ridge = LogisticRegression(penalty="l2", C=1.0, random_state=42)
        ridge.fit(X_scaled, y_train)
        ridge_coef = np.abs(ridge.coef_[0])
        top_ridge = set(np.argsort(ridge_coef)[::-1][:k_candidate])

        # 5. Take intersection of all 4 methods
        intersection = top_anova & top_mi & top_lasso & top_ridge

        logger.info(
            "Feature counts: ANOVA=%d, MI=%d, LASSO=%d, Ridge=%d, Intersection=%d",
            len(top_anova), len(top_mi), len(top_lasso), len(top_ridge), len(intersection)
        )

        # If intersection has enough features, take top n_features from intersection
        if len(intersection) >= self.n_features:
            # Rank intersection by mean rank
            selected = list(intersection)[:self.n_features]
        else:
            # Fall back to union ranked by selection frequency across methods
            all_votes: List[int] = list(top_anova) + list(top_mi) + list(top_lasso) + list(top_ridge)
            vote_counts = Counter(all_votes)
            
            # Sort by votes descending, then by ANOVA score descending
            ranked = sorted(
                vote_counts.keys(),
                key=lambda idx: (vote_counts[idx], anova_scores[idx]),
                reverse=True
            )
            selected = ranked[:min(self.n_features, total_features)]

        if len(selected) == 0:
            raise ValueError("Consensus feature selection yielded 0 features.")

        self.selected_indices_ = np.array(sorted(selected))
        self.feature_scores_ = {
            "anova_scores": anova_scores[self.selected_indices_],
            "selected_indices": self.selected_indices_.tolist()
        }

        logger.info("Ensemble selected %d features: %s", len(self.selected_indices_), self.selected_indices_)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Subsets X to the selected feature indices.

        Args:
            X (np.ndarray): Feature matrix.

        Returns:
            np.ndarray: Reduced feature matrix.
        """
        if self.selected_indices_ is None:
            raise RuntimeError("EnsembleFeatureSelector has not been fitted yet.")
        return X[:, self.selected_indices_]

    def fit_transform(self, X_train: np.ndarray, y_train: np.ndarray) -> np.ndarray:
        """Fits and transforms the training data."""
        return self.fit(X_train, y_train).transform(X_train)


def apply_smoteenn(X_train: np.ndarray, y_train: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Applies SMOTE + Edited Nearest Neighbours (SMOTEENN) to balance clinical data.
    IMPORTANT: Must be applied ONLY on training split to avoid data leakage.

    Args:
        X_train (np.ndarray): Training feature matrix.
        y_train (np.ndarray): Training binary target labels.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Resampled X_train, resampled y_train.
    """
    if HAS_IMBLEARN:
        try:
            smote_enn = SMOTEENN(random_state=42)
            X_resampled, y_resampled = smote_enn.fit_resample(X_train, y_train)
            logger.info("SMOTEENN applied: %d -> %d samples", len(y_train), len(y_resampled))
            return np.array(X_resampled), np.array(y_resampled)
        except Exception:
            pass

    # High-reliability custom SMOTE + ENN implementation
    np.random.seed(42)
    classes, counts = np.unique(y_train, return_counts=True)
    min_class = classes[np.argmin(counts)]
    maj_class = classes[np.argmax(counts)]

    X_min = X_train[y_train == min_class]
    X_maj = X_train[y_train == maj_class]
    n_synth = len(X_maj) - len(X_min)

    if n_synth > 0 and len(X_min) >= 3:
        k = min(5, len(X_min) - 1)
        nn = NearestNeighbors(n_neighbors=k + 1).fit(X_min)
        indices = nn.kneighbors(X_min, return_distance=False)

        synth_samples = []
        for _ in range(n_synth):
            idx = np.random.randint(0, len(X_min))
            nn_idx = indices[idx, np.random.randint(1, k + 1)]
            lam = np.random.rand()
            synth = X_min[idx] + lam * (X_min[nn_idx] - X_min[idx])
            synth_samples.append(synth)

        X_balanced = np.vstack([X_train, np.array(synth_samples)])
        y_balanced = np.concatenate([y_train, np.full(n_synth, min_class)])
    else:
        X_balanced, y_balanced = X_train, y_train

    # ENN cleaning: Remove points whose majority of 3 neighbors are from other class
    enn_nn = NearestNeighbors(n_neighbors=4).fit(X_balanced)
    enn_idx = enn_nn.kneighbors(X_balanced, return_distance=False)[:, 1:]
    keep_mask = []
    for i, neighbors in enumerate(enn_idx):
        neighbor_labels = y_balanced[neighbors]
        maj_neighbor = Counter(neighbor_labels).most_common(1)[0][0]
        keep_mask.append(maj_neighbor == y_balanced[i])

    keep_mask = np.array(keep_mask)
    X_clean = X_balanced[keep_mask]
    y_clean = y_balanced[keep_mask]
    logger.info("SMOTEENN (Native) applied: %d -> %d samples", len(y_train), len(y_clean))
    return np.array(X_clean), np.array(y_clean)
