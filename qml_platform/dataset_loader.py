import os
import re
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedGroupKFold, train_test_split


class BiomedicalDatasetLoader:
    """Parkinson's voice-feature ingestion and preprocessing for classical + QML models."""

    TARGET_CANDIDATES = [
        "status", "Status", "Diagnosis", "diagnosis", "target", "Target",
        "label", "Label", "outcome", "Outcome"
    ]
    ID_CANDIDATES = ["name", "subject", "subject#", "subject_id", "patient_id", "id", "ID"]

    FEATURE_DISPLAY = {
        "MDVP:Fo(Hz)": "Mean fundamental frequency",
        "MDVP:Fhi(Hz)": "Maximum fundamental frequency",
        "MDVP:Flo(Hz)": "Minimum fundamental frequency",
        "MDVP:Jitter(%)": "Jitter (%)",
        "MDVP:Jitter(Abs)": "Jitter (absolute)",
        "MDVP:RAP": "Relative average perturbation",
        "MDVP:PPQ": "Pitch perturbation quotient",
        "Jitter:DDP": "Jitter DDP",
        "MDVP:Shimmer": "Shimmer",
        "MDVP:Shimmer(dB)": "Shimmer (dB)",
        "Shimmer:APQ3": "Shimmer APQ3",
        "Shimmer:APQ5": "Shimmer APQ5",
        "MDVP:APQ": "Amplitude perturbation quotient",
        "Shimmer:DDA": "Shimmer DDA",
        "NHR": "Noise-to-harmonics ratio",
        "HNR": "Harmonics-to-noise ratio",
        "RPDE": "Recurrence period density entropy",
        "DFA": "Detrended fluctuation analysis",
        "spread1": "Nonlinear spread 1",
        "spread2": "Nonlinear spread 2",
        "D2": "Correlation dimension",
        "PPE": "Pitch period entropy",
    }

    def __init__(self, filepath="data/parkinsons.data", n_qubits=4,
                 reduction_method="select_k_best", test_size=0.25, random_state=42):
        self.filepath = filepath
        self.n_qubits = n_qubits
        self.reduction_method = reduction_method
        self.test_size = test_size
        self.random_state = random_state

        self.df_raw = None
        self.df_features_numeric = None
        self.feature_names = []
        self.target_name = "status"
        self.id_column = None
        self.groups = None
        self.split_strategy = "row-stratified"
        self.train_indices = None
        self.test_indices = None

        self.selected_feature_indices = []
        self.selected_feature_names = []

        self.imputer = SimpleImputer(strategy="median")
        self.scaler_classical = StandardScaler()
        self.scaler_quantum = MinMaxScaler(feature_range=(0, np.pi))
        self.selector = None

        self.X_train_raw = None
        self.X_test_raw = None
        self.y_train = None
        self.y_test = None
        self.X_train_classical = None
        self.X_test_classical = None
        self.X_train_quantum = None
        self.X_test_quantum = None

    @staticmethod
    def _subject_from_name(value):
        value = str(value)
        # UCI names look like phon_R01_S01_1 ... phon_R01_S01_6.
        # Remove only the final recording suffix so all recordings from one person stay together.
        return re.sub(r"[_-]\d+$", "", value)

    def load_data(self):
        if not os.path.exists(self.filepath):
            raise FileNotFoundError(
                f"Parkinson's dataset not found: {self.filepath}. "
                "Run scripts/prepare_parkinsons_data.ps1 first."
            )

        try:
            df = pd.read_csv(self.filepath, sep=None, engine="python")
        except Exception:
            df = pd.read_csv(self.filepath)

        if df.empty:
            raise ValueError("Dataset is empty")

        self.df_raw = df.copy()

        self.target_name = next((c for c in self.TARGET_CANDIDATES if c in df.columns), None)
        if self.target_name is None:
            raise ValueError(
                "No binary target column found. Expected one of: " + ", ".join(self.TARGET_CANDIDATES)
            )

        self.id_column = next((c for c in self.ID_CANDIDATES if c in df.columns), None)
        y_series = pd.to_numeric(df[self.target_name], errors="coerce")
        if y_series.isna().any():
            # Fall back to deterministic categorical encoding.
            unique = [x for x in df[self.target_name].dropna().unique()]
            if len(unique) != 2:
                raise ValueError(f"Target {self.target_name!r} is not binary: {unique}")
            mapping = {v: i for i, v in enumerate(sorted(unique, key=str))}
            y_series = df[self.target_name].map(mapping).astype(int)

        if not set(pd.unique(y_series)).issubset({0, 1}):
            raise ValueError(f"Target {self.target_name!r} must encode control=0 and Parkinson's=1")

        drop_cols = [self.target_name]
        if self.id_column:
            drop_cols.append(self.id_column)
        X_df = df.drop(columns=drop_cols, errors="ignore").copy()

        # Keep only usable numeric predictors. Categorical metadata is intentionally excluded
        # rather than silently ordinal-encoding it into a medical model.
        for col in X_df.columns:
            X_df[col] = pd.to_numeric(X_df[col], errors="coerce")
        X_df = X_df.dropna(axis=1, how="all")
        if X_df.shape[1] < 2:
            raise ValueError("Not enough numeric features for training")

        self.feature_names = list(X_df.columns)
        self.df_features_numeric = X_df
        y = y_series.to_numpy(dtype=int)

        if self.id_column:
            raw_ids = df[self.id_column].astype(str)
            if self.id_column.lower() == "name":
                self.groups = raw_ids.map(self._subject_from_name).to_numpy()
            else:
                self.groups = raw_ids.to_numpy()
        else:
            self.groups = np.arange(len(df)).astype(str)

        # Prevent repeated recordings from the same participant appearing in train and test.
        # The UCI Parkinson's classification dataset has multiple recordings per participant.
        unique_groups = np.unique(self.groups)
        if self.id_column and len(unique_groups) >= 8:
            splitter = StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=self.random_state)
            train_idx, test_idx = next(splitter.split(X_df, y, groups=self.groups))
            self.split_strategy = "participant-grouped stratified split"
        else:
            idx = np.arange(len(df))
            train_idx, test_idx = train_test_split(
                idx, test_size=self.test_size, random_state=self.random_state, stratify=y
            )
            self.split_strategy = "row-stratified split"

        self.train_indices = np.asarray(train_idx)
        self.test_indices = np.asarray(test_idx)

        X_train_df = X_df.iloc[self.train_indices]
        X_test_df = X_df.iloc[self.test_indices]
        self.y_train = y[self.train_indices]
        self.y_test = y[self.test_indices]

        # Fit all preprocessing only on training data to avoid leakage.
        self.X_train_raw = self.imputer.fit_transform(X_train_df)
        self.X_test_raw = self.imputer.transform(X_test_df)
        self._preprocess_features()
        return self

    def _preprocess_features(self):
        self.X_train_classical = self.scaler_classical.fit_transform(self.X_train_raw)
        self.X_test_classical = self.scaler_classical.transform(self.X_test_raw)

        k = min(self.n_qubits, self.X_train_classical.shape[1])
        if self.reduction_method == "pca":
            self.selector = PCA(n_components=k, random_state=self.random_state)
            tr = self.selector.fit_transform(self.X_train_classical)
            te = self.selector.transform(self.X_test_classical)
            self.selected_feature_indices = list(range(k))
            self.selected_feature_names = [f"PC_{i + 1}" for i in range(k)]
        else:
            self.selector = SelectKBest(score_func=f_classif, k=k)
            tr = self.selector.fit_transform(self.X_train_classical, self.y_train)
            te = self.selector.transform(self.X_test_classical)
            self.selected_feature_indices = list(self.selector.get_support(indices=True))
            self.selected_feature_names = [self.feature_names[i] for i in self.selected_feature_indices]

        self.X_train_quantum = self.scaler_quantum.fit_transform(tr)
        self.X_test_quantum = self.scaler_quantum.transform(te)

    def transform_single_patient(self, patient_dict):
        row = []
        for name in self.feature_names:
            value = patient_dict.get(name, patient_dict.get(name.lower(), np.nan))
            if value in (None, ""):
                value = np.nan
            try:
                row.append(float(value))
            except (TypeError, ValueError):
                row.append(np.nan)

        row_arr = np.asarray([row], dtype=float)
        row_imputed = self.imputer.transform(row_arr)
        row_classical = self.scaler_classical.transform(row_imputed)
        row_reduced = self.selector.transform(row_classical)
        row_quantum = self.scaler_quantum.transform(row_reduced)
        return row_classical, row_quantum

    def get_dataset_summary(self):
        y = pd.to_numeric(self.df_raw[self.target_name], errors="coerce").fillna(0).astype(int)
        train_groups = np.unique(self.groups[self.train_indices]) if self.groups is not None and self.train_indices is not None else []
        test_groups = np.unique(self.groups[self.test_indices]) if self.groups is not None and self.test_indices is not None else []
        return {
            "dataset_name": "UCI Oxford Parkinson's Disease Detection (voice measurements)",
            "total_samples": int(len(self.df_raw)),
            "train_samples": int(len(self.y_train)),
            "test_samples": int(len(self.y_test)),
            "total_features": int(len(self.feature_names)),
            "target_name": self.target_name,
            "target_definition": "status: control=0 / Parkinson's cohort=1",
            "id_column": self.id_column,
            "feature_names": self.feature_names,
            "n_qubits": self.n_qubits,
            "selected_features": self.selected_feature_names,
            "reduction_method": self.reduction_method,
            "split_strategy": self.split_strategy,
            "unique_participants": int(len(np.unique(self.groups))) if self.groups is not None else None,
            "train_participants": int(len(train_groups)),
            "test_participants": int(len(test_groups)),
            "train_class_distribution": {
                "control (0)": int((self.y_train == 0).sum()),
                "parkinsons (1)": int((self.y_train == 1).sum()),
            },
            "test_class_distribution": {
                "control (0)": int((self.y_test == 0).sum()),
                "parkinsons (1)": int((self.y_test == 1).sum()),
            },
            "class_distribution": {
                "control (0)": int((y == 0).sum()),
                "parkinsons (1)": int((y == 1).sum()),
            },
        }

    def get_cohort_statistics(self):
        df = self.df_raw
        y = pd.to_numeric(df[self.target_name], errors="coerce").fillna(0).astype(int)
        stats = {}
        for feat in self.feature_names:
            vals = pd.to_numeric(df[feat], errors="coerce")
            ctrl = vals[y == 0].dropna()
            pdv = vals[y == 1].dropna()
            allv = vals.dropna()
            stats[feat] = {
                "display_name": self.FEATURE_DISPLAY.get(feat, feat),
                "control_mean": float(ctrl.mean()) if len(ctrl) else 0.0,
                "control_std": float(ctrl.std(ddof=0)) if len(ctrl) else 0.0,
                "parkinsons_mean": float(pdv.mean()) if len(pdv) else 0.0,
                "parkinsons_std": float(pdv.std(ddof=0)) if len(pdv) else 0.0,
                "overall_median": float(allv.median()) if len(allv) else 0.0,
            }
        return stats

    def compare_patient_with_cohort(self, patient_dict):
        stats = self.get_cohort_statistics()
        comparisons = []
        pd_lean = 0
        control_lean = 0

        for feat in self.feature_names:
            val = patient_dict.get(feat, patient_dict.get(feat.lower(), None))
            if val in (None, ""):
                continue
            try:
                val = float(val)
            except (TypeError, ValueError):
                continue

            s = stats[feat]
            pooled = max((s["control_std"] + s["parkinsons_std"]) / 2.0, 1e-9)
            d_control = abs(val - s["control_mean"]) / pooled
            d_pd = abs(val - s["parkinsons_mean"]) / pooled
            if abs(d_control - d_pd) < 0.15:
                alignment = "Neutral / overlapping"
                level = "neutral"
            elif d_pd < d_control:
                alignment = "Closer to Parkinson's cohort"
                level = "pd"
                pd_lean += 1
            else:
                alignment = "Closer to control cohort"
                level = "control"
                control_lean += 1

            col = pd.to_numeric(self.df_raw[feat], errors="coerce").dropna().to_numpy()
            percentile = float(np.mean(col <= val) * 100.0) if len(col) else 50.0
            comparisons.append({
                "feature": feat,
                "display_name": s["display_name"],
                "patient_value": round(val, 6),
                "control_mean": round(s["control_mean"], 6),
                "parkinsons_mean": round(s["parkinsons_mean"], 6),
                "percentile": round(percentile, 1),
                "alignment": alignment,
                "alignment_level": level,
            })

        return {
            "comparisons": comparisons,
            "pd_leaning_features": pd_lean,
            "control_leaning_features": control_lean,
            "note": "Cohort similarity is descriptive and is not a clinical normal-range assessment.",
        }

    def get_demo_sample(self, label):
        label = int(label)
        candidates = [i for i in self.test_indices if int(self.df_raw.iloc[i][self.target_name]) == label]
        if not candidates:
            candidates = [i for i in range(len(self.df_raw)) if int(self.df_raw.iloc[i][self.target_name]) == label]
        idx = int(candidates[0])
        row = self.df_features_numeric.iloc[idx]
        return {k: float(row[k]) for k in self.feature_names if pd.notna(row[k])}
