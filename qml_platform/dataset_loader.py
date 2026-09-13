import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer

class BiomedicalDatasetLoader:
    """
    Data Ingestion and Preprocessing Pipeline for Biomedical Data.
    Handles loading, imputation, scaling, and dimensionality reduction for QML.
    """
    def __init__(self, filepath="inputdata.txt", n_qubits=4, reduction_method="select_k_best", test_size=0.25, random_state=42):
        self.filepath = filepath
        self.n_qubits = n_qubits
        self.reduction_method = reduction_method
        self.test_size = test_size
        self.random_state = random_state

        self.df_raw = None
        self.feature_names = []
        self.target_name = "Diagnosis"
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

    def load_data(self):
        """Loads data from inputdata.txt or fallback inputdata.txt.txt."""
        target_path = self.filepath
        if not os.path.exists(target_path):
            alt_path = self.filepath + ".txt"
            if os.path.exists(alt_path):
                target_path = alt_path
            elif os.path.exists("inputdata.txt"):
                target_path = "inputdata.txt"
            elif os.path.exists("inputdata.txt.txt"):
                target_path = "inputdata.txt.txt"
            else:
                raise FileNotFoundError(f"Cannot locate dataset file: {self.filepath}")

        # Auto-detect delimiter
        try:
            self.df_raw = pd.read_csv(target_path, sep=None, engine='python')
        except Exception:
            self.df_raw = pd.read_csv(target_path)

        # Identify target column
        candidate_targets = ['Diagnosis', 'diagnosis', 'target', 'Target', 'label', 'Label', 'outcome', 'Outcome', 'status', 'Status']
        found_target = None
        for col in candidate_targets:
            if col in self.df_raw.columns:
                found_target = col
                break
        
        if found_target:
            self.target_name = found_target
        else:
            self.target_name = self.df_raw.columns[-1]

        # Separate features and target
        X_df = self.df_raw.drop(columns=[self.target_name])
        y_series = self.df_raw[self.target_name]

        # Handle binary classification target encoding
        if y_series.dtype == 'object' or set(y_series.unique()) - {0, 1}:
            unique_vals = y_series.dropna().unique()
            val_map = {val: i for i, val in enumerate(sorted(unique_vals))}
            y_series = y_series.map(val_map)

        # Convert categorical features to numeric if any
        for col in X_df.columns:
            if X_df[col].dtype == 'object':
                X_df[col] = pd.factorize(X_df[col])[0]

        self.feature_names = list(X_df.columns)
        X_imputed = self.imputer.fit_transform(X_df)
        y = y_series.values.astype(int)

        # Stratified train-test split
        self.X_train_raw, self.X_test_raw, self.y_train, self.y_test = train_test_split(
            X_imputed, y, test_size=self.test_size, random_state=self.random_state, stratify=y
        )

        self._preprocess_features()
        return self

    def _preprocess_features(self):
        """Prepares classical and quantum-encoded features."""
        # 1. Classical Scaling (StandardScaler)
        self.X_train_classical = self.scaler_classical.fit_transform(self.X_train_raw)
        self.X_test_classical = self.scaler_classical.transform(self.X_test_raw)

        # 2. Dimensionality Reduction for Quantum Circuit
        n_features = self.X_train_raw.shape[1]
        k = min(self.n_qubits, n_features)

        if self.reduction_method == "select_k_best":
            self.selector = SelectKBest(score_func=f_classif, k=k)
            X_tr_reduced = self.selector.fit_transform(self.X_train_classical, self.y_train)
            X_te_reduced = self.selector.transform(self.X_test_classical)
            self.selected_feature_indices = list(self.selector.get_support(indices=True))
            self.selected_feature_names = [self.feature_names[i] for i in self.selected_feature_indices]
        elif self.reduction_method == "pca":
            self.selector = PCA(n_components=k, random_state=self.random_state)
            X_tr_reduced = self.selector.fit_transform(self.X_train_classical)
            X_te_reduced = self.selector.transform(self.X_test_classical)
            self.selected_feature_indices = list(range(k))
            self.selected_feature_names = [f"PC_{i+1}" for i in range(k)]
        else:
            X_tr_reduced = self.X_train_classical[:, :k]
            X_te_reduced = self.X_test_classical[:, :k]
            self.selected_feature_indices = list(range(k))
            self.selected_feature_names = self.feature_names[:k]

        # 3. Quantum Scaling: Angle embedding requires input in [0, pi]
        self.X_train_quantum = self.scaler_quantum.fit_transform(X_tr_reduced)
        self.X_test_quantum = self.scaler_quantum.transform(X_te_reduced)

    def transform_single_patient(self, patient_dict):
        """Transforms a single patient's raw features into classical and quantum representations."""
        row = []
        for name in self.feature_names:
            val = patient_dict.get(name, patient_dict.get(name.lower(), 0.0))
            row.append(float(val))

        row_arr = np.array([row])
        row_imputed = self.imputer.transform(row_arr)
        row_classical = self.scaler_classical.transform(row_imputed)

        if self.reduction_method in ["select_k_best", "pca"] and self.selector is not None:
            row_reduced = self.selector.transform(row_classical)
        else:
            row_reduced = row_classical[:, :min(self.n_qubits, row_classical.shape[1])]

        row_quantum = self.scaler_quantum.transform(row_reduced)
        return row_classical, row_quantum

    def get_dataset_summary(self):
        """Returns statistical overview of the loaded biomedical dataset."""
        return {
            "total_samples": len(self.df_raw),
            "train_samples": len(self.y_train),
            "test_samples": len(self.y_test),
            "total_features": len(self.feature_names),
            "target_name": self.target_name,
            "feature_names": self.feature_names,
            "n_qubits": self.n_qubits,
            "selected_features": self.selected_feature_names,
            "reduction_method": self.reduction_method,
            "class_distribution": {
                "negative (0)": int(np.sum(self.df_raw[self.target_name] == 0)),
                "positive (1)": int(np.sum(self.df_raw[self.target_name] == 1))
            }
        }

    CLINICAL_NORMAL_RANGES = {
        "Age": {"min": 18, "max": 85, "unit": "years", "normal_str": "18 - 85 yrs"},
        "Gender": {"min": 0, "max": 1, "unit": "code", "normal_str": "0 (Female) / 1 (Male)"},
        "BMI": {"min": 18.5, "max": 24.9, "unit": "kg/m²", "normal_str": "18.5 - 24.9"},
        "Total_Bilirubin": {"min": 0.2, "max": 1.2, "unit": "mg/dL", "normal_str": "0.2 - 1.2 mg/dL"},
        "Direct_Bilirubin": {"min": 0.0, "max": 0.3, "unit": "mg/dL", "normal_str": "0.0 - 0.3 mg/dL"},
        "Alkaline_Phosphatase": {"min": 44, "max": 147, "unit": "IU/L", "normal_str": "44 - 147 IU/L"},
        "Alamine_Aminotransferase": {"min": 7, "max": 56, "unit": "IU/L", "normal_str": "7 - 56 IU/L"},
        "Aspartate_Aminotransferase": {"min": 10, "max": 40, "unit": "IU/L", "normal_str": "10 - 40 IU/L"},
        "Total_Proteins": {"min": 6.0, "max": 8.3, "unit": "g/dL", "normal_str": "6.0 - 8.3 g/dL"},
        "Albumin": {"min": 3.5, "max": 5.0, "unit": "g/dL", "normal_str": "3.5 - 5.0 g/dL"},
        "Albumin_and_Globulin_Ratio": {"min": 1.0, "max": 2.5, "unit": "ratio", "normal_str": "1.0 - 2.5"},
        "Fasting_Glucose": {"min": 70, "max": 99, "unit": "mg/dL", "normal_str": "70 - 99 mg/dL"},
        "Serum_Cholesterol": {"min": 120, "max": 199, "unit": "mg/dL", "normal_str": "< 200 mg/dL"},
        "Platelet_Count": {"min": 150, "max": 450, "unit": "x10³/µL", "normal_str": "150 - 450 x10³/µL"}
    }

    def get_cohort_statistics(self):
        """Computes statistical distributions for healthy (0) vs diseased (1) cohorts in the dataset."""
        if self.df_raw is None:
            self.load_data()

        df = self.df_raw
        target = self.target_name
        df_healthy = df[df[target] == 0]
        df_diseased = df[df[target] == 1]

        stats = {}
        for feat in self.feature_names:
            h_vals = pd.to_numeric(df_healthy[feat], errors='coerce').dropna()
            d_vals = pd.to_numeric(df_diseased[feat], errors='coerce').dropna()

            ref = self.CLINICAL_NORMAL_RANGES.get(feat, {"min": 0, "max": 100, "unit": "", "normal_str": "N/A"})
            stats[feat] = {
                "healthy_mean": round(float(h_vals.mean()), 2) if len(h_vals) else 0.0,
                "healthy_median": round(float(h_vals.median()), 2) if len(h_vals) else 0.0,
                "healthy_std": round(float(h_vals.std()), 2) if len(h_vals) else 0.0,
                "diseased_mean": round(float(d_vals.mean()), 2) if len(d_vals) else 0.0,
                "diseased_median": round(float(d_vals.median()), 2) if len(d_vals) else 0.0,
                "diseased_std": round(float(d_vals.std()), 2) if len(d_vals) else 0.0,
                "normal_range": ref["normal_str"],
                "unit": ref["unit"]
            }
        return stats

    def compare_patient_with_cohort(self, patient_dict):
        """
        Compares an individual patient's vitals directly against the feeded dataset distributions.
        Returns biomarker-level deviation indicators, healthy vs diseased cohort means,
        and percentile ranks.
        """
        cohort_stats = self.get_cohort_statistics()
        comparisons = []
        abnormal_count = 0

        for feat in self.feature_names:
            val = float(patient_dict.get(feat, patient_dict.get(feat.lower(), 0.0)))
            c_stat = cohort_stats.get(feat, {})
            ref = self.CLINICAL_NORMAL_RANGES.get(feat, {"min": 0, "max": 100, "unit": ""})

            # Determine clinical deviation status
            min_val = ref.get("min", 0)
            max_val = ref.get("max", 100)

            if feat == "Gender":
                status = "Male" if val == 1 else "Female"
                status_level = "info"
            elif val < min_val:
                status = "Subnormal / Low"
                status_level = "warning"
                abnormal_count += 1
            elif val > max_val:
                # Severity check
                if val > (max_val * 1.5):
                    status = "Severely Elevated"
                    status_level = "danger"
                else:
                    status = "Elevated"
                    status_level = "warning"
                abnormal_count += 1
            else:
                status = "Optimal / Normal"
                status_level = "success"

            # Compute patient percentile against total dataset
            if feat in self.df_raw.columns:
                col_vals = pd.to_numeric(self.df_raw[feat], errors='coerce').dropna().values
                percentile = round(float(np.mean(col_vals <= val) * 100), 1)
            else:
                percentile = 50.0

            comparisons.append({
                "biomarker": feat,
                "display_name": feat.replace("_", " "),
                "patient_value": val,
                "unit": ref.get("unit", ""),
                "normal_range": c_stat.get("normal_range", ""),
                "healthy_cohort_mean": c_stat.get("healthy_mean", 0.0),
                "diseased_cohort_mean": c_stat.get("diseased_mean", 0.0),
                "status": status,
                "status_level": status_level,
                "percentile": percentile
            })

        return {
            "comparisons": comparisons,
            "abnormal_biomarkers_count": abnormal_count,
            "total_biomarkers_evaluated": len(comparisons),
            "summary_risk_flag": "High Abnormality" if abnormal_count >= 3 else ("Moderate" if abnormal_count >= 1 else "Normal Profile")
        }
