import os
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import mean_absolute_error, r2_score


class ParkinsonSeverityRegressor:
    """Experimental UPDRS symptom-burden regression using the UCI telemonitoring cohort.

    The telemonitoring dataset contains people with early-stage Parkinson's disease only.
    Therefore this model is never used to diagnose PD; it is only shown after an elevated
    classification-pattern result, and its output is explicitly labelled as an experimental
    dataset-target estimate.
    """

    CLASSIFICATION_TO_TELE = {
        "MDVP:Jitter(%)": "Jitter(%)",
        "MDVP:Jitter(Abs)": "Jitter(Abs)",
        "MDVP:RAP": "Jitter:RAP",
        "Jitter:DDP": "Jitter:DDP",
        "MDVP:Shimmer": "Shimmer",
        "MDVP:Shimmer(dB)": "Shimmer(dB)",
        "Shimmer:APQ3": "Shimmer:APQ3",
        "Shimmer:APQ5": "Shimmer:APQ5",
        "Shimmer:DDA": "Shimmer:DDA",
        "NHR": "NHR",
        "HNR": "HNR",
        "RPDE": "RPDE",
        "DFA": "DFA",
        "PPE": "PPE",
    }

    def __init__(self, filepath="data/parkinsons_updrs.data", random_state=42):
        self.filepath = filepath
        self.random_state = random_state
        self.df = None
        self.features = []
        self.imputer = SimpleImputer(strategy="median")
        self.motor_model = RandomForestRegressor(
            n_estimators=250, max_depth=10, min_samples_leaf=2,
            random_state=random_state, n_jobs=-1
        )
        self.total_model = RandomForestRegressor(
            n_estimators=250, max_depth=10, min_samples_leaf=2,
            random_state=random_state, n_jobs=-1
        )
        self.metrics = {}
        self.motor_values = None
        self.total_values = None

    def fit(self):
        if not os.path.exists(self.filepath):
            raise FileNotFoundError(self.filepath)
        df = pd.read_csv(self.filepath)
        required = {"subject#", "motor_UPDRS", "total_UPDRS"}
        if not required.issubset(df.columns):
            raise ValueError("Telemonitoring file does not contain expected UPDRS columns")

        # Use only voice measures with exact, defensible counterparts in the classification dataset.
        self.features = [v for v in self.CLASSIFICATION_TO_TELE.values() if v in df.columns]
        if len(self.features) < 6:
            raise ValueError("Too few shared voice features for UPDRS regression")

        X_df = df[self.features].apply(pd.to_numeric, errors="coerce")
        y_motor = pd.to_numeric(df["motor_UPDRS"], errors="coerce").to_numpy(float)
        y_total = pd.to_numeric(df["total_UPDRS"], errors="coerce").to_numpy(float)
        groups = df["subject#"].astype(str).to_numpy()

        splitter = GroupShuffleSplit(n_splits=1, test_size=0.22, random_state=self.random_state)
        train_idx, test_idx = next(splitter.split(X_df, y_motor, groups=groups))

        X_train = self.imputer.fit_transform(X_df.iloc[train_idx])
        X_test = self.imputer.transform(X_df.iloc[test_idx])
        self.motor_model.fit(X_train, y_motor[train_idx])
        self.total_model.fit(X_train, y_total[train_idx])

        p_motor = self.motor_model.predict(X_test)
        p_total = self.total_model.predict(X_test)
        self.metrics = {
            "motor_mae": round(float(mean_absolute_error(y_motor[test_idx], p_motor)), 2),
            "motor_r2": round(float(r2_score(y_motor[test_idx], p_motor)), 3),
            "total_mae": round(float(mean_absolute_error(y_total[test_idx], p_total)), 2),
            "total_r2": round(float(r2_score(y_total[test_idx], p_total)), 3),
            "test_subjects": int(len(np.unique(groups[test_idx]))),
            "train_subjects": int(len(np.unique(groups[train_idx]))),
        }
        self.df = df
        self.motor_values = y_motor
        self.total_values = y_total
        return self

    def predict_from_classification_features(self, patient):
        row = {f: np.nan for f in self.features}
        reverse = self.CLASSIFICATION_TO_TELE
        used = 0
        for cls_name, tele_name in reverse.items():
            if tele_name not in row:
                continue
            value = patient.get(cls_name, patient.get(cls_name.lower(), None))
            if value not in (None, ""):
                try:
                    row[tele_name] = float(value)
                    used += 1
                except (TypeError, ValueError):
                    pass

        X = pd.DataFrame([row], columns=self.features)
        Xi = self.imputer.transform(X)
        motor = float(self.motor_model.predict(Xi)[0])
        total = float(self.total_model.predict(Xi)[0])
        motor_pct = float(np.mean(self.motor_values <= motor) * 100.0)
        total_pct = float(np.mean(self.total_values <= total) * 100.0)
        return {
            "available": True,
            "estimated_motor_updrs": round(motor, 1),
            "estimated_total_updrs": round(total, 1),
            "motor_cohort_percentile": round(motor_pct, 1),
            "total_cohort_percentile": round(total_pct, 1),
            "shared_voice_features_used": used,
            "shared_voice_features_total": len(self.features),
            "validation": self.metrics,
            "note": (
                "Experimental symptom-burden estimate trained on the UCI Parkinson's telemonitoring cohort. "
                "It is not a Hoehn & Yahr stage, not a measure of neuronal damage, and not a clinical UPDRS examination."
            ),
        }
