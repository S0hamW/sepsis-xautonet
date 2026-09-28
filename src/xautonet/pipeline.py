"""Unified end-to-end training, validation, and benchmarking pipeline for XAutoNet."""

import os
import json
from typing import Dict, Any, List, Tuple, Optional
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import (
    KFold,
    StratifiedKFold,
    StratifiedGroupKFold,
    GroupKFold,
)
from sklearn.preprocessing import StandardScaler

from xautonet.config import (
    FILTER_FEATURES,
    BOTTLENECK_FEATURES,
    DHM_BETA,
    AUTOENCODER_CONFIG,
    CLASSIFIER_CONFIG,
    CLINICAL_NORMAL_RANGES,
)

from xautonet.data.loader import generate_synthetic_physionet_cohort
from xautonet.data.imputation import MICEImputerWrapper
from xautonet.data.outliers import handle_clinical_outliers
from xautonet.data.feature_engineering import compute_sirs_score
from xautonet.data.balancing import balance_clinical_dataset
from xautonet.models.autoencoder import Conv1DAutoencoder
from xautonet.models.classifier import XAutoNetClassifier, evaluate_classifier_metrics
from xautonet.models.baselines import benchmark_baseline_models
from xautonet.xai.gradcam import GradCAM1D, compute_discounting_heatmap
from xautonet.xai.shap_explainer import DeepSHAPExplainer


class XAutoNetPipeline:
    """Manages the full lifecycle of the XAutoNet Sepsis Prediction system."""

    def __init__(
        self,
        artifact_dir: str = "artifacts",
        device: Optional[str] = None,
    ):
        self.artifact_dir = artifact_dir
        os.makedirs(self.artifact_dir, exist_ok=True)
        self.device = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))

        self.scaler = StandardScaler()
        self.autoencoder = Conv1DAutoencoder(input_dim=19, latent_dim=11).to(self.device)
        self.classifier = XAutoNetClassifier(input_dim=11).to(self.device)
        self.gradcam = None
        self.shap_explainer = None
        self.is_trained = False

    def train_autoencoder(
        self,
        X_train: np.ndarray,
        epochs: int = 15,
        batch_size: int = 128,
        lr: float = 0.001,
    ) -> List[float]:
        """Train 1D CNN Autoencoder with MSE reconstruction loss."""
        tensor_x = torch.tensor(X_train, dtype=torch.float32)
        dataset = TensorDataset(tensor_x)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        optimizer = optim.Adam(self.autoencoder.parameters(), lr=lr)
        criterion = nn.MSELoss()
        losses = []

        self.autoencoder.train()
        for epoch in range(epochs):
            epoch_loss = 0.0
            for batch in loader:
                x_b = batch[0].to(self.device)
                optimizer.zero_grad()
                _, recon = self.autoencoder(x_b)
                loss = criterion(recon, x_b)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item() * len(x_b)
            losses.append(epoch_loss / len(X_train))

        return losses

    def train_classifier(
        self,
        X_latent_train: np.ndarray,
        y_train: np.ndarray,
        epochs: int = 15,
        batch_size: int = 128,
        lr: float = 0.001,
    ) -> List[float]:
        """Train XAutoNet classifier on 11-dimensional latent features with BCE loss."""
        tensor_x = torch.tensor(X_latent_train, dtype=torch.float32)
        tensor_y = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
        dataset = TensorDataset(tensor_x, tensor_y)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        optimizer = optim.Adam(self.classifier.parameters(), lr=lr)
        criterion = nn.BCELoss()
        losses = []

        self.classifier.train()
        for epoch in range(epochs):
            epoch_loss = 0.0
            for batch in loader:
                x_b, y_b = batch[0].to(self.device), batch[1].to(self.device)
                optimizer.zero_grad()
                pred = self.classifier(x_b)
                loss = criterion(pred, y_b)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item() * len(x_b)
            losses.append(epoch_loss / len(X_latent_train))

        return losses

    def run_5fold_cross_validation(
        self,
        X: Any,
        y: Any,
        groups: Optional[Any] = None,
        n_splits: int = 5,
        epochs_ae: int = 10,
        epochs_clf: int = 10,
        balance_training: bool = True,
        target_normal: Optional[int] = None,
        target_sepsis: Optional[int] = None,
    ) -> pd.DataFrame:
        """Evaluate XAutoNet using strict leakage-free 5-Fold Cross Validation.

        Requirements enforced:
        1. Patient/train/test separation happens before fitting any preprocessing.
        2. MICE imputation is fitted exclusively on training data within each fold.
        3. SMOTE/balancing happens strictly on the training portion of each fold.
        4. Validation data is never balanced and never influences preprocessing.
        """
        # Format X, y, and groups
        if isinstance(X, pd.DataFrame):
            feature_names = list(X.columns)
            X_mat = X.values.astype(np.float64)
        else:
            feature_names = FILTER_FEATURES
            X_mat = np.asarray(X, dtype=np.float64)

        if isinstance(y, pd.Series):
            y_arr = y.values.astype(int)
        else:
            y_arr = np.asarray(y, dtype=int)

        # 1. Patient / train / test separation BEFORE fitting preprocessing
        if groups is not None:
            groups_arr = np.asarray(groups)
            try:
                splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=42)
                splits = list(splitter.split(X_mat, y_arr, groups=groups_arr))
            except Exception:
                splitter = GroupKFold(n_splits=n_splits)
                splits = list(splitter.split(X_mat, y_arr, groups=groups_arr))
        else:
            try:
                splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
                splits = list(splitter.split(X_mat, y_arr))
            except Exception:
                splitter = KFold(n_splits=n_splits, shuffle=True, random_state=42)
                splits = list(splitter.split(X_mat, y_arr))

        fold_results = []

        for fold, (train_idx, val_idx) in enumerate(splits, start=1):
            X_tr_raw, y_tr = X_mat[train_idx], y_arr[train_idx]
            X_val_raw, y_val = X_mat[val_idx], y_arr[val_idx]

            df_tr = pd.DataFrame(X_tr_raw, columns=feature_names)
            df_val = pd.DataFrame(X_val_raw, columns=feature_names)

            # 2. MICE imputation fitted ONLY on training data within this fold
            if df_tr.isnull().values.any() or df_val.isnull().values.any():
                mice = MICEImputerWrapper(max_iter=10, random_state=42 + fold, features=feature_names)
                mice.fit(df_tr)
                df_tr_imp = mice.transform(df_tr)
                df_val_imp = mice.transform(df_val)
            else:
                df_tr_imp = df_tr
                df_val_imp = df_val

            # Handle outliers within physiological permissible bounds
            df_tr_clean = handle_clinical_outliers(df_tr_imp)
            df_val_clean = handle_clinical_outliers(df_val_imp)

            X_tr_clean = df_tr_clean.values.astype(np.float64)
            X_val_clean = df_val_clean.values.astype(np.float64)

            # 3. SMOTE/oversampling/balancing happens ONLY on the training portion
            if balance_training and (len(np.unique(y_tr)) > 1):
                n_tr_norm = int(np.sum(y_tr == 0))
                n_tr_sep = int(np.sum(y_tr == 1))

                if target_normal is not None and target_sepsis is not None:
                    tgt_norm = target_normal
                    tgt_sep = target_sepsis
                else:
                    tgt_norm = min(32000, n_tr_norm)
                    tgt_sep = min(25000, max(n_tr_sep, int(tgt_norm * 0.78)))

                X_tr_bal, y_tr_bal = balance_clinical_dataset(
                    X_tr_clean,
                    y_tr,
                    target_normal=tgt_norm,
                    target_sepsis=tgt_sep,
                    random_state=42 + fold,
                )
            else:
                X_tr_bal, y_tr_bal = X_tr_clean, y_tr

            # 4. Validation data must never influence preprocessing or balancing
            # Fit scaler ONLY on the balanced training data
            fold_scaler = StandardScaler().fit(X_tr_bal)
            X_tr_scaled = fold_scaler.transform(X_tr_bal)
            X_val_scaled = fold_scaler.transform(X_val_clean)

            # Train fold AE on training data
            ae = Conv1DAutoencoder(input_dim=19, latent_dim=11).to(self.device)
            tensor_x = torch.tensor(X_tr_scaled, dtype=torch.float32)
            loader = DataLoader(TensorDataset(tensor_x), batch_size=128, shuffle=True)
            opt_ae = optim.Adam(ae.parameters(), lr=0.001)
            crit_ae = nn.MSELoss()

            ae.train()
            for _ in range(epochs_ae):
                for b in loader:
                    xb = b[0].to(self.device)
                    opt_ae.zero_grad()
                    _, recon = ae(xb)
                    crit_ae(recon, xb).backward()
                    opt_ae.step()

            # Extract latent representations
            ae.eval()
            with torch.no_grad():
                lat_tr = ae.encode(torch.tensor(X_tr_scaled, dtype=torch.float32).to(self.device)).cpu().numpy()
                lat_val = ae.encode(torch.tensor(X_val_scaled, dtype=torch.float32).to(self.device)).cpu().numpy()

            # Train fold classifier on training fold
            clf = XAutoNetClassifier(input_dim=11).to(self.device)
            loader_clf = DataLoader(
                TensorDataset(torch.tensor(lat_tr, dtype=torch.float32), torch.tensor(y_tr_bal, dtype=torch.float32).unsqueeze(1)),
                batch_size=128,
                shuffle=True,
            )
            opt_clf = optim.Adam(clf.parameters(), lr=0.001)
            crit_clf = nn.BCELoss()

            clf.train()
            for _ in range(epochs_clf):
                for xb, yb in loader_clf:
                    xb, yb = xb.to(self.device), yb.to(self.device)
                    opt_clf.zero_grad()
                    crit_clf(clf(xb), yb).backward()
                    opt_clf.step()

            # Evaluate fold metrics on untouched validation fold
            clf.eval()
            with torch.no_grad():
                val_probs = clf(torch.tensor(lat_val, dtype=torch.float32).to(self.device)).cpu().numpy().squeeze()
                if np.ndim(val_probs) == 0:
                    val_probs = np.array([float(val_probs)])
                y_val_pred = (val_probs >= 0.5).astype(int)

            metrics = evaluate_classifier_metrics(y_val, y_val_pred)
            metrics["Fold"] = str(fold)
            fold_results.append(metrics)

        df = pd.DataFrame(fold_results)[["Fold", "F1 Score", "Precision", "Recall", "Accuracy"]]
        
        # Calculate Mean and SD
        mean_row = {
            "Fold": "Mean ± SD",
            "F1 Score": f"{df['F1 Score'].mean():.2f} ± {df['F1 Score'].std():.3f}",
            "Precision": f"{df['Precision'].mean():.2f} ± {df['Precision'].std():.3f}",
            "Recall": f"{df['Recall'].mean():.2f} ± {df['Recall'].std():.3f}",
            "Accuracy": f"{df['Accuracy'].mean():.2f} ± {df['Accuracy'].std():.3f}",
        }
        df = pd.concat([df, pd.DataFrame([mean_row])], ignore_index=True)
        return df

    def run_paper_5fold_evaluation(
        self,
        X: Any,
        y: Any,
        n_splits: int = 5,
        epochs_ae: int = 10,
        epochs_clf: int = 10,
    ) -> pd.DataFrame:
        """Paper-style 5-Fold Cross Validation evaluated directly on the balanced cohort.

        Reproduces the Table I evaluation protocol from Section III.A of the IEEE IRI 2023 paper:
        - Evaluates directly on the balanced 57,000-instance dataset (32k normal + 25k sepsis).
        - StratifiedKFold without group separation (since instances are hourly samples).
        - Uncalibrated 0.5 decision threshold as used in the paper.
        """
        if isinstance(X, pd.DataFrame):
            X_mat = X.values.astype(np.float64)
        else:
            X_mat = np.asarray(X, dtype=np.float64)

        if isinstance(y, pd.Series):
            y_arr = y.values.astype(int)
        else:
            y_arr = np.asarray(y, dtype=int)

        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        fold_results = []

        for fold, (train_idx, val_idx) in enumerate(skf.split(X_mat, y_arr), start=1):
            X_tr, y_tr = X_mat[train_idx], y_arr[train_idx]
            X_val, y_val = X_mat[val_idx], y_arr[val_idx]

            fold_scaler = StandardScaler().fit(X_tr)
            X_tr_scaled = fold_scaler.transform(X_tr)
            X_val_scaled = fold_scaler.transform(X_val)

            ae = Conv1DAutoencoder(input_dim=19, latent_dim=11).to(self.device)
            tensor_x = torch.tensor(X_tr_scaled, dtype=torch.float32)
            loader = DataLoader(TensorDataset(tensor_x), batch_size=128, shuffle=True)
            opt_ae = optim.Adam(ae.parameters(), lr=0.001)
            crit_ae = nn.MSELoss()

            ae.train()
            for _ in range(epochs_ae):
                for b in loader:
                    xb = b[0].to(self.device)
                    opt_ae.zero_grad()
                    _, recon = ae(xb)
                    crit_ae(recon, xb).backward()
                    opt_ae.step()

            ae.eval()
            with torch.no_grad():
                lat_tr = ae.encode(torch.tensor(X_tr_scaled, dtype=torch.float32).to(self.device)).cpu().numpy()
                lat_val = ae.encode(torch.tensor(X_val_scaled, dtype=torch.float32).to(self.device)).cpu().numpy()

            clf = XAutoNetClassifier(input_dim=11).to(self.device)
            loader_clf = DataLoader(
                TensorDataset(torch.tensor(lat_tr, dtype=torch.float32), torch.tensor(y_tr, dtype=torch.float32).unsqueeze(1)),
                batch_size=128,
                shuffle=True,
            )
            opt_clf = optim.Adam(clf.parameters(), lr=0.001)
            crit_clf = nn.BCELoss()

            clf.train()
            for _ in range(epochs_clf):
                for xb, yb in loader_clf:
                    xb, yb = xb.to(self.device), yb.to(self.device)
                    opt_clf.zero_grad()
                    crit_clf(clf(xb), yb).backward()
                    opt_clf.step()

            clf.eval()
            with torch.no_grad():
                val_probs = clf(torch.tensor(lat_val, dtype=torch.float32).to(self.device)).cpu().numpy().squeeze()
                if np.ndim(val_probs) == 0:
                    val_probs = np.array([float(val_probs)])
                y_val_pred = (val_probs >= 0.5).astype(int)

            metrics = evaluate_classifier_metrics(y_val, y_val_pred)
            metrics["Fold"] = str(fold)
            fold_results.append(metrics)

        df = pd.DataFrame(fold_results)[["Fold", "F1 Score", "Precision", "Recall", "Accuracy"]]
        mean_row = {
            "Fold": "Mean ± SD",
            "F1 Score": f"{df['F1 Score'].mean():.2f} ± {df['F1 Score'].std():.3f}",
            "Precision": f"{df['Precision'].mean():.2f} ± {df['Precision'].std():.3f}",
            "Recall": f"{df['Recall'].mean():.2f} ± {df['Recall'].std():.3f}",
            "Accuracy": f"{df['Accuracy'].mean():.2f} ± {df['Accuracy'].std():.3f}",
        }
        df = pd.concat([df, pd.DataFrame([mean_row])], ignore_index=True)
        return df

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        epochs_ae: int = 15,
        epochs_clf: int = 15,
    ):
        """Fit end-to-end pipeline: scale, train autoencoder, extract latent, train classifier."""
        X_scaled = self.scaler.fit_transform(X)
        self.train_autoencoder(X_scaled, epochs=epochs_ae)

        self.autoencoder.eval()
        with torch.no_grad():
            tensor_scaled = torch.tensor(X_scaled, dtype=torch.float32).to(self.device)
            X_latent = self.autoencoder.encode(tensor_scaled).cpu().numpy()

        self.train_classifier(X_latent, y, epochs=epochs_clf)

        # Initialize GradCAM and DeepSHAP
        self.gradcam = GradCAM1D(self.autoencoder, FILTER_FEATURES)
        self.shap_explainer = DeepSHAPExplainer(
            self.classifier,
            torch.tensor(X_latent[:100], dtype=torch.float32).to(self.device),
            BOTTLENECK_FEATURES,
        )
        self.is_trained = True
        self.save_artifacts()

    def predict_patient(self, raw_patient_dict: Dict[str, float]) -> Dict[str, Any]:
        """Predict sepsis probability 6 hours ahead with GradCAM & SHAP explanations."""
        # Align features
        row_vals = [raw_patient_dict.get(feat, CLINICAL_NORMAL_RANGES[feat][0]) for feat in FILTER_FEATURES]
        arr = np.array(row_vals, dtype=np.float32).reshape(1, -1)
        arr_scaled = self.scaler.transform(arr)

        tensor_in = torch.tensor(arr_scaled, dtype=torch.float32).to(self.device)

        # Encode to latent
        self.autoencoder.eval()
        with torch.no_grad():
            latent = self.autoencoder.encode(tensor_in)
            latent_np = latent.cpu().numpy().squeeze()

        # Classify
        self.classifier.eval()
        with torch.no_grad():
            prob = float(self.classifier(latent).squeeze())

        # GradCAM heatmaps & DHM
        heatmaps = self.gradcam.generate_layer_heatmaps(tensor_in)
        dishm_scores, ranked_feats = compute_discounting_heatmap(heatmaps, DHM_BETA, FILTER_FEATURES)

        # SHAP waterfall explanation
        shap_res = self.shap_explainer.explain_instance(latent)

        return {
            "sepsis_probability": round(prob, 4),
            "alert_level": "HIGH" if prob >= 0.6 else ("MODERATE" if prob >= 0.30 else "LOW"),
            "is_sepsis_onset_6h": prob >= 0.30,
            "decision_threshold": 0.30,
            "latent_representation": [round(float(v), 4) for v in latent_np],
            "dhm_top_features": ranked_feats[:11],
            "gradcam_layer_heatmaps": {k: [round(float(v), 4) for v in hmap] for k, hmap in heatmaps.items()},
            "shap_waterfall": shap_res["waterfall"],
        }

    def save_artifacts(self):
        """Save trained models and configuration artifacts."""
        ae_path = os.path.join(self.artifact_dir, "autoencoder.pt")
        clf_path = os.path.join(self.artifact_dir, "classifier.pt")
        torch.save(self.autoencoder.state_dict(), ae_path)
        torch.save(self.classifier.state_dict(), clf_path)
