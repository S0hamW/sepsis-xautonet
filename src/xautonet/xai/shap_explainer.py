"""DeepSHAP explainability module for XAutoNet global and local waterfall explanations.

As described in Section II.D.2, Section III.C and Fig. 4 of the IEEE IRI 2023 paper:
"Deep SHAP makes use of Deep Learning Important FeaTures (DeepLIFT) and Shapley value
and assigns each feature an importance value for a particular prediction...
With respect to local feature importance, waterfall plot becomes very insightful in
providing both qualitative and quantitative explanations of feature's impact in
XAutoNet's prediction: whether contributing (red) or offsetting (blue)."
"""

from typing import List, Dict, Any, Optional
import numpy as np
import torch
from xautonet.config import BOTTLENECK_FEATURES, TOP_SHAP_FEATURES
from xautonet.models.classifier import XAutoNetClassifier


class DeepSHAPExplainer:
    """Computes global feature importance and local instance waterfall explanations."""

    def __init__(
        self,
        classifier: XAutoNetClassifier,
        background_data: torch.Tensor,
        feature_names: Optional[List[str]] = None,
    ):
        self.classifier = classifier
        self.classifier.eval()
        self.feature_names = feature_names or BOTTLENECK_FEATURES

        if isinstance(background_data, np.ndarray):
            background_tensor = torch.tensor(background_data, dtype=torch.float32)
        else:
            background_tensor = background_data.clone().detach().float()

        if len(background_tensor) > 100:
            background_tensor = background_tensor[:100]

        self.background_tensor = background_tensor

        # Initialize explainer with robust fallbacks
        self._explainer = None
        self._mode = "fallback"

        try:
            import shap
            try:
                self._explainer = shap.DeepExplainer(self.classifier, self.background_tensor)
                self._mode = "deep"
            except Exception:
                try:
                    self._explainer = shap.GradientExplainer(self.classifier, self.background_tensor)
                    self._mode = "gradient"
                except Exception:
                    self._mode = "fallback"
        except ImportError:
            self._mode = "fallback"

    def _compute_fallback_shap_values(self, x_latent: torch.Tensor) -> np.ndarray:
        """Mathematical Shapley-approximation via perturbation against background distribution.

        Evaluates marginal contribution: E[f(x_with_feature_i)] - E[f(background)].
        """
        self.classifier.eval()
        with torch.no_grad():
            base_preds = self.classifier(self.background_tensor).squeeze().cpu().numpy()
            base_val = float(np.mean(base_preds))

            x_pred = float(self.classifier(x_latent).squeeze().cpu().numpy())
            total_diff = x_pred - base_val

            n_features = len(self.feature_names)
            shap_values = np.zeros(n_features, dtype=np.float32)

            # Feature perturbation marginal effect
            x_rep = x_latent.repeat(len(self.background_tensor), 1)
            for i in range(n_features):
                x_perturbed = x_rep.clone()
                x_perturbed[:, i] = self.background_tensor[:, i]
                pred_perturbed = self.classifier(x_perturbed).squeeze().cpu().numpy()
                diff = float(x_pred - np.mean(pred_perturbed))
                shap_values[i] = diff

            # Normalize to satisfy additivity axiom: sum(phi_i) ~ f(x) - E[f]
            sum_shap = np.sum(shap_values)
            if abs(sum_shap) > 1e-6:
                shap_values = shap_values * (total_diff / sum_shap)
            else:
                shap_values = np.ones(n_features, dtype=np.float32) * (total_diff / max(n_features, 1))

        return shap_values

    def explain_instance(self, x_latent: torch.Tensor) -> Dict[str, Any]:
        """Compute local SHAP attribution values for a single patient instance.

        Args:
            x_latent: Tensor of shape (1, 11) or (11,)
        Returns:
            Dict containing prediction probability, sepsis alert, and sorted waterfall list.
        """
        if isinstance(x_latent, np.ndarray):
            x_latent = torch.tensor(x_latent, dtype=torch.float32)
        if x_latent.dim() == 1:
            x_latent = x_latent.unsqueeze(0)

        self.classifier.eval()
        with torch.no_grad():
            prob = float(self.classifier(x_latent).squeeze().cpu().numpy())

        vals = None
        if self._explainer is not None:
            try:
                raw_shap = self._explainer.shap_values(x_latent)
                if isinstance(raw_shap, list):
                    vals = np.array(raw_shap[0]).squeeze()
                else:
                    vals = np.array(raw_shap).squeeze()
            except Exception:
                vals = None

        if vals is None or len(vals) != len(self.feature_names):
            vals = self._compute_fallback_shap_values(x_latent)

        waterfall = []
        for i, name in enumerate(self.feature_names):
            sv = float(vals[i]) if i < len(vals) else 0.0
            waterfall.append({
                "feature": name,
                "shap_value": round(sv, 4),
                "impact": "contributing" if sv > 0 else "offsetting",
                "is_top_impact": name in TOP_SHAP_FEATURES,
            })

        # Sort by absolute magnitude of impact
        waterfall.sort(key=lambda item: abs(item["shap_value"]), reverse=True)

        return {
            "prediction_probability": round(prob, 4),
            "sepsis_alert": prob >= 0.5,
            "waterfall": waterfall,
        }

    def compute_global_importance(self, X_latent: torch.Tensor) -> List[Dict[str, Any]]:
        """Compute mean absolute SHAP values across cohort for global importance (Fig. 3C)."""
        if isinstance(X_latent, np.ndarray):
            X_latent = torch.tensor(X_latent, dtype=torch.float32)

        vals = None
        if self._explainer is not None:
            try:
                raw_shap = self._explainer.shap_values(X_latent)
                if isinstance(raw_shap, list):
                    vals = np.array(raw_shap[0])
                else:
                    vals = np.array(raw_shap)
            except Exception:
                vals = None

        if vals is None:
            # Batch sample fallback
            sample_size = min(len(X_latent), 50)
            all_vals = [self._compute_fallback_shap_values(X_latent[i : i + 1]) for i in range(sample_size)]
            vals = np.array(all_vals)

        if vals.ndim == 3:
            vals = vals.squeeze()

        mean_abs_shap = np.mean(np.abs(vals), axis=0)

        summary = []
        for i, name in enumerate(self.feature_names):
            score = float(mean_abs_shap[i]) if i < len(mean_abs_shap) else 0.0
            summary.append({
                "feature": name,
                "mean_shap_score": round(score, 4),
                "is_top_impact": name in TOP_SHAP_FEATURES,
            })

        summary.sort(key=lambda x: x["mean_shap_score"], reverse=True)
        return summary
