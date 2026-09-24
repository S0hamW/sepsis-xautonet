"""Explainability package for XAutoNet: 1D GradCAM and DeepSHAP modules."""

from xautonet.xai.gradcam import GradCAM1D, compute_discounting_heatmap
from xautonet.xai.shap_explainer import DeepSHAPExplainer

__all__ = ["GradCAM1D", "compute_discounting_heatmap", "DeepSHAPExplainer"]
