"""Unit tests for 1D GradCAM, Discounting HeatMap (DHM), and DeepSHAP."""

import pytest
import torch
import numpy as np
from xautonet.config import FILTER_FEATURES, BOTTLENECK_FEATURES, DHM_BETA
from xautonet.models.autoencoder import Conv1DAutoencoder
from xautonet.models.classifier import XAutoNetClassifier
from xautonet.xai.gradcam import GradCAM1D, compute_discounting_heatmap
from xautonet.xai.shap_explainer import DeepSHAPExplainer


def test_gradcam_heatmaps_and_dhm():
    ae = Conv1DAutoencoder(input_dim=19, latent_dim=11)
    gradcam = GradCAM1D(ae, FILTER_FEATURES)

    x = torch.randn(1, 19)
    heatmaps = gradcam.generate_layer_heatmaps(x)

    for layer in ["E1", "E2", "E3", "E4"]:
        assert layer in heatmaps
        assert len(heatmaps[layer]) == 19
        assert np.all(heatmaps[layer] >= 0.0)

    # Compute DHM
    dishm_scores, ranked = compute_discounting_heatmap(heatmaps, DHM_BETA, FILTER_FEATURES)
    assert len(dishm_scores) == 19
    assert len(ranked) == 19
    # Ranked should be in descending order of scores
    for i in range(len(ranked) - 1):
        assert ranked[i][1] >= ranked[i+1][1]


def test_shap_explainer_instance_waterfall():
    clf = XAutoNetClassifier(input_dim=11)
    background = torch.randn(20, 11)

    explainer = DeepSHAPExplainer(clf, background, BOTTLENECK_FEATURES)
    patient_latent = torch.randn(1, 11)

    explanation = explainer.explain_instance(patient_latent)
    assert "prediction_probability" in explanation
    assert "waterfall" in explanation
    assert len(explanation["waterfall"]) == 11
    
    first_item = explanation["waterfall"][0]
    assert "feature" in first_item
    assert "shap_value" in first_item
    assert "impact" in first_item
    assert first_item["impact"] in ["contributing", "offsetting"]
