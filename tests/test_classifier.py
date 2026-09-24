"""Unit tests for the XAutoNet deep classifier and evaluation metrics."""

import pytest
import torch
import numpy as np
from xautonet.models.classifier import XAutoNetClassifier, evaluate_classifier_metrics


def test_classifier_output_shape_and_range():
    clf = XAutoNetClassifier(input_dim=11)
    x = torch.randn(8, 11)

    probs = clf(x)
    assert probs.shape == (8, 1)
    
    probs_np = probs.detach().numpy()
    assert np.all(probs_np >= 0.0) and np.all(probs_np <= 1.0)

    preds = clf.predict(x.numpy())
    assert preds.shape == (8,)
    assert set(preds).issubset({0, 1})


def test_evaluation_metrics_calculation():
    y_true = np.array([0, 0, 1, 1, 1])
    y_pred = np.array([0, 0, 1, 1, 0])  # 4/5 accurate, 2/2 precision, 2/3 recall

    metrics = evaluate_classifier_metrics(y_true, y_pred)
    assert metrics["Accuracy"] == 0.8
    assert metrics["Precision"] == 1.0
    assert round(metrics["Recall"], 2) == 0.67
    assert "F1 Score" in metrics
