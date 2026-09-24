"""Baseline comparative models matching Table II from the IEEE IRI 2023 paper.

In Table II, the proposed XAutoNet is compared against 9 traditional ML algorithms:
KNN, Gradient Boost, Random Forest, Naïve Bayes, XG Boost, Decision Tree, SVM,
Logistic Regression, and ADA Boost.
"""

from typing import Dict, Any
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import (
    GradientBoostingClassifier,
    RandomForestClassifier,
    AdaBoostClassifier,
    HistGradientBoostingClassifier,
)
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from xautonet.models.classifier import evaluate_classifier_metrics


def get_baseline_models() -> Dict[str, Any]:
    """Instantiate the 9 baseline ML algorithms evaluated in Table II."""
    return {
        "KNN": KNeighborsClassifier(n_neighbors=5),
        "Gradient Boost": GradientBoostingClassifier(n_estimators=100, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "Naïve Bayes": GaussianNB(),
        "XG Boost": HistGradientBoostingClassifier(random_state=42),
        "Decision Tree": DecisionTreeClassifier(max_depth=6, random_state=42),
        "SVM": SVC(kernel="rbf", probability=True, random_state=42),
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "ADA Boost": AdaBoostClassifier(n_estimators=50, random_state=42),
    }


def benchmark_baseline_models(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> pd.DataFrame:
    """Train and evaluate all baseline ML models to produce the comparative evaluation in Table II."""
    models = get_baseline_models()
    results = []

    for name, model in models.items():
        try:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            metrics = evaluate_classifier_metrics(y_test, y_pred)
            metrics["Model"] = name
            results.append(metrics)
        except Exception as e:
            results.append({
                "Model": name,
                "Accuracy": 0.0,
                "F1 Score": 0.0,
                "Precision": 0.0,
                "Recall": 0.0,
                "Error": str(e),
            })

    df = pd.DataFrame(results)[["Model", "Accuracy", "F1 Score", "Precision", "Recall"]]
    return df
