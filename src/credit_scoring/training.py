"""Training orchestration.

Loads and validates the data, compares candidate models with cross-validation,
calibrates the winner, picks the cost-minimising decision threshold, evaluates on
a held-out test set, builds the drift reference profile, and registers the model
with full metadata.
"""

from __future__ import annotations

from datetime import datetime, timezone

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from sklearn.calibration import CalibratedClassifierCV, calibration_curve  # noqa: E402
from sklearn.model_selection import StratifiedKFold, cross_validate  # noqa: E402

from .config import get_settings  # noqa: E402
from .data import load_dataset, train_test_frames, validate_frame  # noqa: E402
from .domain import FEATURES, TARGET  # noqa: E402
from .features import candidate_pipelines  # noqa: E402
from .logging_setup import get_logger  # noqa: E402
from .models import best_cost_threshold, get_registry, operating_point, ranking_metrics  # noqa: E402
from .models.threshold import total_cost  # noqa: E402
from .monitoring import build_reference, save_reference  # noqa: E402

log = get_logger(__name__)


def _cross_validate(pipelines, X, y) -> dict:
    settings = get_settings()
    cv = StratifiedKFold(n_splits=settings.cv_folds, shuffle=True, random_state=settings.random_state)
    results = {}
    for name, pipe in pipelines.items():
        scores = cross_validate(pipe, X, y, cv=cv, scoring=["roc_auc", "average_precision"], n_jobs=-1)
        results[name] = {
            "cv_roc_auc": round(float(scores["test_roc_auc"].mean()), 4),
            "cv_pr_auc": round(float(scores["test_average_precision"].mean()), 4),
        }
    return results
