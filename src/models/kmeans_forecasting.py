"""K-Means cluster-average estimator compatible with the tabular forecast API.

K-Means itself assigns a cluster, not a numeric target. This estimator maps each
training cluster to its average demand and returns that average as the MW forecast.
"""
from __future__ import annotations

import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


class KMeansDemandForecaster:
    """Predict demand using the mean target of the nearest learned cluster."""

    def __init__(self, n_clusters: int = 4, random_state: int = 42, n_init: int = 10,
                 max_iter: int = 100):
        self.n_clusters = int(n_clusters)
        self.random_state = int(random_state)
        self.n_init = int(n_init)
        self.max_iter = int(max_iter)
        self.scaler = StandardScaler()
        self.clusterer = KMeans(
            n_clusters=self.n_clusters,
            random_state=self.random_state,
            n_init=self.n_init,
            max_iter=self.max_iter,
        )
        self.cluster_means_: np.ndarray | None = None
        self.cluster_sizes_: np.ndarray | None = None

    def fit(self, features, target):
        x = np.asarray(features, dtype=float)
        y = np.asarray(target, dtype=float).reshape(-1)
        if x.ndim != 2 or len(x) != len(y):
            raise ValueError("K-Means requires a 2D feature matrix and one target per row.")
        if len(x) < self.n_clusters:
            raise ValueError(
                f"K-Means needs at least {self.n_clusters} training rows; received {len(x)}."
            )
        if not np.isfinite(x).all() or not np.isfinite(y).all():
            raise ValueError("K-Means training values must be finite.")
        scaled = self.scaler.fit_transform(x)
        assignments = self.clusterer.fit_predict(scaled)
        self.cluster_sizes_ = np.asarray([
            int(np.count_nonzero(assignments == cluster))
            for cluster in range(self.n_clusters)
        ])
        global_mean = float(y.mean())
        self.cluster_means_ = np.asarray([
            float(y[assignments == cluster].mean()) if np.any(assignments == cluster) else global_mean
            for cluster in range(self.n_clusters)
        ])
        return self

    def predict_cluster(self, features) -> np.ndarray:
        if self.cluster_means_ is None:
            raise RuntimeError("K-Means forecaster has not been fitted.")
        x = np.asarray(features, dtype=float)
        if x.ndim != 2 or not np.isfinite(x).all():
            raise ValueError("K-Means prediction features must be a finite 2D matrix.")
        return self.clusterer.predict(self.scaler.transform(x)).astype(int)

    def predict(self, features) -> np.ndarray:
        return self.cluster_means_[self.predict_cluster(features)]
