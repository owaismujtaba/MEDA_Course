import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

class PCAModel:
    def __init__(self, n_components=0.99):
        self.pca = PCA(n_components=n_components)
        self.scores = None
        self.loadings = None
        self.variance_ratio = None

    def fit(self, X):
        # Scale data? (Mean centering is done by PCA automatically)
        # For trade data, often we use log transform or auto-scaling (mean=0, std=1)
        # Let's use auto-scaling to give equal weight to all chapters
        X_scaled = (X - np.mean(X, axis=0)) / (np.std(X, axis=0) + 1e-9)
        self.pca.fit(X_scaled)
        self.scores = self.pca.transform(X_scaled)
        self.loadings = self.pca.components_.T
        self.variance_ratio = self.pca.explained_variance_ratio_
        return self