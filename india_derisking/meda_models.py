import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

class ASCA:
    def __init__(self):
        self.grand_mean = None
        self.effects = {}
        self.pca_models = {}
        self.residuals = None

    def fit(self, X, factors):
        """
        Perform ANOVA decomposition and fit PCA on each effect.
        factors: pd.DataFrame with columns matching factor names.
        """
        self.grand_mean = np.mean(X, axis=0)
        X_centered = X - self.grand_mean
        
        # 1. Main Effects
        for factor in factors.columns:
            # Calculate means for each level of the factor
            unique_levels = factors[factor].unique()
            effect_matrix = np.zeros_like(X)
            
            for level in unique_levels:
                mask = (factors[factor] == level).values
                level_mean = np.mean(X_centered[mask, :], axis=0)
                effect_matrix[mask, :] = level_mean
            
            self.effects[factor] = effect_matrix
            X_centered -= effect_matrix # Subtract effect from centered data for next step (or interaction)
            
        # 2. Interaction (Simple implementation for 2 factors)
        if len(factors.columns) == 2:
            # What's left in X_centered now is interaction + residuals
            # In a basic ASCA, we can further decompose or just treat it as interaction if we have replicates.
            # Since we have Year x Partner, we have exactly one observation per cell (no replicates).
            # This means Interaction and Residuals are confounded.
            # Usually, we treat the remaining part as interaction if we assume no residuals, 
            # or residuals if we assume no interaction.
            # In Trade data, Interaction (Year x Partner) is very important (e.g. shift in partner share over time).
            self.effects['interaction'] = X_centered
            self.residuals = np.zeros_like(X) # No replicates
        else:
            self.residuals = X_centered

        # 3. Fit PCA on each effect
        for effect_name, effect_matrix in self.effects.items():
            pca = PCA(n_components=min(effect_matrix.shape[0], effect_matrix.shape[1], 5))
            pca.fit(effect_matrix)
            self.pca_models[effect_name] = pca

    def get_scores(self, effect_name):
        effect_matrix = self.effects[effect_name]
        return self.pca_models[effect_name].transform(effect_matrix)

    def get_loadings(self, effect_name):
        return self.pca_models[effect_name].components_.T

class PCAModel:
    def __init__(self, n_components=5):
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

class OMEDA:
    def __init__(self):
        self.contributions = None
        self.class_mean = None
        self.total_mean = None

    def fit(self, X, mask):
        """
        Calculate contributions of variables to a specific class.
        mask: Boolean array for the class of interest.
        """
        self.total_mean = np.mean(X, axis=0)
        self.class_mean = np.mean(X[mask, :], axis=0)
        
        # Difference in means
        diff = self.class_mean - self.total_mean
        
        # In a simple oMEDA, we look at the difference scaled by standard deviation
        # to see which variables are most "abnormal" for this class.
        std = np.std(X, axis=0) + 1e-9
        self.contributions = diff / std
        
        return self
