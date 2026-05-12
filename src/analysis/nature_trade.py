import os
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from src.data.data import load_trade_data
from src.analysis.models import PCAModel

def run_trade_nature():
    
    X, factors, chapters = load_trade_data()
    print(f"Loaded {X.shape[0]} observations across {len(chapters)} chapters.")
    
    print("Running Global PCA...")
    pca_global = PCAModel(n_components=5)
    pca_global.fit(X)
    
    plot_scores(pca_global.scores, factors, 'Year', 'trade_nature.png')
    
    
def plot_scores(scores, factors, factor_to_color, filename):
    plt.figure(figsize=(10, 6))
    dir = Path(os.getcwd(), 'figures')
    os.makedirs(dir, exist_ok=True)
    
    sns.scatterplot(
        x=scores[:, 0], 
        y=scores[:, 1], 
        hue=factors[factor_to_color],
        style=factors.get('Partner', None),
        palette='viridis',
        s=100
    )
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(Path(dir, filename))
    plt.close()