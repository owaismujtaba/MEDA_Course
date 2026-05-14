import os
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from src.data.data import load_trade_data
from src.analysis.models import PCAModel
import pdb

def run_trade_nature():
    
    X, factors, chapters = load_trade_data()
    print(f"Loaded {X.shape[0]} observations across {len(chapters)} chapters.")
    
    print("Running Global PCA...")
    pca_global = PCAModel(n_components=5)
    pca_global.fit(X)
    
    plot_scores(pca_global.scores, factors, 'Year', 'pca_global_scores.png')
    plot_loadings(
        pca_global.loadings, 
        chapters, 
        'pca_global_loadings.png'
    )



def plot_scores(scores, factors, factor_to_color, filename, pca_model=None):
    # Setup directory
    dir = Path(os.getcwd(), 'figures')
    os.makedirs(dir, exist_ok=True)
    
    # 12x8 matches the exact aspect ratio of the target image
    plt.figure(figsize=(12, 8))
    
    # Merge for Seaborn plotting
    plot_df = factors.copy()
    plot_df['PC1'] = scores[:, 0]
    plot_df['PC2'] = scores[:, 1]
    
    # 1. Scatter Plot
    sns.scatterplot(
        data=plot_df,
        x='PC1', 
        y='PC2', 
        hue=factor_to_color,
        style=factor_to_color,
        size='Year', 
        sizes=(40, 250), 
        alpha=0.8,
        palette=['#e74c3c', '#3498db']
    )
    
    # 2. Trajectory Lines and Year Annotations
    for factor_val in plot_df[factor_to_color].unique():
        factor_data = plot_df[plot_df[factor_to_color] == factor_val].sort_values('Year')
        
        plt.plot(
            factor_data['PC1'], 
            factor_data['PC2'], 
            alpha=0.4, 
            linewidth=2, 
            zorder=1
        )
        
        start_point = factor_data.iloc[0]
        end_point = factor_data.iloc[-1]
        plt.text(start_point['PC1'], start_point['PC2'] + 0.1, str(start_point['Year']), fontsize=9, fontweight='bold')
        plt.text(end_point['PC1'], end_point['PC2'] + 0.1, str(end_point['Year']), fontsize=9, fontweight='bold')

    # 3. Formatting to match the target image
    plt.title('PCA Score Plot: Evolution of Trade Structures', fontsize=14)
    
    # Dynamically inject the variance explained into the labels if the PCA model is passed
    if pca_model is not None:
        plt.xlabel(f"PC1 ({pca_model.explained_variance_ratio_[0]*100:.1f}% Variance Explained)", fontsize=12)
        plt.ylabel(f"PC2 ({pca_model.explained_variance_ratio_[1]*100:.1f}% Variance Explained)", fontsize=12)
    else:
        plt.xlabel('PC1', fontsize=12)
        plt.ylabel('PC2', fontsize=12)
        
    # Remove top and right axis lines
    sns.despine()
    
    # Legend and Grid
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    
    # Save
    plt.savefig(Path(dir, filename))
    plt.close()
def plot_loadings(loadings, chapter_names, filename, top_n=10):
    # Select top contributors to PC1 and PC2
    dir = Path(os.getcwd(), 'figures')
    os.makedirs(dir, exist_ok=True)
    pc1 = loadings[:, 0]
    pc2 = loadings[:, 1]
    
    # Get indices of top N
    idx1 = np.argsort(np.abs(pc1))[-top_n:]
    idx2 = np.argsort(np.abs(pc2))[-top_n:]
    combined_idx = list(set(idx1) | set(idx2))
    
    plt.figure(figsize=(12, 8))
    for i in combined_idx:
        plt.arrow(0, 0, pc1[i], pc2[i], color='r', alpha=0.5)
        plt.text(pc1[i], pc2[i], str(chapter_names[i]), fontsize=9)
    
    plt.axhline(0, color='black', lw=1)
    plt.axvline(0, color='black', lw=1)
    plt.xlabel('PC1 Loadings')
    plt.ylabel('PC2 Loadings')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(Path(dir, filename))
    plt.savefig(filename)
    plt.close()
