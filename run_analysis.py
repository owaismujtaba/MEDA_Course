import os
import pandas as pd
import numpy as np
from india_derisking.data_loader import load_trade_data, load_partner_diversification, load_trade_deficit
from india_derisking.meda_models import ASCA, PCAModel
from india_derisking.visualizer import plot_scores, plot_loadings, plot_china_dependence, plot_diversification, plot_trade_deficit

def main():
    print("Starting India Trade Derisking Analysis...")
    
    # 1. Load Data
    X, factors, chapters = load_trade_data()
    print(f"Loaded {X.shape[0]} observations across {len(chapters)} chapters.")
    
    # 2. Basic PCA on all data (exploring overall variance)
    print("Running Global PCA...")
    pca_global = PCAModel(n_components=5)
    pca_global.fit(X)
    
    plot_scores(
        pca_global.scores, 
        factors, 
        'Year', 
        'Global PCA Scores (Year & Partner)', 
        'output/pca_global_scores.png'
    )
    plot_loadings(
        pca_global.loadings, 
        chapters, 
        'Global PCA Loadings (Chapter Contributions)', 
        'output/pca_global_loadings.png'
    )
    
    # 3. ASCA (ANOVA Decomposition)
    print("Running ASCA Decomposition...")
    asca = ASCA()
    asca.fit(X, factors)
    
    # Analyze Year Effect
    year_scores = asca.get_scores('Year')
    year_loadings = asca.get_loadings('Year')
    plot_scores(
        year_scores, 
        factors, 
        'Year', 
        'ASCA: Year Effect Scores', 
        'output/asca_year_scores.png'
    )
    plot_loadings(
        year_loadings, 
        chapters, 
        'ASCA: Year Effect Loadings', 
        'output/asca_year_loadings.png'
    )
    
    # Analyze Partner Effect
    partner_scores = asca.get_scores('Partner')
    partner_loadings = asca.get_loadings('Partner')
    plot_scores(
        partner_scores, 
        factors, 
        'Partner', 
        'ASCA: Partner Effect Scores', 
        'output/asca_partner_scores.png'
    )
    plot_loadings(
        partner_loadings, 
        chapters, 
        'ASCA: Partner Effect Loadings', 
        'output/asca_partner_loadings.png'
    )
    
    # 4. China Dependence Heatmap
    print("Generating Dependence Heatmap...")
    plot_china_dependence(X, factors, chapters, 'output/china_dependence_heatmap.png')
    
    # 5. Partner Diversification
    print("Analyzing Partner Diversification...")
    df_div = load_partner_diversification()
    plot_diversification(df_div, 'output/partner_diversification.png')
    
    # 6. Trade Deficit
    print("Analyzing Trade Deficit...")
    df_def = load_trade_deficit()
    plot_trade_deficit(df_def, 'output/trade_deficit.png')
    
    print("Analysis complete. Results saved in output/ directory.")

if __name__ == "__main__":
    if not os.path.exists('output'):
        os.makedirs('output')
    main()
