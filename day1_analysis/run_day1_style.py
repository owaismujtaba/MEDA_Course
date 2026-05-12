import os
import pandas as pd
import numpy as np
from india_derisking.data_loader import load_trade_data
from india_derisking.meda_models import PCAModel, OMEDA
from india_derisking.visualizer import plot_scores, plot_loadings, plot_omeda_contributions

def main():
    print("Starting Day 1 Style In-Depth Analysis...")
    
    # 1. Load Data
    X, factors, chapters = load_trade_data()
    
    # 2. PCA: Global View (Day 1 Style)
    # Scaling is crucial for PCA with chapters of different magnitudes
    pca = PCAModel(n_components=5)
    pca.fit(X)
    
    # Plot Scores
    plot_scores(
        pca.scores, 
        factors, 
        'Partner', 
        'PCA Scores: China vs Rest of World (Day 1 View)', 
        'output_day1/pca_scores_partners.png'
    )
    
    # Plot Loadings
    plot_loadings(
        pca.loadings, 
        chapters, 
        'PCA Loadings: Variables Driving Trade Variation', 
        'output_day1/pca_loadings.png'
    )
    
    # 3. oMEDA: What makes China unique?
    print("Running oMEDA: China vs Global Mean...")
    omeda_chn = OMEDA()
    chn_mask = (factors['Partner'] == 'CHN').values
    omeda_chn.fit(X, chn_mask)
    
    plot_omeda_contributions(
        omeda_chn.contributions, 
        chapters, 
        'oMEDA: Chapters distinguishing China from ROW', 
        'output_day1/omeda_china_vs_all.png'
    )
    
    # 4. oMEDA: What changed in 2024 vs 2015?
    print("Running oMEDA: 2024 vs Global Mean...")
    omeda_2024 = OMEDA()
    y2024_mask = (factors['Year'] == 2024).values
    omeda_2024.fit(X, y2024_mask)
    
    plot_omeda_contributions(
        omeda_2024.contributions, 
        chapters, 
        'oMEDA: Chapters distinguishing 2024 from History', 
        'output_day1/omeda_2024_vs_all.png'
    )
    
    print("Day 1 style analysis complete. Results in output_day1/")

if __name__ == "__main__":
    if not os.path.exists('output_day1'):
        os.makedirs('output_day1')
    main()
