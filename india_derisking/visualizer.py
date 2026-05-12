import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

def plot_scores(scores, factors, factor_to_color, title, filename):
    plt.figure(figsize=(10, 6))
    sns.scatterplot(
        x=scores[:, 0], 
        y=scores[:, 1], 
        hue=factors[factor_to_color],
        style=factors.get('Partner', None),
        palette='viridis',
        s=100
    )
    plt.title(title)
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.grid(True, alpha=0.3)
    plt.savefig(filename)
    plt.close()

def plot_loadings(loadings, chapter_names, title, filename, top_n=10):
    # Select top contributors to PC1 and PC2
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
    plt.title(title)
    plt.xlabel('PC1 Loadings')
    plt.ylabel('PC2 Loadings')
    plt.grid(True, alpha=0.3)
    plt.savefig(filename)
    plt.close()

def plot_china_dependence(X, factors, chapters, filename):
    # Calculate China share per year and chapter
    # X has rows for (Year, Partner)
    years = factors['Year'].unique()
    chn_mask = (factors['Partner'] == 'CHN').values
    row_mask = (factors['Partner'] == 'ROW').values
    
    X_chn = X[chn_mask, :]
    X_row = X[row_mask, :]
    
    # Avoid division by zero
    total = X_chn + X_row + 1e-9
    share = X_chn / total
    
    # Plot heatmap for top 20 chapters by value
    total_val = np.sum(total, axis=0)
    top_idx = np.argsort(total_val)[-20:]
    
    plt.figure(figsize=(14, 10))
    sns.heatmap(
        share[:, top_idx].T, 
        xticklabels=years, 
        yticklabels=[chapters[i] for i in top_idx],
        cmap='YlOrRd',
        annot=True,
        fmt=".2f"
    )
    plt.title('India\'s Import Dependence on China (Share by Chapter)')
    plt.xlabel('Year')
    plt.ylabel('HS Chapter')
    plt.savefig(filename)
    plt.close()

def plot_diversification(df_pivot, filename):
    plt.figure(figsize=(12, 7))
    # Normalize to 100% for share analysis
    df_share = df_pivot.div(df_pivot.sum(axis=1), axis=0) * 100
    df_share.plot(kind='area', stacked=True, ax=plt.gca(), cmap='tab20')
    plt.title('Import Share of Top 10 Partners Over Time')
    plt.xlabel('Year')
    plt.ylabel('Share of Top 10 Imports (%)')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

def plot_trade_deficit(df_deficit, filename):
    plt.figure(figsize=(10, 6))
    plt.bar(df_deficit['Year'], df_deficit['Deficit'] / 1e6, color='salmon', alpha=0.7, label='Trade Deficit')
    plt.plot(df_deficit['Year'], df_deficit['TradeValue in 1000 USD_Imp'] / 1e6, marker='o', label='Imports')
    plt.plot(df_deficit['Year'], df_deficit['TradeValue in 1000 USD_Exp'] / 1e6, marker='s', label='Exports')
    plt.title('India-China Trade Balance (Billion USD)')
    plt.xlabel('Year')
    plt.ylabel('Value (Billion USD)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(filename)
    plt.close()

def plot_omeda_contributions(contributions, chapters, title, filename, top_n=15):
    idx = np.argsort(np.abs(contributions))[-top_n:]
    
    plt.figure(figsize=(10, 8))
    colors = ['red' if c > 0 else 'blue' for c in contributions[idx]]
    plt.barh([str(chapters[i]) for i in idx], contributions[idx], color=colors)
    plt.title(title)
    plt.xlabel('Standardized Difference (Class Mean - Total Mean)')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
