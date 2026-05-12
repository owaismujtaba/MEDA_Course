import pandas as pd
import numpy as np
import os

def load_trade_data(filepath='data/ChapterImportWLD_CHN.csv'):
    """
    Loads chapter-level trade data and prepares it for MEDA.
    Returns:
        X (np.ndarray): Data matrix (Observations x Variables)
        factors (pd.DataFrame): Design matrix with factors (Year, Partner)
        chapters (list): List of chapter codes (Variable names)
    """
    df = pd.read_csv(filepath)
    
    # Pivot to get CHN and WLD values
    pivot = df.pivot_table(
        index=['Year', 'ProductCode'], 
        columns='PartnerISO3', 
        values='TradeValue in 1000 USD'
    ).reset_index()
    
    # Calculate Rest of World (ROW)
    pivot['ROW'] = pivot['WLD'] - pivot['CHN']
    pivot = pivot.fillna(0)
    
    # Melt to have Partner as a column
    melted = pivot.melt(
        id_vars=['Year', 'ProductCode'], 
        value_vars=['CHN', 'ROW'], 
        var_name='Partner', 
        value_name='Value'
    )
    
    # Pivot to get Chapters as columns
    final_pivot = melted.pivot_table(
        index=['Year', 'Partner'], 
        columns='ProductCode', 
        values='Value'
    ).fillna(0)
    
    # Standardize: We might want to use log or scaling
    # For now, let's keep raw values but provide metadata
    X = final_pivot.values
    factors = final_pivot.index.to_frame(index=False)
    chapters = final_pivot.columns.tolist()
    
    return X, factors, chapters

def load_partner_diversification(filepath='data/IndiaImportsAllCountries.csv', top_n=10):
    """
    Loads total imports for top N partners over time.
    """
    df = pd.read_csv(filepath)
    # Get top N partners by total trade value
    top_partners = df.groupby('PartnerISO3')['TradeValue in 1000 USD'].sum().sort_values(ascending=False).head(top_n).index.tolist()
    
    df_filtered = df[df['PartnerISO3'].isin(top_partners)]
    
    pivot = df_filtered.pivot_table(
        index='Year', 
        columns='PartnerISO3', 
        values='TradeValue in 1000 USD'
    ).fillna(0)
    
    return pivot

def load_trade_deficit(imp_file='data/IndiaImportsChinaTotal.csv', exp_file='data/IndiaExportsChinaTotal.csv'):
    """
    Loads imports and exports with China to calculate trade deficit.
    """
    imp_df = pd.read_csv(imp_file)
    exp_df = pd.read_csv(exp_file)
    
    df = pd.merge(
        imp_df[['Year', 'TradeValue in 1000 USD']], 
        exp_df[['Year', 'TradeValue in 1000 USD']], 
        on='Year', 
        suffixes=('_Imp', '_Exp')
    )
    
    df['Deficit'] = df['TradeValue in 1000 USD_Imp'] - df['TradeValue in 1000 USD_Exp']
    return df

if __name__ == "__main__":
    X, factors, chapters = load_trade_data()
    print(f"Data shape: {X.shape}")
    print(f"Factors head:\n{factors.head()}")
    print(f"Number of chapters: {len(chapters)}")
