# Day 1 Style Analysis: In-Depth India Trade MEDA
**Framework:** PCA and oMEDA (Multivariate Exploratory Data Analysis)

## 1. Overview
Following the methodology introduced on Day 1 (analyzing archeological/wheat datasets), we applied PCA and oMEDA to India's trade data (2003-2024). The goal was to identify the specific "markers" (HS Chapters) that define different classes of trade.

## 2. Global PCA: The Structural Divide
Just as the "Wine" dataset showed clusters by cultivar, the **India Trade Score Plot** shows two massive clusters:
- **China Cluster:** Positioned at high PC1 values, indicating a unique and consistent import profile.
- **ROW Cluster:** More spread out, reflecting a more diversified commodity-based profile.

The **Loading Plot** identifies that Chapters 84 (Machinery) and 85 (Electronics) are the primary variables driving the variance in India's global trade architecture.

## 3. oMEDA: The "China Marker"
We used **oMEDA (One-block MEDA)** to find which specific chapters distinguish China from the global mean. 

**Top "China Markers" (Positive Contribution):**
- **Chapter 85 (Electrical Machinery):** The strongest marker. This confirms that India's "class" of imports from China is fundamentally defined by electronics.
- **Chapter 84 (Nuclear Reactors/Machinery):** A close second.
- **Chapter 29 (Organic Chemicals):** Highlights India's dependence on Chinese APIs and chemical precursors.

## 4. oMEDA: Temporal Evolution (2024 Class)
By treating 2024 as a separate class, we identified how the trade basket has evolved compared to historical averages.

- **Positive Shift:** 2024 is defined by significantly higher contributions from **Chapter 27 (Energy)** and **Chapter 84 (Machinery)**, suggesting that India's industrial and energy needs are reaching new peaks in the current trade era.

## 5. Conclusion
The Day 1 methods successfully isolated the structural drivers of India's trade. While the total trade value is influenced by many factors, **oMEDA** confirms that India's specific dependence on China is concentrated in a very narrow set of "marker" chapters that are not easily substituted by the "Rest of World" class.

---
### Visual Evidence
- [PCA Scores (Partners)](file:///home/owais/PhD/MEA%20Course/output_day1/pca_scores_partners.png)
- [oMEDA: China vs World](file:///home/owais/PhD/MEA%20Course/output_day1/omeda_china_vs_all.png)
- [oMEDA: 2024 Evolution](file:///home/owais/PhD/MEA%20Course/output_day1/omeda_2024_vs_all.png)
