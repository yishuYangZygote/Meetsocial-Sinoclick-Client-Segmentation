# Client Segmentation for SinoClick (Meetsocial) — Unsupervised Learning on 10,000 B2B Advertisers

> End-to-end data science project: from raw SQL extraction on Alibaba Cloud MaxCompute, through feature engineering on 76 mixed-type fields, PCA dimensionality reduction, a six-model clustering competition, to final business-ready client profiles delivered to operations teams.

## Business Context

**Meetsocial / SinoClick** is a leading cross-border digital advertising SaaS platform serving thousands of B2B clients who run ad campaigns on Facebook, Google, and TikTok across global markets. The Customer Success and Operations teams needed a data-driven segmentation of the entire client base to:

- **Identify high-value clients** and match them with dedicated CSM (Customer Success Manager) one-on-one service
- **Discover high-potential growth clients** that could be incubated through targeted engagement
- **Flag churn-risk segments** and low-activity groups for reactivation campaigns
- **Enable precision operations** by assigning differentiated service strategies to each segment

Previously, client service strategies were applied uniformly. This project replaced intuition-based targeting with a statistically validated, multi-dimensional segmentation framework.

## Technical Approach

### Data Pipeline

| Stage | Details |
| --- | --- |
| **Source** | 8 production tables on Alibaba Cloud MaxCompute (DataWorks), covering client master data, ad account activity, spend by media channel, campaign performance, and web behavioral events |
| **Extraction** | Custom SQL (~1,500 lines) joining across tables A through H; built a temporary wide table of 9,999 active client records with 76 fields |
| **Scope** | All clients with historical ad spend; partitioned data spanning 2022-12 through 2025-07 |

### Feature Engineering

Constructed two purpose-built feature sets from the unified data dictionary:

**Feature Set A** (for model training) — mathematically precise:

- `log1p` transformation on 36 right-skewed spending/activity metrics to normalize distributions
- StandardScaler normalization across all numerical features
- Ordinal encoding for ranked variables (e.g., `onboarding_intent_tier` with 6 levels)
- Rank-based percentile binning for `days_since_last_spend` (custom quartile boundaries)
- 8 binary business-logic flags (4 core product categories + 4 high-risk indicators)
- Ratio features preserved in [0,1] range (platform spend ratios, value concentration indices)

**Feature Set B** (for business interpretation) — human-readable:

- Descriptive fields: top product categories, media channels, geographic regions, company attributes
- Retained for cluster profiling and decision tree validation post-clustering

### Dimensionality Reduction

- **Multicollinearity diagnosis**: identified 9 clusters of highly correlated features (threshold > 0.8)
- **PCA**: reduced ~50 transformed features to 19 principal components retaining 90%+ cumulative explained variance
- Evaluated both manual feature pruning and direct PCA; found PCA alone yielded stronger downstream model performance

### Model Competition ("Model Gauntlet")

Six clustering algorithms competed on the PCA-reduced feature space, evaluated across K = 2 through 6:

| Model | Role | Key Characteristic |
|---|---|---|
| **K-Means** | Performance baseline | Spherical clusters, fast, interpretable |
| **GMM** | Probabilistic insight | Soft clustering with membership probabilities — identifies "incubatable" boundary clients |
| **Agglomerative** | Structural insight | Dendrogram reveals hierarchical client relationships |
| **Birch** | Efficiency validation | Scalable alternative to K-Means; cross-validates stability |
| **DBSCAN** | Outlier discovery | Density-based; identifies anomalous clients outside any cluster |
| **SOM** | Topological visualization | Neural-network-based 2D mapping of client ecosystem |

**Evaluation metrics**: Silhouette Score (primary), BIC (for GMM), ANOVA F-test on cluster separation, plus qualitative business interpretability assessment with the operations team.

**Result**: K-Means consistently achieved the highest silhouette scores across all K values. DBSCAN was ruled out (one cluster absorbed 94% of clients, lacking business utility). Final configurations selected: **K=3** and **K=4** using K-Means.

### Cluster Profiling & Validation

For each winning configuration (K=3 and K=4), produced comprehensive profiles:

**Quantitative profiling**:

- 95% confidence intervals and means for 10 core KPIs (monthly average spend, active days, ad counts, first-7-day spend, category diversity, etc.)
- ANOVA F-tests confirming statistically significant inter-cluster differences (p < 0.05) across all key metrics
- Box plot and violin plot visualizations for distribution comparison

**Qualitative profiling**:

- Top product categories, media channel preferences, geographic targeting patterns per cluster
- Company size distribution (KA / SMB / PA), business background analysis
- Composite risk flag penetration rates

**Interpretability validation**:

- Shallow decision tree (max_depth=3) trained on cluster labels using Feature Set B
- Extracted human-readable business rules (e.g., "If monthly_avg_spend > $X AND main_media = Facebook, then Cluster 1 with 95% probability")
- Rules cross-validated against manual profiling analysis

## Deliverables

Two parallel segmentation analyses, each delivering complete results:

### Segmentation A: All Clients (KA + PA + SMB)

Full client base including Key Accounts, yielding segmentation across the complete value spectrum.

### Segmentation B: SMB Only

Focused on small-and-medium business clients, providing granular operational segments for the SMB service team.

Each analysis includes:

- Labeled client lists (every `corporation_id` mapped to its cluster)
- Raw data, transformed data, and descriptive feature tables per K-value
- Quantitative profile spreadsheets (confidence intervals, means, risk rates)
- Descriptive profile spreadsheets (top categories, channels, regions per cluster)
- The full Python pipeline (Jupyter notebook + standalone `.py` script)

## Repository Structure

```text
├── README.md
├── Customer_Segmentation_Research_Study_Analysis.pdf      # Research literature review (Chinese)
├── Customer_Segmentation_Research_Study_Analysis_English_Translation.txt
├── Core_Data_Processing_and_Scientific_Computing.txt      # Library dependency reference
│
├── Final Result/
│   ├── Handover_with_KA_PA/                               # Full client base segmentation
│   │   ├── Client_Segmentation_with_KA_PA.ipynb           # Complete Jupyter notebook
│   │   ├── Client_Segmentation_with_KA_PA.py              # Standalone Python script
│   │   ├── SQL_Phase1_Label_Extraction_*.txt               # SQL extraction queries (~1,500 lines)
│   │   ├── SQL_Final_Data_Extraction_Analysis_Dataset.csv  # Source dataset (9,999 records)
│   │   ├── Unified_Data_Dictionary_v1.0.xlsx               # Field definitions & metadata
│   │   ├── Unified_Data_Dictionary_v2.0_*.csv              # Feature construction classifications
│   │   ├── feature_set_A_output.csv                        # Engineered training features
│   │   ├── df_k3_kmeans_*.csv                              # K=3 results (raw, transformed, descriptive)
│   │   ├── df_k4_kmeans_*.csv                              # K=4 results (raw, transformed, descriptive)
│   │   ├── df_k3_dbscan_*.csv                              # DBSCAN experimental results
│   │   ├── k3_with_KA_PA_Profile/                          # K=3 cluster profile spreadsheet
│   │   ├── k4_with_KA_PA_Profile/                          # K=4 cluster profile spreadsheet
│   │   └── output1.xlsx, output2.xlsx                      # Numeric & descriptive profile tables
│   │
│   └── Handover_with_only_SMB/                             # SMB-only segmentation
│       ├── Client_Segmentation_with_only_SMB.ipynb
│       ├── Client_Segmentation_with_only_SMB.py
│       └── (same structure as above, filtered to SMB clients)
│
└── Process/                                                # Research & planning documentation
    ├── Code_Building_Plan/                                 # Iterative methodology plans (V1→V3)
    │   ├── Client_Segmentation_Modeling_Plan_(Revised_V2.0).txt
    │   ├── Phase2_Diversified_Modeling_and_Dimensionality_Reduction_Strategy.txt
    │   ├── Summary_Final_Model_Competition_Plan_(V2.0).txt
    │   ├── Final_Execution_Plan_(Mentor_Revised_V3.0).txt
    │   └── Final_Decision_Confirmation.txt
    ├── Company_Team_Model_Examples/                         # Internal reference: prior GCC analysis
    └── Table_Introductions/                                # Data source documentation (A through H)
        ├── A/ ... H/                                       # Per-table schema docs with English translations
        ├── Ad_Performance_Calculation.png
        └── Table_Data_Source_Details.png
```

## Tech Stack

| Category | Tools |
|---|---|
| **Data Infrastructure** | Alibaba Cloud MaxCompute, DataWorks, SQL |
| **Core Libraries** | pandas, NumPy, scikit-learn, SciPy, matplotlib, seaborn |
| **ML Models** | K-Means, GMM, Agglomerative Clustering, Birch, DBSCAN, MiniSom (SOM) |
| **Techniques** | PCA, ANOVA F-test, Silhouette Analysis, BIC, Decision Tree Validation, Confidence Intervals |
| **Environment** | Jupyter Notebook, Python 3.x |

## Key Results

- Segmented ~10,000 active B2B advertisers into 3 and 4 distinct operational tiers
- All 9 core KPIs showed statistically significant separation across clusters (ANOVA p < 0.05)
- Delivered actionable client profiles enabling differentiated service strategies: high-touch CSM for high-value segments, automated self-service for long-tail segments, targeted incubation for growth-potential segments
- Decision tree validation confirmed cluster definitions could be explained by 2-3 simple business rules

## Author

**Fergie (Yishu) Yang**
Machine Learning Engineer, Meetsocial — SinoClick Business Unit
