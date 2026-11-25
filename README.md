# Customer Segmentation using RFM Analysis and K-Means Clustering

## Project Overview
This project applies RFM (Recency, Frequency, Monetary) analysis and K-Means clustering to segment customers based on their purchasing behaviour.

An interactive Streamlit dashboard allows users to explore customer segments, understand purchasing patterns, view business recommendations, and predict the segment of a customer using RFM values.

## Business Problem
Businesses need to understand customer purchasing behaviour to improve customer retention, identify valuable customers, and design targeted marketing strategies.

This project addresses that problem by grouping customers into meaningful segments using historical transaction data.

## Dataset
- **Source:** UCI Machine Learning Repository – Online Retail II
- **Records:** Historical online retail transactions
- **Features:** Invoice, StockCode, Quantity, InvoiceDate, Price, Customer ID, Country

## Technologies Used
- Python
- Pandas and NumPy
- Matplotlib and Seaborn
- Scikit-learn
- Streamlit
- Plotly
- Joblib

## Methodology

1. Data loading and exploration
2. Data cleaning and preprocessing
3. Exploratory Data Analysis (EDA)
4. RFM feature engineering
5. Log transformation to reduce skewness
6. Feature scaling using StandardScaler
7. K-Means clustering
8. Cluster evaluation using the Elbow Method and Silhouette Score
9. Customer segment interpretation
10. Interactive dashboard development

## Customer Segments

| Segment | Description | Recommended Strategy |
|---|---|---|
| Champions | Recent, frequent and high-spending customers | Loyalty rewards and exclusive offers |
| At-Risk Valuable | Previously valuable customers who have not purchased recently | Personalized win-back campaigns |
| Recent Low-Frequency | Recently active customers with relatively low purchase frequency | Encourage repeat purchases |
| Hibernating | Inactive customers with relatively low historical spending | Low-cost reactivation campaigns |

## Model Evaluation

- Algorithm: K-Means Clustering
- Selected clusters: 4
- Silhouette Score: 0.3354
- Alternative K=2 silhouette score: 0.4135

Four clusters were selected to provide more actionable business segments, despite K=2 achieving a higher silhouette score.

## Streamlit Dashboard Features

- Executive Overview
- Customer Segment Analysis
- Customer Explorer
- Business Recommendations
- Interactive Customer Segment Prediction
- Downloadable Customer Segmentation Report

## Project Structure

```text
Customer-Segmentation-RFM/
│
├── app.py
├── requirements.txt
├── README.md
│
├── data/
│   ├── Original dataset
│   └── Cleaned transaction data
│
├── outputs/
│   ├── customer_segmentation_results.csv
│   └── customer_segment_summary.csv
│
├── models/
│   ├── kmeans_model.pkl
│   └── scaler.pkl
│
└── notebooks/
    └── customer_segmentation.ipynb
```

## Installation and Setup

Clone the repository:

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
cd Customer-Segmentation-RFM
```

Create a virtual environment:

```bash
python -m venv venv
```

Activate it on Windows:

```bash
venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the dashboard:

```bash
streamlit run app.py
```

## Future Improvements

- Integrate customer lifetime value estimation.
- Explore alternative clustering algorithms.
- Add automated customer retention campaign recommendations.
- Deploy the dashboard online.
- Improve segment validation using additional business metrics.

## Author

**Anurag Sharma**  
B.Tech Computer Science and Engineering  
Amritsar Group of Colleges