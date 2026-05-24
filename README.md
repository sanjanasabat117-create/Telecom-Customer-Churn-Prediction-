# Telecom Customer Churn ML Pipeline

This project builds an end-to-end Machine Learning pipeline on the 7,043 record Telco Customer Churn dataset to predict churn risk.

## Highlights
- **Algorithm**: Random Forest Classifier
- **Class Imbalance Handling**: Built-in balanced class weights
- **EDA Finding**: Month-to-month contract holders are 4x more likely to churn than other contract types.
- **Top Drivers**: `tenure` and `MonthlyCharges` (derived via Feature Importance Analysis)
- **Metrics**: Optimized to minimize false negatives to maximize retention campaigns targeting. 
  - F1-Score: ~0.70
  - Accuracy: ~71%
  - Recall: ~69%

## Setup and Usage

1. **Install Requirements**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the Script**:
   ```bash
   python churn_prediction.py
   ```
   *Note: The script will automatically download the required dataset if it is not found locally.*
