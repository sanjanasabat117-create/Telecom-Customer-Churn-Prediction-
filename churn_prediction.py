import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import requests
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, f1_score, recall_score
from sklearn.preprocessing import LabelEncoder

def download_data():
    url = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"
    file_path = "Telco-Customer-Churn.csv"
    if not os.path.exists(file_path):
        print(f"Downloading dataset from {url}...")
        response = requests.get(url)
        with open(file_path, 'wb') as f:
            f.write(response.content)
        print("Download complete.")
    return file_path

def main():
    file_path = download_data()
    df = pd.read_csv(file_path)
    print(f"Loaded dataset with {len(df)} records.\n")
    
    # Preprocessing
    # TotalCharges is object because of some spaces. Convert to numeric
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df.dropna(inplace=True)
    
    # EDA: Month-to-month contract holders are 4x more likely to churn
    churn_rates = df.groupby('Contract')['Churn'].value_counts(normalize=True).unstack()
    print("--- Exploratory Data Analysis ---")
    print("Churn Rates by Contract Type:")
    print(churn_rates)
    
    mtm_churn_rate = churn_rates.loc['Month-to-month', 'Yes']
    other_churn_rate = (df[df['Contract'] != 'Month-to-month']['Churn'] == 'Yes').mean()
    ratio = mtm_churn_rate / other_churn_rate
    print(f"\nMonth-to-month churn rate: {mtm_churn_rate:.1%}")
    print(f"Other contracts churn rate: {other_churn_rate:.1%}")
    print(f"Month-to-month contract holders are {ratio:.1f}x more likely to churn.\n")
    
    # Prepare data for modeling
    X = df.drop(['customerID', 'Churn'], axis=1)
    y = df['Churn'].map({'Yes': 1, 'No': 0})
    
    # Encode categoricals
    for col in X.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col])
        
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train Random Forest Classifier with balanced class weights
    # We use a max_depth to prevent overfitting and hit the desired metrics
    print("--- Model Training ---")
    clf = RandomForestClassifier(
        n_estimators=100, 
        max_depth=5, 
        class_weight='balanced', 
        random_state=42
    )
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    
    print("Metrics:")
    print(f"Accuracy: {acc:.2%}")
    print(f"F1-Score: {f1:.2f}")
    print(f"Recall:   {rec:.2%}")
    print()
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    
    # Feature Importance
    importances = clf.feature_importances_
    features = X.columns
    feat_df = pd.DataFrame({'Feature': features, 'Importance': importances}).sort_values(by='Importance', ascending=False)
    
    print("--- Feature Importance ---")
    print("Top 5 Drivers of Churn:")
    print(feat_df.head(5))
    
if __name__ == "__main__":
    main()
