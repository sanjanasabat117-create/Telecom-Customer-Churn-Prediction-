import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import requests
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, f1_score, recall_score, confusion_matrix
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
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df.dropna(inplace=True)
    
    # ---------------------------------------------------------
    # Visualization 1: Churn by Contract Type
    # ---------------------------------------------------------
    plt.figure(figsize=(8, 5))
    sns.countplot(data=df, x='Contract', hue='Churn', palette='Set2')
    plt.title('Customer Churn by Contract Type')
    plt.ylabel('Number of Customers')
    plt.xlabel('Contract Type')
    plt.tight_layout()
    plt.savefig('churn_by_contract.png')
    plt.close()
    
    churn_rates = df.groupby('Contract')['Churn'].value_counts(normalize=True).unstack()
    mtm_churn_rate = churn_rates.loc['Month-to-month', 'Yes']
    other_churn_rate = (df[df['Contract'] != 'Month-to-month']['Churn'] == 'Yes').mean()
    ratio = mtm_churn_rate / other_churn_rate
    print(f"Month-to-month churn rate: {mtm_churn_rate:.1%}")
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
    
    # Train Random Forest Classifier
    clf = RandomForestClassifier(
        n_estimators=100, 
        max_depth=5, 
        class_weight='balanced', 
        random_state=42
    )
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    
    print("--- Model Evaluation ---")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.2%}")
    print(f"F1-Score: {f1_score(y_test, y_pred):.2f}")
    print(f"Recall:   {recall_score(y_test, y_pred):.2%}\n")
    
    # ---------------------------------------------------------
    # Visualization 2: Confusion Matrix
    # ---------------------------------------------------------
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['No Churn', 'Churn'], yticklabels=['No Churn', 'Churn'])
    plt.title('Confusion Matrix')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')
    plt.close()
    
    # ---------------------------------------------------------
    # Visualization 3: Feature Importance
    # ---------------------------------------------------------
    importances = clf.feature_importances_
    features = X.columns
    feat_df = pd.DataFrame({'Feature': features, 'Importance': importances}).sort_values(by='Importance', ascending=False)
    
    plt.figure(figsize=(10, 6))
    sns.barplot(data=feat_df.head(10), x='Importance', y='Feature', palette='viridis')
    plt.title('Top 10 Drivers of Customer Churn')
    plt.xlabel('Relative Importance')
    plt.ylabel('Feature')
    plt.tight_layout()
    plt.savefig('feature_importance.png')
    plt.close()
    
    print("Visualizations saved as PNG files: churn_by_contract.png, confusion_matrix.png, feature_importance.png")

if __name__ == "__main__":
    main()
