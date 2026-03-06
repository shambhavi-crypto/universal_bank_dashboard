import pandas as pd
import numpy as np
import os

def load_data():
    try:
        possible_paths = ['data/bank_data.csv', './data/bank_data.csv', 'bank_data.csv']
        for path in possible_paths:
            if os.path.exists(path):
                return pd.read_csv(path)
        return create_embedded_data()
    except:
        return create_embedded_data()

def create_embedded_data():
    np.random.seed(42)
    n = 5000
    ages = np.random.randint(23, 68, n)
    experience = np.clip(ages - 23 + np.random.randint(-3, 3, n), -3, 43)
    income = np.clip(np.random.exponential(50, n) + 10, 8, 224).astype(int)
    family = np.random.choice([1, 2, 3, 4], n, p=[0.25, 0.3, 0.25, 0.2])
    ccavg = np.clip(np.random.exponential(1.5, n), 0, 10).round(2)
    education = np.random.choice([1, 2, 3], n, p=[0.4, 0.35, 0.25])
    mortgage = np.random.choice([0]*7 + list(range(75, 650, 25)), n)
    securities = np.random.choice([0, 1], n, p=[0.9, 0.1])
    cd_account = np.random.choice([0, 1], n, p=[0.94, 0.06])
    online = np.random.choice([0, 1], n, p=[0.4, 0.6])
    creditcard = np.random.choice([0, 1], n, p=[0.7, 0.3])
    loan_prob = 0.05 + (income > 100) * 0.15 + (cd_account == 1) * 0.25 + (education == 3) * 0.05 + (ccavg > 3) * 0.08
    personal_loan = (np.random.random(n) < np.clip(loan_prob, 0, 0.9)).astype(int)
    zip_codes = np.random.choice([90089, 91107, 91320, 91711, 92037, 92121, 93106, 93407, 94025, 94305, 94720, 95014, 95616], n)
    
    return pd.DataFrame({
        'ID': range(1, n+1), 'Age': ages, 'Experience': experience, 'Income': income,
        'ZIP Code': zip_codes, 'Family': family, 'CCAvg': ccavg, 'Education': education,
        'Mortgage': mortgage, 'Securities Account': securities, 'CD Account': cd_account,
        'Online': online, 'CreditCard': creditcard, 'Personal Loan': personal_loan
    })

def clean_data(df):
    df_clean = df.copy()
    df_clean['Experience'] = df_clean['Experience'].apply(lambda x: max(0, x))
    df_clean['Age_Group'] = pd.cut(df_clean['Age'], bins=[0, 30, 40, 50, 60, 100], labels=['<30', '30-40', '40-50', '50-60', '60+'])
    df_clean['Income_Group'] = pd.cut(df_clean['Income'], bins=[0, 50, 100, 150, 300], labels=['Low (<50K)', 'Medium (50-100K)', 'High (100-150K)', 'Very High (>150K)'])
    df_clean['CCAvg_Group'] = pd.cut(df_clean['CCAvg'], bins=[-1, 2, 4, 6, 15], labels=['Low (<2K)', 'Medium (2-4K)', 'High (4-6K)', 'Very High (>6K)'])
    df_clean['Education_Label'] = df_clean['Education'].map({1: 'Undergraduate', 2: 'Graduate', 3: 'Advanced/Professional'})
    df_clean['Family_Label'] = df_clean['Family'].map({1: 'Single', 2: 'Couple', 3: 'Small Family', 4: 'Large Family'})
    df_clean['Loan_Status'] = df_clean['Personal Loan'].map({0: 'Rejected', 1: 'Accepted'})
    return df_clean

def get_summary_statistics(df):
    return {
        'total_customers': len(df),
        'loan_accepted': int(df['Personal Loan'].sum()),
        'loan_rejected': int(len(df) - df['Personal Loan'].sum()),
        'acceptance_rate': float((df['Personal Loan'].sum() / len(df)) * 100),
        'avg_age': float(df['Age'].mean()),
        'avg_income': float(df['Income'].mean()),
        'avg_ccavg': float(df['CCAvg'].mean()),
        'avg_mortgage': float(df['Mortgage'].mean())
    }

def get_comparison_stats(df):
    accepted = df[df['Personal Loan'] == 1]
    rejected = df[df['Personal Loan'] == 0]
    metrics = ['Average Income ($K)', 'Average CC Spending ($K)', 'Average Mortgage ($K)', 'Average Age', 'CD Account (%)', 'Online Banking (%)']
    acc_vals = [accepted['Income'].mean(), accepted['CCAvg'].mean(), accepted['Mortgage'].mean(), accepted['Age'].mean(), accepted['CD Account'].mean()*100, accepted['Online'].mean()*100] if len(accepted) > 0 else [0]*6
    rej_vals = [rejected['Income'].mean(), rejected['CCAvg'].mean(), rejected['Mortgage'].mean(), rejected['Age'].mean(), rejected['CD Account'].mean()*100, rejected['Online'].mean()*100] if len(rejected) > 0 else [0]*6
    comparison_df = pd.DataFrame({'Metric': metrics, 'Accepted': [round(v, 2) for v in acc_vals], 'Rejected': [round(v, 2) for v in rej_vals]})
    comparison_df['Difference'] = comparison_df['Accepted'] - comparison_df['Rejected']
    return comparison_df