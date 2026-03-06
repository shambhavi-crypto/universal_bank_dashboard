import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score, roc_curve
import plotly.graph_objects as go

class LoanPredictor:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.feature_columns = ['Age', 'Experience', 'Income', 'Family', 'CCAvg', 'Education', 'Mortgage', 'Securities Account', 'CD Account', 'Online', 'CreditCard']
        self.is_trained = False
        
    def train(self, df):
        available_cols = [c for c in self.feature_columns if c in df.columns]
        self.feature_columns = available_cols
        X, y = df[available_cols], df['Personal Loan']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        X_train_scaled, X_test_scaled = self.scaler.fit_transform(X_train), self.scaler.transform(X_test)
        self.model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, class_weight='balanced')
        self.model.fit(X_train_scaled, y_train)
        y_pred, y_pred_proba = self.model.predict(X_test_scaled), self.model.predict_proba(X_test_scaled)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        self.is_trained = True
        return {
            'metrics': {'accuracy': accuracy_score(y_test, y_pred), 'precision': precision_score(y_test, y_pred, zero_division=0), 'recall': recall_score(y_test, y_pred, zero_division=0), 'f1': f1_score(y_test, y_pred, zero_division=0), 'roc_auc': roc_auc_score(y_test, y_pred_proba)},
            'confusion_matrix': confusion_matrix(y_test, y_pred),
            'feature_importance': pd.DataFrame({'Feature': self.feature_columns, 'Importance': self.model.feature_importances_}).sort_values('Importance', ascending=False),
            'roc_data': {'fpr': fpr, 'tpr': tpr}
        }
    
    def predict(self, customer_data):
        if not self.is_trained: return None
        input_df = pd.DataFrame([customer_data])
        for col in self.feature_columns:
            if col not in input_df.columns: input_df[col] = 0
        prob = self.model.predict_proba(self.scaler.transform(input_df[self.feature_columns]))[0]
        return {'prediction': 'Likely to Accept' if prob[1] >= 0.5 else 'Unlikely to Accept', 'probability_accept': prob[1], 'probability_reject': prob[0]}

def create_confusion_matrix_chart(cm):
    fig = go.Figure(data=go.Heatmap(z=cm, x=['Predicted Reject', 'Predicted Accept'], y=['Actual Reject', 'Actual Accept'], colorscale='Blues', text=cm, texttemplate='%{text}', textfont={"size": 16}))
    fig.update_layout(title='Confusion Matrix', height=400)
    return fig

def create_roc_curve_chart(fpr, tpr, auc_score):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name=f'ROC (AUC={auc_score:.3f})', line=dict(color='#1f77b4', width=2)))
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', name='Random', line=dict(color='gray', dash='dash')))
    fig.update_layout(title='ROC Curve', xaxis_title='False Positive Rate', yaxis_title='True Positive Rate', height=400)
    return fig

def get_customer_segments(df):
    segments = {}
    high = df[(df['Income'] >= 100) & (df['Education'] >= 2) & (df['CCAvg'] >= 3)]
    segments['High Potential'] = {'count': len(high), 'acceptance_rate': high['Personal Loan'].mean() * 100 if len(high) > 0 else 0, 'description': 'High income, Graduate+, High CC spending'}
    medium = df[(df['Income'] >= 50) & (df['Income'] < 100) & (df['CCAvg'] >= 2)]
    segments['Medium Potential'] = {'count': len(medium), 'acceptance_rate': medium['Personal Loan'].mean() * 100 if len(medium) > 0 else 0, 'description': 'Medium income, Moderate CC spending'}
    cd = df[df['CD Account'] == 1]
    segments['CD Account Holders'] = {'count': len(cd), 'acceptance_rate': cd['Personal Loan'].mean() * 100 if len(cd) > 0 else 0, 'description': 'Customers with CD accounts'}
    return segments

def generate_recommendations(df, model_results):
    accepted, rejected = df[df['Personal Loan'] == 1], df[df['Personal Loan'] == 0]
    recs = []
    if len(accepted) > 0 and len(rejected) > 0:
        recs.append({'title': '💰 Target High-Income Customers', 'description': f"Avg income of acceptors: ${accepted['Income'].mean():.0f}K vs ${rejected['Income'].mean():.0f}K", 'action': 'Focus on customers with income > $100K', 'priority': 'High'})
    if 'CD Account' in df.columns:
        cd_rate = df[df['CD Account'] == 1]['Personal Loan'].mean() * 100
        recs.append({'title': '🏦 Prioritize CD Account Holders', 'description': f'CD holders have {cd_rate:.1f}% acceptance rate', 'action': 'Create exclusive offers for CD customers', 'priority': 'High'})
    recs.append({'title': '🎓 Focus on Graduates', 'description': 'Higher education correlates with acceptance', 'action': 'Partner with professional associations', 'priority': 'Medium'})
    recs.append({'title': '💳 Target High CC Spenders', 'description': 'High credit card usage indicates loan interest', 'action': 'Pre-approved offers for CC spending > $4K/month', 'priority': 'Medium'})
    return recs

def generate_personalized_offers(prediction_result, customer_data):
    prob = prediction_result['probability_accept']
    if prob >= 0.7:
        return [{'type': '⭐ Premium Personal Loan', 'interest_rate': '8.5% - 10.5%', 'max_amount': '$50,000', 'tenure': 'Up to 7 years', 'special': 'Zero processing fee + Instant approval'}]
    elif prob >= 0.4:
        return [{'type': '✅ Standard Personal Loan', 'interest_rate': '10.5% - 13.5%', 'max_amount': '$30,000', 'tenure': 'Up to 5 years', 'special': '50% off processing fee'}]
    else:
        return [{'type': '📋 Starter Personal Loan', 'interest_rate': '12.5% - 15.5%', 'max_amount': '$15,000', 'tenure': 'Up to 3 years', 'special': 'Build credit history program'}]