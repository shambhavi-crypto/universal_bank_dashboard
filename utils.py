import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np

COLORS = {'primary': '#1f77b4', 'accepted': '#2ecc71', 'rejected': '#e74c3c'}
LOAN_COLORS = {'Accepted': '#2ecc71', 'Rejected': '#e74c3c'}

def create_donut_chart(df, column, title, hole=0.6):
    value_counts = df[column].value_counts()
    colors = [COLORS['accepted'] if 'Accept' in str(l) else COLORS['rejected'] for l in value_counts.index] if 'Loan' in column or 'Status' in column else px.colors.qualitative.Set2[:len(value_counts)]
    fig = go.Figure(data=[go.Pie(labels=value_counts.index, values=value_counts.values, hole=hole, marker_colors=colors, textinfo='label+percent', textposition='outside')])
    fig.update_layout(title=dict(text=title, x=0.5), showlegend=True, height=400, margin=dict(t=60, b=60, l=20, r=20))
    return fig

def create_histogram(df, column, title, nbins=30):
    fig = px.histogram(df, x=column, color='Loan_Status' if 'Loan_Status' in df.columns else None, nbins=nbins, title=title, color_discrete_map=LOAN_COLORS, barmode='overlay', opacity=0.7)
    fig.update_layout(xaxis_title=column, yaxis_title='Count', height=400)
    return fig

def create_box_plot(df, x_col, y_col, title):
    fig = px.box(df, x=x_col, y=y_col, color=x_col, color_discrete_map=LOAN_COLORS if x_col == 'Loan_Status' else None, title=title)
    fig.update_layout(height=400, showlegend=False)
    return fig

def create_correlation_heatmap(df):
    numeric_cols = [c for c in ['Age', 'Experience', 'Income', 'Family', 'CCAvg', 'Education', 'Mortgage', 'Securities Account', 'CD Account', 'Online', 'CreditCard', 'Personal Loan'] if c in df.columns]
    corr = df[numeric_cols].corr()
    fig = go.Figure(data=go.Heatmap(z=corr.values, x=corr.columns, y=corr.columns, colorscale='RdBu_r', zmid=0, text=np.round(corr.values, 2), texttemplate='%{text}', textfont={"size": 10}))
    fig.update_layout(title='Correlation Heatmap', height=600, xaxis=dict(tickangle=45))
    return fig

def create_sunburst_chart(df):
    data = df.groupby(['Education_Label', 'Income_Group', 'Loan_Status']).size().reset_index(name='Count')
    fig = px.sunburst(data, path=['Education_Label', 'Income_Group', 'Loan_Status'], values='Count', color='Loan_Status', color_discrete_map=LOAN_COLORS, title='Drill-Down: Education → Income → Loan Status (Click to Expand)')
    fig.update_layout(height=500)
    return fig

def create_treemap(df):
    data = df.groupby(['Education_Label', 'Family_Label', 'Loan_Status']).size().reset_index(name='Count')
    fig = px.treemap(data, path=['Education_Label', 'Family_Label', 'Loan_Status'], values='Count', color='Loan_Status', color_discrete_map=LOAN_COLORS, title='Hierarchical View (Click to Drill Down)')
    fig.update_layout(height=500)
    return fig

def create_scatter_plot(df, x_col, y_col, title):
    fig = px.scatter(df, x=x_col, y=y_col, color='Loan_Status', color_discrete_map=LOAN_COLORS, title=title, opacity=0.6)
    fig.update_layout(height=450)
    return fig

def create_grouped_bar(df, group_col, title):
    grouped = df.groupby(group_col)['Personal Loan'].agg(['sum', 'count']).reset_index()
    grouped['Acceptance_Rate'] = (grouped['sum'] / grouped['count'] * 100).round(2)
    grouped.columns = [group_col, 'Accepted', 'Total', 'Acceptance_Rate']
    fig = go.Figure(go.Bar(x=grouped[group_col], y=grouped['Acceptance_Rate'], text=grouped['Acceptance_Rate'].apply(lambda x: f'{x:.1f}%'), textposition='outside', marker_color=COLORS['primary']))
    fig.update_layout(title=title, xaxis_title=group_col, yaxis_title='Acceptance Rate (%)', height=400, showlegend=False)
    return fig, grouped

def create_comparison_chart(comparison_df):
    fig = go.Figure()
    fig.add_trace(go.Bar(name='Accepted', x=comparison_df['Metric'], y=comparison_df['Accepted'], marker_color=COLORS['accepted'], text=comparison_df['Accepted'], textposition='outside'))
    fig.add_trace(go.Bar(name='Rejected', x=comparison_df['Metric'], y=comparison_df['Rejected'], marker_color=COLORS['rejected'], text=comparison_df['Rejected'], textposition='outside'))
    fig.update_layout(title='Comparison: Accepted vs Rejected', barmode='group', height=500, xaxis_tickangle=-45)
    return fig

def create_feature_importance_chart(importance_df):
    fig = px.bar(importance_df.sort_values('Importance', ascending=True), x='Importance', y='Feature', orientation='h', title='Feature Importance', color='Importance', color_continuous_scale='Viridis')
    fig.update_layout(height=450)
    return fig

def create_gauge_chart(value, title, max_val=100):
    fig = go.Figure(go.Indicator(mode="gauge+number", value=value, domain={'x': [0, 1], 'y': [0, 1]}, title={'text': title}, gauge={'axis': {'range': [None, max_val]}, 'bar': {'color': COLORS['primary']}, 'steps': [{'range': [0, 30], 'color': '#ffcccc'}, {'range': [30, 70], 'color': '#ffffcc'}, {'range': [70, 100], 'color': '#ccffcc'}]}))
    fig.update_layout(height=300)
    return fig

def create_multi_donut(df):
    fig = make_subplots(rows=2, cols=2, specs=[[{'type': 'domain'}]*2]*2, subplot_titles=['Securities Account', 'CD Account', 'Online Banking', 'Credit Card'])
    for i, (service, pos) in enumerate(zip(['Securities Account', 'CD Account', 'Online', 'CreditCard'], [(1,1), (1,2), (2,1), (2,2)])):
        if service in df.columns:
            counts = df[service].value_counts()
            fig.add_trace(go.Pie(labels=['No', 'Yes'], values=[counts.get(0, 0), counts.get(1, 0)], hole=0.5, marker_colors=['#e74c3c', '#2ecc71'], textinfo='percent', showlegend=False), row=pos[0], col=pos[1])
    fig.update_layout(title='Banking Services Distribution', height=500)
    return fig