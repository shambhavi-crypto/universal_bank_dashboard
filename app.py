import streamlit as st
import pandas as pd
import numpy as np
from data_processor import load_data, clean_data, get_summary_statistics, get_comparison_stats
from utils import (create_donut_chart, create_histogram, create_box_plot, create_correlation_heatmap, 
                   create_sunburst_chart, create_treemap, create_scatter_plot, create_grouped_bar, 
                   create_comparison_chart, create_feature_importance_chart, create_gauge_chart, create_multi_donut, COLORS, LOAN_COLORS)
from model import (LoanPredictor, create_confusion_matrix_chart, create_roc_curve_chart,
                   get_customer_segments, generate_recommendations, generate_personalized_offers)

st.set_page_config(page_title="Universal Bank - Loan Analysis", page_icon="🏦", layout="wide")

st.markdown("""
<style>
    .metric-card {background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 20px; border-radius: 10px; text-align: center; color: white;}
    .insight-box {background-color: #f0f7ff; border-left: 4px solid #1f77b4; padding: 15px; margin: 10px 0; border-radius: 0 8px 8px 0;}
    .recommendation-card {background: white; border: 1px solid #e0e0e0; border-radius: 10px; padding: 20px; margin: 10px 0;}
    .priority-high {color: #d62728; font-weight: bold;}
    .priority-medium {color: #ff7f0e; font-weight: bold;}
    .priority-low {color: #2ca02c; font-weight: bold;}
</style>
""", unsafe_allow_html=True)

if 'model_trained' not in st.session_state: st.session_state.model_trained = False
if 'predictor' not in st.session_state: st.session_state.predictor = LoanPredictor()

@st.cache_data
def load_and_process_data():
    return clean_data(load_data())

df = load_and_process_data()
stats = get_summary_statistics(df)

with st.sidebar:
    st.title("🏦 Universal Bank")
    st.markdown("---")
    page = st.radio("📍 Navigation", ["🏠 Overview", "📊 Descriptive Analysis", "🔍 Diagnostic Analysis", "📈 Predictive Analysis", "💡 Prescriptive Analysis", "🎯 Customer Predictor"])
    st.markdown("---")
    st.subheader("🔍 Filters")
    income_range = st.slider("Income Range ($K)", int(df['Income'].min()), int(df['Income'].max()), (int(df['Income'].min()), int(df['Income'].max())))
    education_filter = st.multiselect("Education Level", df['Education_Label'].unique().tolist(), df['Education_Label'].unique().tolist())
    family_filter = st.multiselect("Family Size", sorted(df['Family'].unique().tolist()), sorted(df['Family'].unique().tolist()))
    df_filtered = df[(df['Income'] >= income_range[0]) & (df['Income'] <= income_range[1]) & (df['Education_Label'].isin(education_filter)) & (df['Family'].isin(family_filter))]
    st.info(f"📊 {len(df_filtered):,} of {len(df):,} customers")

if page == "🏠 Overview":
    st.markdown("<h1 style='text-align:center; color:#1f77b4;'>🏦 Universal Bank Personal Loan Analysis</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align:center; color:#666;'>Understanding Customer Behavior for Personal Loan Acceptance</p>", unsafe_allow_html=True)
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Customers", f"{stats['total_customers']:,}")
    col2.metric("Loans Accepted", f"{stats['loan_accepted']:,}", f"{stats['acceptance_rate']:.1f}%")
    col3.metric("Loans Rejected", f"{stats['loan_rejected']:,}")
    col4.metric("Avg Income", f"${stats['avg_income']:.0f}K")
    st.markdown("---")
    c1, c2 = st.columns(2)
    with c1: st.plotly_chart(create_donut_chart(df_filtered, 'Loan_Status', 'Personal Loan Distribution'), use_container_width=True)
    with c2: st.plotly_chart(create_sunburst_chart(df_filtered), use_container_width=True)
    st.markdown("### 💡 Key Insights")
    c1, c2, c3 = st.columns(3)
    with c1: st.markdown(f"<div class='insight-box'><strong>📊 Average Income</strong><br>Accepted: ${df[df['Personal Loan']==1]['Income'].mean():.0f}K<br>Rejected: ${df[df['Personal Loan']==0]['Income'].mean():.0f}K</div>", unsafe_allow_html=True)
    with c2: st.markdown(f"<div class='insight-box'><strong>💳 CD Account Impact</strong><br>CD Holders: {df[df['CD Account']==1]['Personal Loan'].mean()*100:.1f}% acceptance<br>Non-CD: {df[df['CD Account']==0]['Personal Loan'].mean()*100:.1f}%</div>", unsafe_allow_html=True)
    with c3: 
        top_edu = df.groupby('Education_Label')['Personal Loan'].mean().idxmax()
        st.markdown(f"<div class='insight-box'><strong>🎓 Top Segment</strong><br>{top_edu}<br>Rate: {df.groupby('Education_Label')['Personal Loan'].mean().max()*100:.1f}%</div>", unsafe_allow_html=True)

elif page == "📊 Descriptive Analysis":
    st.markdown("## 📊 Descriptive Analysis")
    tab1, tab2, tab3 = st.tabs(["Demographics", "Financial Profile", "Banking Services"])
    with tab1:
        c1, c2 = st.columns(2)
        with c1: st.plotly_chart(create_histogram(df_filtered, 'Age', 'Age Distribution'), use_container_width=True)
        with c2: st.plotly_chart(create_grouped_bar(df_filtered, 'Age_Group', 'Acceptance by Age')[0], use_container_width=True)
        c1, c2 = st.columns(2)
        with c1: st.plotly_chart(create_donut_chart(df_filtered, 'Education_Label', 'Education Distribution'), use_container_width=True)
        with c2: st.plotly_chart(create_grouped_bar(df_filtered, 'Education_Label', 'Acceptance by Education')[0], use_container_width=True)
    with tab2:
        c1, c2 = st.columns(2)
        with c1: st.plotly_chart(create_histogram(df_filtered, 'Income', 'Income Distribution', nbins=40), use_container_width=True)
        with c2: st.plotly_chart(create_box_plot(df_filtered, 'Loan_Status', 'Income', 'Income by Loan Status'), use_container_width=True)
        c1, c2 = st.columns(2)
        with c1: st.plotly_chart(create_histogram(df_filtered, 'CCAvg', 'CC Spending Distribution', nbins=40), use_container_width=True)
        with c2: st.plotly_chart(create_grouped_bar(df_filtered, 'Income_Group', 'Acceptance by Income')[0], use_container_width=True)
    with tab3:
        st.plotly_chart(create_multi_donut(df_filtered), use_container_width=True)

elif page == "🔍 Diagnostic Analysis":
    st.markdown("## 🔍 Diagnostic Analysis")
    tab1, tab2 = st.tabs(["Comparison", "Correlation"])
    with tab1:
        st.plotly_chart(create_comparison_chart(get_comparison_stats(df_filtered)), use_container_width=True)
        c1, c2 = st.columns(2)
        with c1: st.plotly_chart(create_scatter_plot(df_filtered, 'Income', 'CCAvg', 'Income vs CC Spending'), use_container_width=True)
        with c2: st.plotly_chart(create_treemap(df_filtered), use_container_width=True)
    with tab2:
        st.plotly_chart(create_correlation_heatmap(df_filtered), use_container_width=True)

elif page == "📈 Predictive Analysis":
    st.markdown("## 📈 Predictive Analysis")
    if st.button("🚀 Train Model", type="primary"):
        with st.spinner("Training..."): 
            st.session_state.model_results = st.session_state.predictor.train(df)
            st.session_state.model_trained = True
        st.success("✅ Model trained!")
    if st.session_state.model_trained:
        m = st.session_state.model_results['metrics']
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Accuracy", f"{m['accuracy']*100:.1f}%")
        c2.metric("Precision", f"{m['precision']*100:.1f}%")
        c3.metric("Recall", f"{m['recall']*100:.1f}%")
        c4.metric("F1 Score", f"{m['f1']*100:.1f}%")
        c5.metric("ROC AUC", f"{m['roc_auc']*100:.1f}%")
        c1, c2 = st.columns(2)
        with c1: st.plotly_chart(create_confusion_matrix_chart(st.session_state.model_results['confusion_matrix']), use_container_width=True)
        with c2: st.plotly_chart(create_roc_curve_chart(st.session_state.model_results['roc_data']['fpr'], st.session_state.model_results['roc_data']['tpr'], m['roc_auc']), use_container_width=True)
        st.plotly_chart(create_feature_importance_chart(st.session_state.model_results['feature_importance']), use_container_width=True)

elif page == "💡 Prescriptive Analysis":
    st.markdown("## 💡 Prescriptive Analysis")
    st.subheader("🎯 Customer Segments")
    segments = get_customer_segments(df_filtered)
    cols = st.columns(3)
    for i, (name, data) in enumerate(segments.items()):
        with cols[i]: st.markdown(f"<div class='recommendation-card'><h4>{name}</h4><p>Count: {data['count']:,}</p><p>Acceptance: {data['acceptance_rate']:.1f}%</p><small>{data['description']}</small></div>", unsafe_allow_html=True)
    st.subheader("📋 Recommendations")
    recs = generate_recommendations(df_filtered, st.session_state.model_results) if st.session_state.model_trained else generate_recommendations(df_filtered, {})
    for r in recs: st.markdown(f"<div class='recommendation-card'><h4>{r['title']}</h4><p>{r['description']}</p><p><strong>Action:</strong> {r['action']}</p><p class='priority-{r['priority'].lower()}'>Priority: {r['priority']}</p></div>", unsafe_allow_html=True)

elif page == "🎯 Customer Predictor":
    st.markdown("## 🎯 Customer Predictor")
    if not st.session_state.model_trained:
        st.warning("⚠️ Train the model first in 'Predictive Analysis'")
        if st.button("Train Now"):
            with st.spinner("Training..."): 
                st.session_state.model_results = st.session_state.predictor.train(df)
                st.session_state.model_trained = True
            st.rerun()
    else:
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Customer Details")
            age = st.slider("Age", 20, 70, 35)
            experience = st.slider("Experience", 0, 45, 10)
            income = st.slider("Income ($K)", 10, 250, 80)
            family = st.selectbox("Family Size", [1, 2, 3, 4])
            ccavg = st.slider("CC Spending ($K/month)", 0.0, 10.0, 2.0, 0.1)
            education = st.selectbox("Education", [1, 2, 3], format_func=lambda x: {1:'Undergraduate', 2:'Graduate', 3:'Advanced'}[x])
            mortgage = st.slider("Mortgage ($K)", 0, 700, 0)
            ca, cb = st.columns(2)
            with ca: securities, cd_account = st.checkbox("Securities Account"), st.checkbox("CD Account")
            with cb: online, credit_card = st.checkbox("Online Banking"), st.checkbox("Credit Card")
            predict_btn = st.button("🔮 Predict", type="primary")
        with c2:
            if predict_btn:
                result = st.session_state.predictor.predict({'Age': age, 'Experience': experience, 'Income': income, 'Family': family, 'CCAvg': ccavg, 'Education': education, 'Mortgage': mortgage, 'Securities Account': int(securities), 'CD Account': int(cd_account), 'Online': int(online), 'CreditCard': int(credit_card)})
                if result:
                    st.plotly_chart(create_gauge_chart(result['probability_accept']*100, 'Acceptance Probability'), use_container_width=True)
                    if result['probability_accept'] >= 0.5: st.success(f"✅ {result['prediction']} ({result['probability_accept']*100:.1f}%)")
                    else: st.error(f"❌ {result['prediction']} ({result['probability_accept']*100:.1f}%)")
                    st.subheader("💰 Personalized Offers")
                    for offer in generate_personalized_offers(result, {}):
                        st.markdown(f"<div class='recommendation-card'><h4>{offer['type']}</h4><p>Rate: {offer.get('interest_rate','N/A')}</p><p>Max: {offer.get('max_amount','N/A')}</p><p>Special: {offer.get('special','')}</p></div>", unsafe_allow_html=True)

st.markdown("---")
st.markdown("<center>🏦 Universal Bank Dashboard | Built with Streamlit</center>", unsafe_allow_html=True)