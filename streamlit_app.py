import os
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import roc_auc_score, roc_curve
from joblib import load

# Optional (for explainability)
import shap
import matplotlib.pyplot as plt


warnings.filterwarnings("ignore")


# --------------- Paths ---------------
ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data"

MODEL_PATH = DATA_PATH / "churn_model.pkl"
PRED_PATH = DATA_PATH / "churn_predictions.csv"
HIGH_RISK_PATH = DATA_PATH / "high_risk_customers.csv"
MASTER_AI_PATH = DATA_PATH / "master_enhanced_with_ai.csv"


# --------------- Caching Loaders ---------------
@st.cache_data(show_spinner=False)
def load_csv_safe(path: Path) -> pd.DataFrame:
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame()


@st.cache_resource(show_spinner=False)
def load_model_safe(path: Path):
    if path.exists():
        try:
            return load(path)
        except Exception:
            return None
    return None


def build_features_from_master(order_df: pd.DataFrame) -> pd.DataFrame:
    if order_df.empty:
        return pd.DataFrame()

    # Ensure fields exist
    df = order_df.copy()
    if 'Order_ID' in df.columns and 'Customer_ID' not in df.columns:
        try:
            df['Customer_ID'] = df['Order_ID'].astype(str).str.extract(r'(\d+)')[0].astype(float).astype(int)
            df['Customer_ID'] = 'CUST' + df['Customer_ID'].astype(str).str.zfill(3)
        except Exception:
            pass

    # Map AI columns if present
    if 'Quality_Issue' in df.columns:
        df['Quality_Issue'] = df['Quality_Issue'].replace({'Perfect': 0, 'Minor': 1, 'Major': 2}).fillna(0)
    if 'urgency' in df.columns:
        df['urgency'] = df['urgency'].replace({'Low': 0, 'Medium': 1, 'High': 2}).fillna(0)
    if 'action' in df.columns:
        df['action'] = df['action'].replace({'None': 0, 'Free Shipping': 1, '₹500 Voucher': 2, 'Call Customer': 3}).fillna(0)

    for col in ['Delay_Days', 'Total_Cost', 'Customer_Rating', 'Distance_KM']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

    # Boolean to numeric
    for col in ['Is_Delayed', 'Severe_Delay']:
        if col in df.columns:
            df[col] = df[col].astype(float)

    group_cols = {
        'Total_Orders': ('Order_ID', 'count'),
        'Avg_Delay': ('Delay_Days', 'mean'),
        'Total_Spend': ('Total_Cost', 'sum'),
        'Avg_Rating': ('Customer_Rating', 'mean'),
        'Damage_Rate': ('Quality_Issue', lambda x: (pd.to_numeric(x, errors='coerce').fillna(0) > 0).mean()),
        'Delay_Rate': ('Is_Delayed', 'mean') if 'Is_Delayed' in df.columns else ('Order_ID', lambda s: np.nan),
        'Severe_Delay_Rate': ('Severe_Delay', 'mean') if 'Severe_Delay' in df.columns else ('Order_ID', lambda s: np.nan),
        'Avg_Distance': ('Distance_KM', 'mean') if 'Distance_KM' in df.columns else ('Order_ID', lambda s: np.nan),
        'Priority_Express': ('Priority', lambda x: (x == 'Express').mean()) if 'Priority' in df.columns else ('Order_ID', lambda s: 0.0),
        'Segment_SMB': ('Customer_Segment', lambda x: (x == 'SMB').mean()) if 'Customer_Segment' in df.columns else ('Order_ID', lambda s: 0.0),
        'Segment_Individual': ('Customer_Segment', lambda x: (x == 'Individual').mean()) if 'Customer_Segment' in df.columns else ('Order_ID', lambda s: 0.0),
        'Has_Feedback': ('Feedback_Text', lambda x: x.notna().mean()) if 'Feedback_Text' in df.columns else ('Order_ID', lambda s: 0.0),
        'AI_Urgency_High': ('urgency', lambda x: (pd.to_numeric(x, errors="coerce").fillna(0) == 2).sum()) if 'urgency' in df.columns else ('Order_ID', lambda s: 0.0),
        'AI_Action_Voucher': ('action', lambda x: (pd.to_numeric(x, errors="coerce").fillna(0) == 2).sum()) if 'action' in df.columns else ('Order_ID', lambda s: 0.0),
    }

    # Build aggregation dict format
    agg_dict = {k: v for k, v in group_cols.items()}
    feat_df = df.groupby('Customer_ID').agg(**agg_dict).reset_index()

    # Coerce numeric
    for col in feat_df.columns:
        if col != 'Customer_ID':
            feat_df[col] = pd.to_numeric(feat_df[col], errors='coerce')
    return feat_df


def pretty_kpi(label: str, value: float, suffix: str = ""):
    st.metric(label, f"{value}{suffix}")


def plot_auc_curve(y_true: np.ndarray, y_prob: np.ndarray):
    try:
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        auc = roc_auc_score(y_true, y_prob)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name=f'AUC = {auc:.3f}', line=dict(width=3)))
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', name='Chance', line=dict(dash='dash')))
        fig.update_layout(title='ROC Curve', template='plotly_dark', xaxis_title='FPR', yaxis_title='TPR')
        st.plotly_chart(fig, use_container_width=True)
    except Exception:
        st.info("AUC not available (insufficient labels or predictions).")


def compute_shap_summary(model, X: pd.DataFrame):
    shap.initjs()
    try:
        explainer = shap.TreeExplainer(model)
        values = explainer(X)
    except Exception:
        def proba_one(A):
            return model.predict_proba(A)[:, 1]
        explainer = shap.Explainer(proba_one, X, algorithm='permutation')
        values = explainer(X)

    st.subheader("Top Feature Drivers (SHAP)")
    fig, ax = plt.subplots(figsize=(8, 5))
    shap.plots.bar(values, max_display=10, show=False)
    st.pyplot(fig, clear_figure=True, use_container_width=True)

    with st.expander("Detailed impact (Beeswarm)"):
        fig2, ax2 = plt.subplots(figsize=(8, 5))
        try:
            shap.plots.beeswarm(values, max_display=20, show=False)
            st.pyplot(fig2, clear_figure=True, use_container_width=True)
        except Exception:
            st.info("Beeswarm not available with current SHAP backend.")


# --------------- UI ---------------
st.set_page_config(page_title="Churn & Logistics Intelligence", page_icon="📦", layout="wide")
st.markdown(
    """
    <style>
    .hero {
        background: linear-gradient(135deg, #1e3a8a 0%, #3b82f6 100%);
        padding: 32px 36px;
        border-radius: 16px;
        margin-bottom: 24px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .hero h1 {
        color: #ffffff;
        margin: 0;
        font-size: 36px;
        font-weight: 700;
    }
    .hero p {
        color: #e0e7ff;
        margin: 8px 0 0 0;
        font-size: 16px;
    }
    .kpi {
        background: #1e40af;
        border: 1px solid #3b82f6;
        padding: 20px;
        border-radius: 12px;
        color: white;
    }
    .stDataFrame {
        border-radius: 10px;
        overflow: hidden;
    }
    .sidebar .stSidebar {
        background: #0f172a;
    }
    </style>
    <div class="hero">
      <h1>📦 Customer Insight Center</h1>
      <p>AI-powered insights, real-time risk scoring, and operational analytics</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# --------------- Load Data (before sidebar uses it) ---------------
model = load_model_safe(MODEL_PATH)
preds_df = load_csv_safe(PRED_PATH)
high_risk_df = load_csv_safe(HIGH_RISK_PATH)
master_ai = load_csv_safe(MASTER_AI_PATH)

# Rebuild feature frame (for SHAP) if possible
feat_df = build_features_from_master(master_ai) if not master_ai.empty else pd.DataFrame()

# Ensure Customer_ID exists in master data for downstream joins
if not master_ai.empty and 'Customer_ID' not in master_ai.columns and 'Order_ID' in master_ai.columns:
    try:
        tmp_ids = master_ai['Order_ID'].astype(str).str.extract(r'(\d+)')[0].astype(float).astype(int)
        master_ai['Customer_ID'] = 'CUST' + tmp_ids.astype(str).str.zfill(3)
    except Exception:
        pass

# Generate predictions for ALL customers if we have model and feature data
if model is not None and not feat_df.empty and ('Customer_ID' in feat_df.columns):
    # ALWAYS score all customers, don't check preds_df size
    try:
        # Score all customers
        X_all = feat_df.drop(columns=[c for c in ['Customer_ID', 'Churned'] if c in feat_df.columns], errors='ignore')
        all_probs = model.predict_proba(X_all)[:, 1]
        all_ids = feat_df['Customer_ID'].values
        
        # Create comprehensive predictions df
        all_preds = pd.DataFrame({
            'Customer_ID': all_ids,
            'Churn_Prob': all_probs
        })
        all_preds['Risk_Level'] = pd.cut(all_probs, [0, 0.3, 0.7, 1.0], labels=['Low', 'Medium', 'High'], include_lowest=True)
        
        # Use comprehensive predictions (REPLACE the loaded one)
        preds_df = all_preds
    except Exception:
        pass

with st.sidebar:
    st.header("⚙️ Settings")
    risk_cut = st.slider("High-Risk Threshold", 0.5, 0.95, 0.7, 0.01, help="Adjust threshold to filter high-risk customers")
    theme = st.selectbox("Theme", ["Dark", "Light"], index=0, help="Toggle app theme")
    st.markdown("---")
    st.subheader("🔍 Filters")
    # Dynamic filters (apply across tabs)
    seg_opts = sorted(master_ai['Customer_Segment'].dropna().unique().tolist()) if not master_ai.empty and 'Customer_Segment' in master_ai.columns else []
    pri_opts = sorted(master_ai['Priority'].dropna().unique().tolist()) if not master_ai.empty and 'Priority' in master_ai.columns else []
    cat_opts = sorted(master_ai['category'].dropna().unique().tolist()) if not master_ai.empty and 'category' in master_ai.columns else []

    seg_sel = st.multiselect("Customer Segment", seg_opts, default=seg_opts, help="Filter by customer type")
    pri_sel = st.multiselect("Priority Level", pri_opts, default=pri_opts, help="Filter by order priority")
    cat_sel = st.multiselect("AI Feedback Category", cat_opts, default=[], help="Filter by AI-detected issue type")
    st.caption("💡 Filters apply across all tabs")


# --------------- Load Data ---------------
model = load_model_safe(MODEL_PATH)
preds_df = load_csv_safe(PRED_PATH)
high_risk_df = load_csv_safe(HIGH_RISK_PATH)
master_ai = load_csv_safe(MASTER_AI_PATH)

# Rebuild feature frame (for SHAP) if possible
feat_df = build_features_from_master(master_ai) if not master_ai.empty else pd.DataFrame()

# Apply sidebar filters to master_ai
def apply_filters(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    filtered = df.copy()
    if 'Customer_Segment' in filtered.columns and seg_sel:
        filtered = filtered[filtered['Customer_Segment'].isin(seg_sel)]
    if 'Priority' in filtered.columns and pri_sel:
        filtered = filtered[filtered['Priority'].isin(pri_sel)]
    if 'category' in filtered.columns and cat_sel:
        filtered = filtered[filtered['category'].isin(cat_sel)]
    return filtered

master_filtered = apply_filters(master_ai)

# Ensure Customer_ID exists in master_filtered for segments tab
if not master_filtered.empty and 'Customer_ID' not in master_filtered.columns and 'Order_ID' in master_filtered.columns:
    try:
        tmp_ids = master_filtered['Order_ID'].astype(str).str.extract(r'(\d+)')[0].astype(float).astype(int)
        master_filtered['Customer_ID'] = 'CUST' + tmp_ids.astype(str).str.zfill(3)
    except Exception:
        pass

# Optional Light theme override
if 'theme' in locals() and theme == 'Light':
    st.markdown(
        """
        <style>
        .hero {background: linear-gradient(90deg,#f3f4f6,#e5e7eb);} 
        .hero h1 {color: #111827;} 
        .hero p {color: #374151;}
        .kpi {background: #ffffff; border: 1px solid #e5e7eb;}
        </style>
        """,
        unsafe_allow_html=True,
    )


# --------------- KPIs ---------------
col1, col2, col3, col4 = st.columns(4)
if not preds_df.empty:
    col1.metric("📊 Customers Analyzed", f"{len(preds_df):,}")
    avg_risk = preds_df['Churn_Prob'].mean()
    col2.metric("⚠️ Average Retention Risk", f"{avg_risk:.1%}")
    
    # Count high-risk customers based on current threshold
    hr_count = len(preds_df[preds_df['Churn_Prob'] > risk_cut])
    col3.metric("🚨 Customers at Risk", f"{hr_count:,}", 
                delta=f"Threshold: {risk_cut:.0%}", help="Customers above the high-risk threshold")
else:
    col1.metric("📊 Customers Analyzed", "0")
    col2.metric("⚠️ Average Retention Risk", "-")
    col3.metric("🚨 Customers at Risk", "0")

if not master_ai.empty and 'Is_Delayed' in master_ai.columns:
    col4.metric("📦 Delivery Delay Rate", f"{master_ai['Is_Delayed'].mean():.0%}")
else:
    col4.metric("📦 Delivery Delay Rate", "-")


# --------------- Tabs ---------------
tab_dash, tab_ai, tab_ops, tab_segments, tab_cohorts, tab_explain, tab_score = st.tabs([
    "Dashboard",
    "AI Feedback",
    "Operations",
    "Segments",
    "Cohorts",
    "Explainability",
    "Score & Intervene",
])


with tab_dash:
    st.subheader("📊 Customer Retention Risk Overview")
    
    # Show data info
    if not preds_df.empty:
        # Update risk level calculation to use dynamic threshold
        safe_threshold = risk_cut * 0.43  # ~30% when risk_cut is 70%
        moderate_threshold = risk_cut * 0.86  # ~60% when risk_cut is 70%
        
        safe_count = len(preds_df[preds_df['Churn_Prob'] < safe_threshold])
        moderate_count = len(preds_df[(preds_df['Churn_Prob'] >= safe_threshold) & (preds_df['Churn_Prob'] < risk_cut)])
        high_count = len(preds_df[preds_df['Churn_Prob'] >= risk_cut])
        
        st.caption(f"📊 Analyzing {len(preds_df)} customers | "
                  f"🟢 Safe: {safe_count} | "
                  f"🟡 At-Risk: {moderate_count} | "
                  f"🔴 High-Risk: {high_count}")
    
    # Add summary metrics
    if not preds_df.empty and 'Churn_Prob' in preds_df.columns:
        st.write("#### 📈 Risk Distribution Across Customers")
        tmp = preds_df.copy()
        # Join with segment to reflect filters if possible and if Customer_ID exists
        if not master_filtered.empty and 'Customer_ID' in master_filtered.columns:
            join_cols = [c for c in ['Customer_ID','Customer_Segment'] if c in master_filtered.columns]
            seg_map = master_filtered[join_cols].drop_duplicates()
            if 'Customer_ID' in tmp.columns and 'Customer_ID' in seg_map.columns:
                tmp = tmp.merge(seg_map, on='Customer_ID', how='left')
                if seg_sel and 'Customer_Segment' in tmp.columns:
                    tmp = tmp[tmp['Customer_Segment'].isin(seg_sel)]
        
        # Use dynamic thresholds for risk levels
        safe_threshold = risk_cut * 0.43
        moderate_threshold = risk_cut * 0.86
        
        tmp['Risk_Level'] = pd.cut(tmp['Churn_Prob'], 
                                  [0, safe_threshold, moderate_threshold, 1.0], 
                                  labels=['Safe', 'Moderate Risk', 'High Risk'], 
                                  include_lowest=True)
        
        # Summary stats
        col1, col2, col3 = st.columns(3)
        low_risk = len(tmp[tmp['Risk_Level'] == 'Safe'])
        med_risk = len(tmp[tmp['Risk_Level'] == 'Moderate Risk'])
        high_risk_count = len(tmp[tmp['Risk_Level'] == 'High Risk'])
        
        col1.metric("🟢 Safe Customers", low_risk, help="Low risk of leaving")
        col2.metric("🟡 At-Risk Customers", med_risk, help="Moderate retention risk")
        col3.metric("🔴 High-Risk Customers", high_risk_count, help="High probability of leaving")
        
        fig = px.histogram(tmp, x='Churn_Prob', color='Risk_Level', nbins=30, 
                          color_discrete_map={'Safe': '#10b981', 'Moderate Risk': '#f59e0b', 'High Risk': '#ef4444'},
                          template='plotly_dark', 
                          title=f'Customer Retention Risk Distribution (High-Risk Threshold: {risk_cut:.0%})')
        
        # Add vertical line for the threshold
        fig.add_vline(x=risk_cut, line_dash="dash", line_color="red", 
                     annotation_text=f"High-Risk Threshold: {risk_cut:.0%}", 
                     annotation_position="top")
        
        fig.update_layout(margin=dict(l=0, r=0, t=40, b=0), xaxis_title='Retention Risk Probability', yaxis_title='Number of Customers')
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No predictions found. Run the notebook to generate artifacts.")

    st.subheader("🚨 Customers Requiring Immediate Attention")
    if not preds_df.empty and {'Customer_ID','Churn_Prob'}.issubset(preds_df.columns):
        hr_view = preds_df[preds_df['Churn_Prob'] > risk_cut].sort_values('Churn_Prob', ascending=False)
        st.dataframe(hr_view, use_container_width=True, height=360)
        st.download_button("Download High-Risk CSV", hr_view.to_csv(index=False).encode('utf-8'), file_name='high_risk_customers_filtered.csv', mime='text/csv')
    elif not high_risk_df.empty and 'Churn_Prob' in high_risk_df.columns:
        hr_view = high_risk_df[high_risk_df['Churn_Prob'] > risk_cut].sort_values('Churn_Prob', ascending=False)
        st.dataframe(hr_view, use_container_width=True, height=360)
        st.download_button("Download High-Risk CSV", hr_view.to_csv(index=False).encode('utf-8'), file_name='high_risk_customers_filtered.csv', mime='text/csv')
    else:
        st.info("No high-risk data found. Generate predictions in the notebook.")


with tab_ai:
    st.subheader("🔮 AI-Tagged Customer Feedback")
    if not master_filtered.empty and {'Feedback_Text', 'category', 'urgency', 'action'}.issubset(master_filtered.columns):
        # Search box for feedback
        search_text = st.text_input("🔍 Search feedback by keyword", placeholder="Type to search...", key="ai_search")
        
        # Clean display columns
        show = master_filtered[['Order_ID', 'Feedback_Text', 'category', 'urgency', 'action']].dropna(subset=['Feedback_Text'])
        
        # Apply search filter
        if search_text:
            show = show[show['Feedback_Text'].str.contains(search_text, case=False, na=False)]
        
        # Display metrics
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Feedback", len(show))
        col2.metric("AI Categories", show['category'].nunique())
        col3.metric("Urgency (High)", len(show[show['urgency'] == 'High']) if 'urgency' in show.columns else 0)
        col4.metric("Actions Required", len(show[show['action'] != 'None']) if 'action' in show.columns else 0)
        
        # Main table
        st.dataframe(show.head(200), use_container_width=True, height=360)

        # Category distribution
        cat_counts = show['category'].value_counts().reset_index()
        cat_counts.columns = ['category', 'count']
        fig = px.bar(cat_counts, x='category', y='count', template='plotly_dark', title='Feedback Category Distribution')
        st.plotly_chart(fig, use_container_width=True)

        # Actions
        action_counts = show['action'].value_counts().reset_index()
        action_counts.columns = ['action', 'count']
        fig2 = px.bar(action_counts, x='action', y='count', template='plotly_dark', title='Recommended Actions')
        st.plotly_chart(fig2, use_container_width=True)
    else:
        st.info("AI tags not available or filtered out. Ensure notebook Cell 4 completed with API access.")


with tab_ops:
    st.subheader("📊 Operational Insights")
    
    # Add correlation heatmap
    st.write("#### Feature Correlation Matrix")
    numeric_cols = ['Delay_Days', 'Total_Cost', 'Customer_Rating', 'Distance_KM', 'Is_Delayed']
    available_cols = [c for c in numeric_cols if c in master_filtered.columns]
    if len(available_cols) > 1:
        corr_matrix = master_filtered[available_cols].corr()
        fig_corr = px.imshow(corr_matrix, text_auto=True, aspect='auto', color_continuous_scale='RdBu',
                            title='Feature Correlation Heatmap', template='plotly_dark')
        st.plotly_chart(fig_corr, use_container_width=True)
    
    if master_filtered.empty:
        st.info("Data not available.")
    else:
        cols = st.columns(3)
        if 'Is_Delayed' in master_filtered.columns:
            cols[0].metric("Delay Rate", f"{master_filtered['Is_Delayed'].mean():.0%}")
        if 'Total_Cost' in master_filtered.columns:
            cols[1].metric("Avg Order Cost", f"₹{master_filtered['Total_Cost'].mean():.0f}")
        if 'Distance_KM' in master_filtered.columns:
            cols[2].metric("Avg Distance", f"{master_filtered['Distance_KM'].mean():.0f} km")

        c1, c2 = st.columns(2)
        if {'Delay_Days','Total_Cost'}.issubset(master_filtered.columns):
            fig = px.scatter(master_filtered, x='Delay_Days', y='Total_Cost', color='Priority' if 'Priority' in master_filtered.columns else None,
                             template='plotly_dark', title='Delay vs Cost')
            c1.plotly_chart(fig, use_container_width=True)

        if {'Route','Is_Delayed'}.issubset(master_filtered.columns):
            delay_by_route = master_filtered.groupby('Route')['Is_Delayed'].mean().reset_index().sort_values('Is_Delayed', ascending=False).head(15)
            fig2 = px.bar(delay_by_route, x='Route', y='Is_Delayed', template='plotly_dark', title='Top Routes by Delay Rate')
            fig2.update_yaxes(tickformat='.0%')
            c2.plotly_chart(fig2, use_container_width=True)

        if {'Carrier','Is_Delayed'}.issubset(master_filtered.columns):
            st.subheader("Carrier Performance")
            perf = master_filtered.groupby('Carrier').agg(
                Delay_Rate=('Is_Delayed','mean'),
                Avg_Cost=('Total_Cost','mean') if 'Total_Cost' in master_filtered.columns else ('Is_Delayed','size'),
                N=('Order_ID','count')
            ).reset_index().sort_values('Delay_Rate', ascending=False)
            perf['Delay_Rate'] = (perf['Delay_Rate']*100).round(1).astype(str) + '%'
            st.dataframe(perf, use_container_width=True)

        # Problem Areas
        st.subheader("Problem Areas")
        pa_cols = st.columns(2)
        if {'Route','Total_Cost'}.issubset(master_filtered.columns):
            top_cost_routes = master_filtered.groupby('Route')['Total_Cost'].mean().reset_index().sort_values('Total_Cost', ascending=False).head(10)
            pa_cols[0].dataframe(top_cost_routes, use_container_width=True)
        if {'Carrier','Is_Delayed'}.issubset(master_filtered.columns):
            top_delay_carriers = master_filtered.groupby('Carrier')['Is_Delayed'].mean().reset_index().sort_values('Is_Delayed', ascending=False).head(10)
            top_delay_carriers['Is_Delayed'] = (top_delay_carriers['Is_Delayed']*100).round(1).astype(str) + '%'
            pa_cols[1].dataframe(top_delay_carriers, use_container_width=True)


with tab_segments:
    st.subheader("📈 Customer Retention Risk by Segment & Priority")
    
    pred_source = None
    if not preds_df.empty and 'Churn_Prob' in preds_df.columns:
        pred_source = preds_df.copy()
        # Extract Customer_ID if present in Order_ID format
        if 'Customer_ID' not in pred_source.columns and 'Order_ID' in pred_source.columns:
            try:
                tmp_ids = pred_source['Order_ID'].astype(str).str.extract(r'(\d+)')[0].astype(float).astype(int)
                pred_source['Customer_ID'] = 'CUST' + tmp_ids.astype(str).str.zfill(3)
            except Exception:
                pass
    elif not high_risk_df.empty and 'Churn_Prob' in high_risk_df.columns:
        pred_source = high_risk_df.copy()

    # Debug info
    st.caption(f"Debug: preds_df columns: {preds_df.columns.tolist() if not preds_df.empty else 'empty'}, "
              f"master_filtered has Customer_ID: {'Customer_ID' in master_filtered.columns if not master_filtered.empty else 'no data'}")

    if pred_source is None or master_filtered.empty or 'Customer_ID' not in master_filtered.columns or 'Customer_ID' not in pred_source.columns:
        st.warning("⚠️ **Data Setup Required**")
        st.info("""
        To view retention risk analysis by segments, please ensure you have:
        - ✅ Predictions file (run notebook Cell 6)
        - ✅ Master data with customer segment and priority information
        
        Once these files are generated, the analysis will appear here automatically.
        """)
    else:
        seg_cols = [c for c in ['Customer_ID','Customer_Segment','Priority'] if c in master_filtered.columns]
        seg_map = master_filtered[seg_cols].drop_duplicates()
        seg_pred = pred_source.merge(seg_map, on='Customer_ID', how='left') if 'Customer_ID' in pred_source.columns and 'Customer_ID' in seg_map.columns else pred_source.copy()
        if seg_sel:
            seg_pred = seg_pred[seg_pred['Customer_Segment'].isin(seg_sel)]
        if pri_sel:
            seg_pred = seg_pred[seg_pred['Priority'].isin(pri_sel)]

        # Add summary before charts
        st.write(f"**Total customers analyzed:** {len(seg_pred)}")
        
        # Add high-risk summary by segment
        st.write(f"#### High-Risk Customers by Segment (Threshold: {risk_cut:.0%})")
        high_risk_by_segment = seg_pred[seg_pred['Churn_Prob'] > risk_cut].groupby('Customer_Segment').size().reset_index(name='High_Risk_Count')
        if not high_risk_by_segment.empty:
            fig_hr_segment = px.bar(high_risk_by_segment, x='Customer_Segment', y='High_Risk_Count', 
                                   template='plotly_dark', title=f'High-Risk Customers by Segment (Threshold: {risk_cut:.0%})',
                                   color='High_Risk_Count', color_continuous_scale='reds')
            st.plotly_chart(fig_hr_segment, use_container_width=True)
        
        c1, c2 = st.columns(2)
        if 'Customer_Segment' in seg_pred.columns:
            agg = seg_pred.groupby('Customer_Segment')['Churn_Prob'].mean().reset_index()
            agg.columns = ['Segment', 'Retention Risk']
            fig = px.bar(agg, x='Segment', y='Retention Risk', template='plotly_dark', 
                        color='Retention Risk', color_continuous_scale='RdYlGn_r',
                        title='Average Retention Risk by Customer Segment')
            fig.update_yaxes(tickformat='.0%', range=[0, 1])
            c1.plotly_chart(fig, use_container_width=True)
        if 'Priority' in seg_pred.columns:
            agg2 = seg_pred.groupby('Priority')['Churn_Prob'].mean().reset_index()
            agg2.columns = ['Priority Level', 'Retention Risk']
            fig2 = px.bar(agg2, x='Priority Level', y='Retention Risk', template='plotly_dark',
                         color='Retention Risk', color_continuous_scale='RdYlGn_r',
                         title='Average Retention Risk by Priority Level')
            fig2.update_yaxes(tickformat='.0%', range=[0, 1])
            c2.plotly_chart(fig2, use_container_width=True)

        # Insight table: top risky segments
        st.subheader("💡 Retention Risk Insights by Segment")
        if not seg_pred.empty:
            # Use the dynamic threshold for insights
            high_risk_customers = seg_pred[seg_pred['Churn_Prob'] > risk_cut]
            if not high_risk_customers.empty:
                overall = seg_pred['Churn_Prob'].mean()
                by_seg = high_risk_customers.groupby('Customer_Segment')['Churn_Prob'].mean().reset_index()
                by_seg['Risk Multiplier'] = by_seg['Churn_Prob'] / overall
                by_seg['High Risk Count'] = high_risk_customers.groupby('Customer_Segment').size().values
                by_seg.columns = ['Customer Segment', 'Average Risk', 'Risk vs Overall', 'High Risk Count']
                by_seg['Average Risk'] = (by_seg['Average Risk'] * 100).round(1).astype(str) + '%'
                by_seg['Risk vs Overall'] = by_seg['Risk vs Overall'].round(2)
                st.dataframe(by_seg.sort_values('Risk vs Overall', ascending=False), use_container_width=True, height=200)
                st.caption(f"💡 **Interpretation:** Values above 1.0 indicate segments at higher risk than the overall average (High-Risk Threshold: {risk_cut:.0%})")
            else:
                st.info(f"No customers exceed the current high-risk threshold of {risk_cut:.0%}")


with tab_cohorts:
    st.subheader("Monthly Cohorts (Volume)")
    if master_filtered.empty or 'Order_Date' not in master_filtered.columns:
        st.info("Order dates missing.")
    else:
        tmp = master_filtered.copy()
        tmp['Order_Month'] = pd.to_datetime(tmp['Order_Date']).dt.to_period('M')
        cohort = tmp.groupby(['Customer_Segment','Order_Month'])['Order_ID'].nunique().reset_index()
        if cohort.empty:
            st.info("No cohort data available.")
        else:
            pivot = cohort.pivot_table(index='Customer_Segment', columns='Order_Month', values='Order_ID', fill_value=0)
            pivot = pivot.reindex(sorted(pivot.columns), axis=1)
            pivot.columns = pivot.columns.astype(str)
            fig = px.imshow(pivot, text_auto=True, aspect='auto', color_continuous_scale='Blues', title='Orders by Segment x Month')
            st.plotly_chart(fig, use_container_width=True)


with tab_explain:
    st.subheader("Model Performance")
    if not preds_df.empty and 'Churn_Prob' in preds_df.columns and model is not None and not feat_df.empty:
        # Try to rebuild y_true via simple heuristic (if provided later you can load a label file)
        # Here we skip AUC if ground truth not available
        st.caption("Note: Ground-truth labels not included in app artifacts, showing SHAP only.")

        X = feat_df.drop(columns=[c for c in ['Customer_ID', 'Churned'] if c in feat_df.columns], errors='ignore')
        if len(X) > 0:
            compute_shap_summary(model, X)
        else:
            st.info("Not enough features to compute SHAP.")
    else:
        st.info("Model or features missing; cannot compute explainability.")


with tab_score:
    st.subheader("🎯 Individual Customer Risk Assessment")
    if model is None:
        st.info("Model not found. Train in the notebook to create artifacts.")
    else:
        # Use feature frame for selectable IDs
        if not feat_df.empty:
            customer_ids = feat_df['Customer_ID'].unique().tolist()
            cid = st.selectbox("Select Customer ID", customer_ids, help="Choose a customer to assess their retention risk")
            row = feat_df[feat_df['Customer_ID'] == cid]
            if not row.empty:
                X_row = row.drop(columns=[c for c in ['Customer_ID', 'Churned'] if c in row.columns], errors='ignore')
                prob = float(model.predict_proba(X_row)[:, 1][0])
                
                # Display risk level with color - using dynamic threshold
                if prob < risk_cut * 0.43:  # Safe threshold
                    st.metric("🟢 Retention Risk Level", "Safe", f"{prob:.1%}")
                elif prob < risk_cut:  # Moderate threshold
                    st.metric("🟡 Retention Risk Level", "Moderate", f"{prob:.1%}")
                else:  # High risk
                    st.metric("🔴 Retention Risk Level", "High", f"{prob:.1%}")
                
                # Add interpretation with current threshold context
                st.caption(f"**Risk Assessment:** This customer has a **{prob:.1%}** probability of leaving. " +
                          (f"🟢 **Low concern** - Customer appears satisfied (below {risk_cut*0.43:.0%} threshold)." if prob < risk_cut * 0.43 else 
                           f"🟡 **Monitor closely** - Consider retention outreach (below high-risk threshold of {risk_cut:.0%})." if prob < risk_cut else 
                           f"🔴 **Urgent action required** - High priority for retention efforts (above {risk_cut:.0%} threshold)."))

                # Try SHAP for this row
                with st.expander("Why? (SHAP)"):
                    try:
                        shap.initjs()
                        try:
                            expl = shap.TreeExplainer(model)
                            vals = expl(X_row)
                        except Exception:
                            def proba_one(A):
                                return model.predict_proba(A)[:, 1]
                            expl = shap.Explainer(proba_one, X_row, algorithm='permutation')
                            vals = expl(X_row)
                        fig3, ax3 = plt.subplots(figsize=(7, 5))
                        shap.plots.bar(vals, max_display=10, show=False)
                        st.pyplot(fig3, clear_figure=True, use_container_width=True)
                    except Exception:
                        st.info("Per-customer SHAP not available.")

        else:
            st.info("Features unavailable. Run the notebook to export master_enhanced_with_ai.csv.")


st.markdown("---")
st.caption("© 2025 Logistics Intelligence. Built with Streamlit, XGBoost, SHAP, and AI tagging.")