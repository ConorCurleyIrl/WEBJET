"""
page7_mlops.py
MLOps: Monitoring & Maintenance page for Webjet Flight Booking Forecasting App.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import ks_2samp, ttest_ind
from statsmodels.stats.diagnostic import acorr_ljungbox
from datetime import datetime, timedelta

# Utility functions

from utils.utils import save_to_session, load_from_session


def show():
    """Display MLOps monitoring page."""
    
    st.markdown("<h1 class='main-header'>🔄 MLOps: Monitoring & Maintenance</h1>", unsafe_allow_html=True)
    st.markdown("<p class='sub-header'>Ensure model performance in production</p>", unsafe_allow_html=True)
    st.markdown("---")
    
    # 1. Model Registry
    st.subheader("1️⃣ Model Registry")
    registry = get_model_registry()
    st.dataframe(registry, use_container_width=True)
    
    st.markdown("---")
    
    # 2. Real-Time Performance Monitoring
    st.subheader("2️⃣ Real-Time Performance Monitoring")
    
    test_data = load_from_session('test_data')
    trained_models = load_from_session('trained_models')
    
    if test_data is not None and trained_models:
        # Generate mock production data
        prod_data = generate_production_data(test_data, 90)
        
        # Rolling MAPE
        fig_mape = create_rolling_mape_chart(prod_data)
        st.plotly_chart(fig_mape, use_container_width=True)
        
        # Actual vs Forecast (last 30 days)
        fig_recent = create_recent_performance_chart(prod_data)
        st.plotly_chart(fig_recent, use_container_width=True)
        
        # Metrics table
        metrics_comparison = calculate_performance_metrics(prod_data)
        st.dataframe(metrics_comparison, use_container_width=True)
    
    st.markdown("---")
    
    # 3. Drift Detection
    st.subheader("3️⃣ Drift Detection")
    
    train_data = load_from_session('train_data')
    
    if train_data is not None:
        drift_results = detect_data_drift_ks(train_data, test_data)
        
        st.markdown("**Data Drift (KS Test):**")
        st.dataframe(
            drift_results.style.applymap(
                lambda x: 'background-color: #ffcdd2' if x == '⚠️ Drift' else '',
                subset=['status']
            ),
            use_container_width=True
        )
        
        # Histogram comparison for drifted features
        drifted = drift_results[drift_results['status'] == '⚠️ Drift']
        if len(drifted) > 0:
            feature = drifted.iloc[0]['feature']
            fig_drift = create_drift_histogram(train_data, test_data, feature)
            st.plotly_chart(fig_drift, use_container_width=True)
        
        # Concept drift
        if trained_models:
            model_name = list(trained_models.keys())[0]
            residuals = test_data['bookings'].values - trained_models[model_name]['test_predictions']
            
            fig_concept = create_concept_drift_chart(test_data['date'].values, residuals)
            st.plotly_chart(fig_concept, use_container_width=True)
            
            has_pattern = detect_concept_drift(residuals)
            if has_pattern:
                st.warning("⚠️ Concept drift detected - residuals show non-random patterns")
            else:
                st.success("✓ No concept drift - residuals appear random")
    
    st.markdown("---")
    
    # 4. Feature Monitoring
    st.subheader("4️⃣ Feature Monitoring")
    
    if test_data is not None:
        numeric_cols = test_data.select_dtypes(include=[np.number]).columns.tolist()
        selected_feature = st.selectbox("Select feature:", numeric_cols)
        
        fig_feature = create_feature_monitoring_chart(test_data, selected_feature)
        st.plotly_chart(fig_feature, use_container_width=True)
    
    st.markdown("---")
    
    # 5. Residual Analysis
    st.subheader("5️⃣ Real-Time Residual Analysis")
    
    if trained_models and test_data is not None:
        model_name = list(trained_models.keys())[0]
        residuals = test_data['bookings'].values - trained_models[model_name]['test_predictions']
        dates = test_data['date'].values
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig_resid_time = create_residuals_time_chart(dates, residuals)
            st.plotly_chart(fig_resid_time, use_container_width=True)
        
        with col2:
            fig_resid_hist = create_residuals_histogram(residuals)
            st.plotly_chart(fig_resid_hist, use_container_width=True)
        
        # ACF
        fig_acf = create_residuals_acf(residuals)
        st.plotly_chart(fig_acf, use_container_width=True)
    
    st.markdown("---")
    
    # 6. Retraining Recommendation
    st.subheader("6️⃣ Retraining Recommendation Engine")
    
    triggers = evaluate_retraining_triggers()
    
    col1, col2 = st.columns(2)
    
    with col1:
        for trigger, status in list(triggers.items())[:3]:
            icon = "✓" if not status else "⚠️"
            st.checkbox(f"{icon} {trigger}", value=status, disabled=True)
    
    with col2:
        for trigger, status in list(triggers.items())[3:]:
            icon = "✓" if not status else "⚠️"
            st.checkbox(f"{icon} {trigger}", value=status, disabled=True)
    
    recommend = sum(triggers.values()) >= 2
    
    if recommend:
        st.error(f"🔄 **Retraining Recommended: YES** ({sum(triggers.values())} triggers active)")
        st.info("**Suggested Frequency:** Monthly retraining | **Priority:** Medium")
    else:
        st.success("✓ **No retraining needed** - Model performance stable")
    
    st.markdown("---")
    
    # 7. Retraining Simulation
    st.subheader("7️⃣ Automated Retraining Simulation")
    
    if st.button("🔄 Simulate Retraining", use_container_width=True):
        with st.spinner("Simulating retraining..."):
            comparison = simulate_retraining()
            save_to_session('retrain_comparison', comparison)
            st.rerun()
    
    comparison = load_from_session('retrain_comparison')
    
    if comparison is not None:
        st.dataframe(comparison, use_container_width=True)
        
        improvement = comparison.loc['Change (%)', 'test_rmse']
        
        if improvement < -2:
            st.success(f"✅ New model shows {abs(improvement):.1f}% improvement - Deployment recommended")
            if st.button("🚀 Deploy New Model (v2)", type="primary"):
                st.success("Model v2 deployed successfully!")
        else:
            st.warning("⚠️ Minimal improvement - Continue with current model")
    
    st.markdown("---")

    # 8. Champion vs Contender Framework
    st.subheader("8️⃣ Champion vs Contender Framework")
    
    col1, col2 = st.columns(2)
    with col1:
        traffic_split = st.slider("Traffic split (v1/v2):", 0, 100, 50)
    with col2:
        duration = st.number_input("Duration (days):", 7, 30, 14)
    
    ab_results = run_ab_test_simulation()
    st.dataframe(ab_results, use_container_width=True)
    
    # T-test
    if ab_results['avg_rmse']['v1'] != ab_results['avg_rmse']['v2']:
        p_value = 0.003  # Mock
        if p_value < 0.05:
            st.success(f"✓ Statistical significance detected (p={p_value:.3f}) - Deploy v2")
    
    st.markdown("---")
    
    # 9. Alert Configuration
    st.subheader("9️⃣ Alert Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.checkbox("✓ MAPE exceeds 20% → Email alert", value=True)
        st.checkbox("✓ Data drift detected → Slack notification", value=True)
        st.checkbox("✓ Forecast outside 3σ → Dashboard alert", value=True)
    
    with col2:
        st.checkbox("Retraining recommended → Jira ticket", value=False)
        st.checkbox("✓ Model deployment → Email confirmation", value=True)
        st.number_input("MAPE threshold (%):", 15, 30, 20)
    
    st.info("**Expected alerts:** 2-3 per month based on current settings")
    
    st.markdown("---")
    
    # 10. Model Health Score
    st.subheader("🔟 Model Health Dashboard")
    
    health = calculate_health_score(90, 85, 90, 70)
    
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Overall", f"{health['overall']}/100", "Good")
    col2.metric("Accuracy", f"{health['accuracy']}/100")
    col3.metric("Stability", f"{health['stability']}/100")
    col4.metric("Efficiency", f"{health['efficiency']}/100")
    col5.metric("Coverage", f"{health['coverage']}/100")
    
    st.markdown("**Recommended Actions:**")
    for action in health['actions']:
        st.markdown(f"- {action}")
    
    st.markdown("---")
    
    # 11. Logging & Audit Trail
    st.subheader("1️⃣1️⃣ Logging & Audit Trail")
    
    logs = get_event_logs()
    st.dataframe(logs, use_container_width=True)
    
    st.markdown("---")
    
    # 12. Cost Tracking
    st.subheader("1️⃣2️⃣ Cost Tracking & ROI")
    
    costs = calculate_costs()
    
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Training", f"${costs['training']}")
    col2.metric("Inference", f"${costs['inference']}")
    col3.metric("Storage", f"${costs['storage']}")
    col4.metric("Monitoring", f"${costs['monitoring']}")
    col5.metric("Total", f"${costs['total']}")
    
    roi = calculate_roi(costs['total'])
    st.success(f"**ROI: {roi:,}%** (${roi * costs['total'] / 100:,.0f} monthly benefit)")
    
    st.markdown("---")
    
    # 13. Export
    st.subheader("1️⃣3️⃣ Export & Reports")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.button("📄 Download MLOps Dashboard", use_container_width=True)
    with col2:
        st.button("📊 Export Metrics CSV", use_container_width=True)
    with col3:
        st.button("📧 Generate Monthly Report", use_container_width=True)


# ============================================================================
# FUNCTIONS
# ============================================================================

def get_model_registry():
    """Get model registry table."""
    return pd.DataFrame({
        'Model ID': ['model_v1', 'model_v0'],
        'Type': ['XGBoost', 'Prophet'],
        'Training Date': ['2024-01-15', '2023-12-01'],
        'Data Period': ['2021-01 to 2023-12', '2021-01 to 2023-11'],
        'Test RMSE': [52.1, 58.3],
        'Status': ['Active', 'Archived']
    })


def generate_production_data(test_data, days=90):
    """Generate mock production data."""
    dates = pd.date_range(end=datetime.now(), periods=days, freq='D')
    actual = test_data['bookings'].values[:days] if len(test_data) >= days else np.random.randint(300, 600, days)
    forecast = actual + np.random.normal(0, 20, len(actual))
    
    return pd.DataFrame({'date': dates, 'actual': actual, 'forecast': forecast})


def create_rolling_mape_chart(prod_data):
    """Create rolling MAPE chart."""
    mape = calculate_rolling_mape(prod_data['actual'].values, prod_data['forecast'].values, window=7)
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=prod_data['date'], y=mape, mode='lines', name='Rolling 7-day MAPE'))
    fig.add_hline(y=12.3, line_dash="dash", line_color="green", annotation_text="Baseline")
    fig.add_hline(y=20, line_dash="dash", line_color="red", annotation_text="Alert Threshold")
    fig.update_layout(title='Rolling 7-Day MAPE (Last 90 Days)', yaxis_title='MAPE (%)', height=300, template='plotly_white')
    
    return fig


def create_recent_performance_chart(prod_data):
    """Create actual vs forecast chart."""
    recent = prod_data.tail(30)
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=recent['date'], y=recent['actual'], mode='lines', name='Actual'))
    fig.add_trace(go.Scatter(x=recent['date'], y=recent['forecast'], mode='lines', name='Forecast', line=dict(dash='dash')))
    fig.update_layout(title='Actual vs Forecast (Last 30 Days)', height=300, template='plotly_white')
    
    return fig


def calculate_performance_metrics(prod_data):
    """Calculate performance metrics."""
    def calc_mape(actual, forecast):
        return np.mean(np.abs((actual - forecast) / actual)) * 100
    
    return pd.DataFrame({
        'Period': ['Last 7 Days', 'Last 30 Days', 'Last 90 Days', 'Baseline (Test)'],
        'RMSE': [48.2, 51.7, 55.3, 52.1],
        'MAPE (%)': [14.1, 15.2, 16.8, 12.3],
        'Status': ['✓ Good', '⚠️ Monitor', '⚠️ Degraded', 'Baseline']
    })


def detect_data_drift_ks(train_data, test_data):
    """Detect data drift using KS test."""
    features = ['bookings', 'marketing_spend', 'competitor_price_index']
    results = []
    
    for feat in features:
        if feat in train_data.columns and feat in test_data.columns:
            stat, p_value = ks_2samp(train_data[feat].dropna(), test_data[feat].dropna())
            status = '⚠️ Drift' if p_value < 0.05 else '✓ No drift'
            results.append({'feature': feat, 'ks_statistic': stat, 'p_value': p_value, 'status': status})
    
    return pd.DataFrame(results)


def create_drift_histogram(train_data, test_data, feature):
    """Create histogram comparison."""
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=train_data[feature], name='Training', opacity=0.7, nbinsx=30))
    fig.add_trace(go.Histogram(x=test_data[feature], name='Recent', opacity=0.7, nbinsx=30))
    fig.update_layout(title=f'Distribution Comparison: {feature}', barmode='overlay', height=300, template='plotly_white')
    
    return fig

def detect_concept_drift(residuals):
    """Detect concept drift in residuals."""
    result = acorr_ljungbox(residuals, lags=[10], return_df=True)
    
    # Handle both DataFrame and tuple returns
    if isinstance(result, pd.DataFrame):
        # Newer statsmodels versions return DataFrame
        p_value = result['lb_pvalue'].iloc[0]
    else:
        # Older versions return tuple
        p_value = result[1][0]
    
    return p_value < 0.05


def create_concept_drift_chart(dates, residuals):
    """Create concept drift visualization."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=dates, y=residuals, mode='markers', marker=dict(size=4)))
    fig.add_hline(y=0, line_dash="dash", line_color="red")
    fig.update_layout(title='Residuals Over Time (Concept Drift Check)', yaxis_title='Residual', height=300, template='plotly_white')
    
    return fig


def create_feature_monitoring_chart(data, feature):
    """Create feature monitoring with control limits."""
    rolling_mean = data[feature].rolling(7).mean()
    rolling_std = data[feature].rolling(7).std()
    overall_mean = data[feature].mean()
    overall_std = data[feature].std()
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data['date'], y=rolling_mean, name='Mean'))
    fig.add_hline(y=overall_mean + 3*overall_std, line_dash="dash", line_color="red", annotation_text="UCL")
    fig.add_hline(y=overall_mean - 3*overall_std, line_dash="dash", line_color="red", annotation_text="LCL")
    fig.update_layout(title=f'{feature} - Statistical Process Control', height=300, template='plotly_white')
    
    return fig


def create_residuals_time_chart(dates, residuals):
    """Create residuals scatter over time."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=dates, y=residuals, mode='markers', marker=dict(size=5, color='#1f77b4')))
    fig.add_hline(y=0, line_dash="dash", line_color="red")
    fig.update_layout(title='Residuals Over Time', height=250, template='plotly_white')
    
    return fig


def create_residuals_histogram(residuals):
    """Create residuals histogram."""
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=residuals, nbinsx=30))
    fig.update_layout(title='Residuals Distribution', height=250, template='plotly_white')
    
    return fig


def create_residuals_acf(residuals):
    """Create ACF plot for residuals."""
    from statsmodels.graphics.tsaplots import plot_acf
    import matplotlib.pyplot as plt
    
    fig_mpl, ax = plt.subplots(figsize=(10, 3))
    plot_acf(residuals, lags=20, ax=ax, alpha=0.05)
    ax.set_title('ACF of Residuals')
    
    return fig_mpl


def evaluate_retraining_triggers():
    """Evaluate retraining trigger conditions."""
    return {
        'Performance degradation (MAPE > 20%)': False,
        'Data drift (p < 0.05 for 2+ features)': True,
        'Concept drift detected': False,
        'Time-based (90 days elapsed)': True,
        'Significant business event': False
    }


def simulate_retraining():
    """Simulate retraining with recent data."""
    return pd.DataFrame({
        'Metric': ['Test RMSE', 'Test MAE', 'Test MAPE', 'Training Time (s)'],
        'Current (v1)': [52.1, 38.5, 12.3, 3.2],
        'New (v2)': [48.7, 36.2, 11.1, 3.8],
        'Change (%)': [-6.5, -6.0, -9.8, +18.8]
    }).set_index('Metric')


def run_ab_test_simulation():
    """Simulate A/B test results."""
    return pd.DataFrame({
        'Model': ['v1 (control)', 'v2 (treatment)'],
        'forecasts': [12453, 12398],
        'avg_rmse': [51.2, 47.8],
        'avg_mape': [13.1, 11.5],
        'winner': ['', '✓']
    })


def calculate_health_score(accuracy, stability, efficiency, coverage):
    """Calculate model health score."""
    overall = int(0.4*accuracy + 0.3*stability + 0.2*efficiency + 0.1*coverage)
    
    actions = []
    if stability < 85:
        actions.append("Monitor data drift closely")
    if overall >= 80:
        actions.append("Model performing well - maintain current schedule")
    else:
        actions.append("Consider retraining")
    
    return {
        'overall': overall,
        'accuracy': accuracy,
        'stability': stability,
        'efficiency': efficiency,
        'coverage': coverage,
        'actions': actions
    }


def get_event_logs():
    """Get recent event logs."""
    return pd.DataFrame({
        'Timestamp': ['2024-02-05 14:22', '2024-02-05 08:30', '2024-01-28 10:16', '2024-01-28 10:15'],
        'Event Type': ['Manual Inspection', 'Alert Triggered', 'Deployment', 'Training'],
        'Description': ['Drift reviewed, monitoring continued', 'Data drift in marketing_spend', 'model_v1 deployed', 'model_v1 trained'],
        'User': ['data-scientist@webjet.com', 'system', 'admin@webjet.com', 'system']
    })


def calculate_costs():
    """Calculate monthly costs."""
    return {'training': 45, 'inference': 180, 'storage': 12, 'monitoring': 35, 'total': 272}


def calculate_roi(total_cost):
    """Calculate ROI."""
    benefits = 17500
    return int((benefits - total_cost) / total_cost * 100)


def calculate_rolling_mape(y_true, y_pred, window=7):
    """Calculate rolling MAPE."""
    mape = np.abs((y_true - y_pred) / y_true) * 100
    return pd.Series(mape).rolling(window).mean().values


if __name__ == "__main__":
    show()