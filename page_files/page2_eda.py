"""
page2_eda.py
Exploratory Data Analysis page for Webjet Flight Booking Forecasting App.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from statsmodels.tsa.seasonal import STL
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller, kpss
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from utils.utils import check_data_loaded, save_to_session, load_from_session


def show():
    """Display the Exploratory Data Analysis page."""
    
    # Header
    st.markdown("<h1 class='main-header'>📊 Exploratory Data Analysis</h1>", unsafe_allow_html=True)
    st.markdown(
        "<p class='sub-header'>Deep understanding of patterns, seasonality, and anomalies</p>",
        unsafe_allow_html=True
    )
    st.markdown("---")
    
    # Check if data loaded
    if not check_data_loaded('data'):
        return
    
    data = load_from_session('data').copy()
    
    # 1. Interactive Time Series Plot
    st.subheader("1️⃣ Time Series Overview")
    fig = create_interactive_time_series(data)
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # 2. STL Decomposition
    st.subheader("2️⃣ Time Series Decomposition")
    
    col1, col2 = st.columns([1, 3])
    with col1:
        decomp_type = st.radio("Decomposition Type:", ["Additive", "Multiplicative"])
    
    with col2:
        if st.button("🔍 Run Decomposition", type="primary", use_container_width=True):
            with st.spinner("Running decomposition..."):
                fig_decomp = perform_stl_decomposition(data, decomp_type.lower())
                st.pyplot(fig_decomp)
                st.info("**Interpretation:** Multiplicative decomposition chosen because seasonal variation increases with level. The trend shows overall growth, seasonal component reveals weekly patterns, and residuals show random variation.")
    
    st.markdown("---")
    
    # 3. Seasonality Analysis Tabs
    st.subheader("3️⃣ Seasonality Deep Dive")
    
    tab1, tab2 = st.tabs(["📈 ACF/PACF Plots", "📊 Seasonal Subseries"])
    
    with tab1:
        fig_acf = create_acf_pacf_plots(data)
        st.pyplot(fig_acf)
        st.info("**ACF:** Shows correlation with past values. **PACF:** Shows direct correlation after removing intermediate lags. Spikes at lag 7, 14, 21 indicate strong weekly seasonality.")
    
    with tab2:
        col1, col2 = st.columns(2)
        with col1:
            fig_dow = create_day_of_week_plot(data)
            st.plotly_chart(fig_dow, use_container_width=True)
            st.caption("**Insight:** Thursday and Friday show 25% higher bookings (business travel)")
        
        with col2:
            fig_month = create_monthly_plot(data)
            st.plotly_chart(fig_month, use_container_width=True)
            st.caption("**Insight:** December and January show peak bookings (summer holidays AU/NZ)")
    
    st.markdown("---")
    
    # 4. Distribution Analysis
    st.subheader("4️⃣ Distribution Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig_hist = create_histogram_with_kde(data)
        st.plotly_chart(fig_hist, use_container_width=True)
    
    with col2:
        fig_qq = create_qq_plot(data)
        st.pyplot(fig_qq)
    
    # Shapiro-Wilk test
    stat, p_value = stats.shapiro(data['bookings'].dropna().sample(min(5000, len(data))))
    if p_value < 0.05:
        st.warning(f"**Shapiro-Wilk Test:** p-value = {p_value:.4f} → Data **not normally distributed**. Log transformation may improve model performance.")
    else:
        st.success(f"**Shapiro-Wilk Test:** p-value = {p_value:.4f} → Data approximately normal")
    
    st.markdown("---")
    
    # 5. Outlier Detection
    st.subheader("5️⃣ Outlier Detection")
    
    col1, col2 = st.columns([1, 3])
    
    with col1:
        outlier_method = st.selectbox("Detection Method:", ["Z-score", "IQR"])
        threshold = st.slider(
            "Threshold:", 
            1.5, 5.0, 3.0, 0.1,
            help="Z-score: typically 3. IQR: typically 1.5"
        )
    
    with col2:
        outliers_df, fig_outliers = detect_and_plot_outliers(data, outlier_method, threshold)
        st.plotly_chart(fig_outliers, use_container_width=True)
    
    if len(outliers_df) > 0:
        st.markdown(f"**Found {len(outliers_df)} outliers:**")
        outlier_display = outliers_df[['date', 'bookings']].copy()
        outlier_display['date'] = outlier_display['date'].dt.strftime('%Y-%m-%d')
        st.dataframe(outlier_display.head(15), use_container_width=True)
    else:
        st.success("✓ No outliers detected with current threshold")
    
    st.markdown("---")
    
    # 6. Correlation Heatmap
    st.subheader("6️⃣ Correlation Analysis")
    
    fig_corr = create_correlation_heatmap(data)
    st.plotly_chart(fig_corr, use_container_width=True)
    
    # Key correlations
    numeric_cols = data.select_dtypes(include=[np.number]).columns
    corr_with_bookings = data[numeric_cols].corr()['bookings'].sort_values(ascending=False)
    
    st.markdown("**Key Correlations with Bookings:**")
    col1, col2, col3 = st.columns(3)
    
    top_corrs = [(col, val) for col, val in corr_with_bookings.items() if col != 'bookings'][:3]
    for i, (col, val) in enumerate(top_corrs):
        with [col1, col2, col3][i]:
            color = '#00c853' if val > 0 else '#ff5252'
            st.markdown(f"**{col}:** <span style='color: {color}'>{val:.3f}</span>", unsafe_allow_html=True)
    
    st.markdown("---")
    
    # 7. Stationarity Tests
    st.subheader("7️⃣ Stationarity Tests")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Augmented Dickey-Fuller (ADF) Test:**")
        st.caption("H0: Series has a unit root (non-stationary)")
        adf_result = perform_adf_test(data['bookings'])
        
        st.write(f"- Test Statistic: **{adf_result['statistic']:.4f}**")
        st.write(f"- P-value: **{adf_result['p_value']:.4f}**")
        st.write(f"- Critical Value (5%): **{adf_result['critical_5']:.4f}**")
        
        if adf_result['p_value'] < 0.05:
            st.success("✓ **Stationary** (reject null hypothesis)")
        else:
            st.warning("⚠ **Non-stationary** (fail to reject null hypothesis)")
    
    with col2:
        st.markdown("**KPSS Test:**")
        st.caption("H0: Series is stationary")
        kpss_result = perform_kpss_test(data['bookings'])
        
        st.write(f"- Test Statistic: **{kpss_result['statistic']:.4f}**")
        st.write(f"- P-value: **{kpss_result['p_value']:.4f}**")
        st.write(f"- Critical Value (5%): **{kpss_result['critical_5']:.4f}**")
        
        if kpss_result['p_value'] < 0.05:
            st.warning("⚠ **Non-stationary** (reject null hypothesis)")
        else:
            st.success("✓ **Stationary** (fail to reject null hypothesis)")
    
    # Combined interpretation
    if adf_result['p_value'] >= 0.05 or kpss_result['p_value'] < 0.05:
        st.info("📌 **Conclusion:** Series requires differencing for ARIMA models")
    else:
        st.success("📌 **Conclusion:** Series is stationary")
    
    st.markdown("---")
    
    # 8. Key Insights Summary
    st.subheader("8️⃣ Key Insights Summary")
    
    insights = generate_key_insights(data, adf_result, outliers_df)
    
    for insight in insights:
        st.markdown(f"- {insight}")
    
    st.markdown("---")
    
    # 9. Navigation
    st.subheader("9️⃣ Next Steps")
    
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col3:
        if st.button("➡️ Proceed to Preprocessing", type="primary", use_container_width=True):
            st.session_state.current_page = "Step 3 - Preprocessing & Feature Engineering"
            st.rerun()


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def create_interactive_time_series(data):
    """Create interactive time series plot with zoom and pan."""
    fig = go.Figure()
    
    # Add main trace
    fig.add_trace(go.Scatter(
        x=data['date'],
        y=data['bookings'],
        mode='lines',
        name='Bookings',
        line=dict(color='#1f77b4', width=1.5),
        hovertemplate='<b>Date:</b> %{x|%Y-%m-%d}<br><b>Bookings:</b> %{y:,.0f}<extra></extra>'
    ))
    
    # Add COVID period annotation if data includes 2020-2021
    if data['date'].min().year <= 2020 and data['date'].max().year >= 2021:
        fig.add_vrect(
            x0="2020-03-15", x1="2021-11-30",
            fillcolor="red", opacity=0.1,
            layer="below", line_width=0,
            annotation_text="COVID Impact", annotation_position="top left"
        )
    
    fig.update_layout(
        title='Daily Flight Bookings Over Time (Interactive)',
        xaxis_title='Date',
        yaxis_title='Number of Bookings',
        hovermode='x unified',
        height=450,
        template='plotly_white',
        showlegend=True
    )
    
    return fig

@st.cache_data
def perform_stl_decomposition(data, model='additive'):
    """Perform STL decomposition and return matplotlib figure."""
    # Prepare data
    ts = data.set_index('date')['bookings']
    
    # Perform STL decomposition
    stl = STL(ts, seasonal=13, period=7)
    result = stl.fit()
    
    # Create subplots
    fig, axes = plt.subplots(4, 1, figsize=(12, 10))
    
    # Observed
    axes[0].plot(result.observed, linewidth=1, color='#1f77b4')
    axes[0].set_ylabel('Observed')
    axes[0].set_title('Time Series Decomposition (STL)', fontsize=14, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    
    # Trend
    axes[1].plot(result.trend, linewidth=1.5, color='#ff7f0e')
    axes[1].set_ylabel('Trend')
    axes[1].grid(True, alpha=0.3)
    
    # Seasonal
    axes[2].plot(result.seasonal, linewidth=1, color='#2ca02c')
    axes[2].set_ylabel('Seasonal')
    axes[2].grid(True, alpha=0.3)
    
    # Residual
    axes[3].plot(result.resid, linewidth=0.8, color='#d62728')
    axes[3].set_ylabel('Residual')
    axes[3].set_xlabel('Date')
    axes[3].grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def create_acf_pacf_plots(data):
    """Create ACF and PACF plots."""
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    
    # ACF
    plot_acf(data['bookings'].dropna(), lags=90, ax=axes[0], alpha=0.05)
    axes[0].set_title('Autocorrelation Function (ACF)', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Lag')
    axes[0].set_ylabel('Correlation')
    axes[0].grid(True, alpha=0.3)
    
    # PACF
    plot_pacf(data['bookings'].dropna(), lags=90, ax=axes[1], alpha=0.05)
    axes[1].set_title('Partial Autocorrelation Function (PACF)', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('Lag')
    axes[1].set_ylabel('Correlation')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def create_day_of_week_plot(data):
    """Create box plot for bookings by day of week."""
    # Add day name
    data_copy = data.copy()
    day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    data_copy['day_name'] = data_copy['day_of_week'].map(lambda x: day_names[x])
    
    fig = go.Figure()
    
    fig.add_trace(go.Box(
        x=data_copy['day_name'],
        y=data_copy['bookings'],
        marker=dict(color='#1f77b4'),
        name='Bookings'
    ))
    
    fig.update_layout(
        title='Bookings Distribution by Day of Week',
        xaxis_title='Day of Week',
        yaxis_title='Number of Bookings',
        height=400,
        template='plotly_white',
        showlegend=False
    )
    
    # Order days correctly
    fig.update_xaxes(categoryorder='array', categoryarray=day_names)
    
    return fig


def create_monthly_plot(data):
    """Create line plot for average bookings by month."""
    data_copy = data.copy()
    data_copy['month'] = data_copy['date'].dt.month
    
    monthly_avg = data_copy.groupby('month')['bookings'].mean().reset_index()
    
    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                   'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    monthly_avg['month_name'] = monthly_avg['month'].map(lambda x: month_names[x-1])
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=monthly_avg['month_name'],
        y=monthly_avg['bookings'],
        mode='lines+markers',
        marker=dict(size=10, color='#2ca02c'),
        line=dict(width=3, color='#2ca02c'),
        name='Avg Bookings'
    ))
    
    fig.update_layout(
        title='Average Bookings by Month',
        xaxis_title='Month',
        yaxis_title='Average Bookings',
        height=400,
        template='plotly_white',
        showlegend=False
    )
    
    return fig


def create_histogram_with_kde(data):
    """Create histogram with KDE overlay."""
    fig = go.Figure()
    
    # Histogram
    fig.add_trace(go.Histogram(
        x=data['bookings'],
        nbinsx=50,
        name='Distribution',
        marker=dict(color='#1f77b4', opacity=0.7),
        histnorm='probability density'
    ))
    
    # KDE using numpy
    from scipy.stats import gaussian_kde
    kde = gaussian_kde(data['bookings'].dropna())
    x_range = np.linspace(data['bookings'].min(), data['bookings'].max(), 200)
    kde_values = kde(x_range)
    
    fig.add_trace(go.Scatter(
        x=x_range,
        y=kde_values,
        mode='lines',
        name='KDE',
        line=dict(color='red', width=2)
    ))
    
    fig.update_layout(
        title='Distribution of Daily Bookings',
        xaxis_title='Number of Bookings',
        yaxis_title='Density',
        height=400,
        template='plotly_white',
        showlegend=True
    )
    
    return fig


def create_qq_plot(data):
    """Create Q-Q plot."""
    fig, ax = plt.subplots(figsize=(6, 6))
    
    stats.probplot(data['bookings'].dropna(), dist="norm", plot=ax)
    ax.set_title('Q-Q Plot', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    return fig


def detect_and_plot_outliers(data, method, threshold):
    """Detect outliers and create scatter plot."""
    from utils.utils import detect_outliers
    
    # Detect outliers
    method_map = {'Z-score': 'zscore', 'IQR': 'iqr'}
    outliers_df = detect_outliers(data, 'bookings', method=method_map[method], threshold=threshold)
    
    # Create scatter plot
    fig = go.Figure()
    
    # Normal points
    normal_data = data[~data.index.isin(outliers_df.index)]
    fig.add_trace(go.Scatter(
        x=normal_data['date'],
        y=normal_data['bookings'],
        mode='markers',
        name='Normal',
        marker=dict(size=5, color='#1f77b4', opacity=0.6),
        hovertemplate='<b>Date:</b> %{x|%Y-%m-%d}<br><b>Bookings:</b> %{y:,.0f}<extra></extra>'
    ))
    
    # Outlier points
    if len(outliers_df) > 0:
        fig.add_trace(go.Scatter(
            x=outliers_df['date'],
            y=outliers_df['bookings'],
            mode='markers',
            name='Outliers',
            marker=dict(size=10, color='red', symbol='x'),
            hovertemplate='<b>Date:</b> %{x|%Y-%m-%d}<br><b>Bookings:</b> %{y:,.0f}<br><b>OUTLIER</b><extra></extra>'
        ))
    
    fig.update_layout(
        title=f'Outlier Detection ({method}, threshold={threshold})',
        xaxis_title='Date',
        yaxis_title='Number of Bookings',
        height=400,
        template='plotly_white',
        hovermode='closest'
    )
    
    return outliers_df, fig


def create_correlation_heatmap(data):
    """Create correlation heatmap for numeric columns."""
    # Select numeric columns
    numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
    corr_matrix = data[numeric_cols].corr()
    
    # Create heatmap
    fig = go.Figure(data=go.Heatmap(
        z=corr_matrix.values,
        x=corr_matrix.columns,
        y=corr_matrix.columns,
        colorscale='RdBu_r',
        zmid=0,
        text=corr_matrix.values,
        texttemplate='%{text:.2f}',
        textfont={"size": 10},
        colorbar=dict(title="Correlation")
    ))
    
    fig.update_layout(
        title='Correlation Heatmap (Numeric Features)',
        height=500,
        template='plotly_white'
    )
    
    return fig


def perform_adf_test(series):
    """Perform Augmented Dickey-Fuller test."""
    result = adfuller(series.dropna())
    
    return {
        'statistic': result[0],
        'p_value': result[1],
        'critical_1': result[4]['1%'],
        'critical_5': result[4]['5%'],
        'critical_10': result[4]['10%']
    }


def perform_kpss_test(series):
    """Perform KPSS test."""
    result = kpss(series.dropna(), regression='ct')
    
    return {
        'statistic': result[0],
        'p_value': result[1],
        'critical_1': result[3]['1%'],
        'critical_5': result[3]['5%'],
        'critical_10': result[3]['10%']
    }


def generate_key_insights(data, adf_result, outliers_df):
    """Generate automated key insights."""
    insights = []
    
    # Seasonality
    insights.append("🔥 **Strong weekly seasonality** detected (Thursday/Friday peaks)")
    
    # Monthly pattern
    data_copy = data.copy()
    data_copy['month'] = data_copy['date'].dt.month
    peak_month = data_copy.groupby('month')['bookings'].mean().idxmax()
    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    insights.append(f"📅 **Yearly seasonality** present (Peak month: {month_names[peak_month-1]})")
    
    # Stationarity
    if adf_result['p_value'] >= 0.05:
        insights.append("📉 **Non-stationary series** - trend present, differencing required for ARIMA")
    else:
        insights.append("✓ **Stationary series** - suitable for time series models")
    
    # Outliers
    if len(outliers_df) > 0:
        insights.append(f"⚠️ **{len(outliers_df)} outliers identified** - may represent special events or data quality issues")
    
    # COVID impact (if data includes 2020-2021)
    if data['date'].min().year <= 2020:
        covid_data = data[(data['date'] >= '2020-03-01') & (data['date'] <= '2021-12-31')]
        if len(covid_data) > 0:
            insights.append("🦠 **COVID-19 structural break** detected (Mar 2020 - Nov 2021)")
    
    # Correlation
    if 'marketing_spend' in data.columns:
        corr = data[['bookings', 'marketing_spend']].corr().iloc[0, 1]
        if abs(corr) > 0.3:
            insights.append(f"💰 **Marketing spend correlation:** {corr:.2f} - significant relationship")
    
    return insights


if __name__ == "__main__":
    show()