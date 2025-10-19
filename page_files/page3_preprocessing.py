"""
page3_preprocessing.py
Data Preprocessing & Feature Engineering page for Webjet Flight Booking Forecasting App.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats
from scipy.special import boxcox
from statsmodels.tsa.stattools import adfuller
from utils.utils import check_data_loaded, save_to_session, load_from_session


def show():
    """Display the Data Preprocessing & Feature Engineering page."""
    
    st.markdown("<h1 class='main-header'>🔧 Data Preprocessing & Feature Engineering</h1>", unsafe_allow_html=True)
    st.markdown("<p class='sub-header'>Transform raw data into model-ready features</p>", unsafe_allow_html=True)
    st.markdown("---")
    
    if not check_data_loaded('data'):
        return
    
    # Initialize session state for processed data
    if 'working_data' not in st.session_state:
        st.session_state.working_data = load_from_session('data').copy()
    
    data = st.session_state.working_data.copy()
    
    # 1. Train/Test Split
    st.subheader("1️⃣ Train/Test Split Configuration")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        test_size = st.slider("Test Set Size (%)", 10, 30, 20, 1) / 100
    
    with col2:
        if st.button("📊 Apply Split", use_container_width=True):
            train_data, test_data, split_date = train_test_split_timeseries(data, test_size)
            save_to_session('train_data', train_data)
            save_to_session('test_data', test_data)
            save_to_session('split_date', split_date)
            st.success(f"✅ Split applied at {split_date.strftime('%Y-%m-%d')}")
    
    # Show split visualization if split exists
    if load_from_session('split_date') is not None:
        fig_split = create_split_timeline(data, load_from_session('split_date'))
        st.plotly_chart(fig_split, use_container_width=True)
        
        train_data = load_from_session('train_data')
        test_data = load_from_session('test_data')
        
        col1, col2 = st.columns(2)
        with col1:
            st.info(f"**Training:** {train_data['date'].min().strftime('%Y-%m-%d')} to {train_data['date'].max().strftime('%Y-%m-%d')} ({len(train_data)} days)")
        with col2:
            st.info(f"**Testing:** {test_data['date'].min().strftime('%Y-%m-%d')} to {test_data['date'].max().strftime('%Y-%m-%d')} ({len(test_data)} days)")
    
    st.markdown("---")
    
    # 2. Feature Creation
    st.subheader("2️⃣ Feature Engineering")
    
    # A. Temporal Features
    with st.expander("📅 A. Temporal Features", expanded=True):
        st.markdown("Create time-based features from the date column:")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            feat_day_of_week = st.checkbox("Day of week (0-6)", value=True)
            feat_month = st.checkbox("Month (1-12)", value=True)
        
        with col2:
            feat_quarter = st.checkbox("Quarter (1-4)", value=True)
            feat_is_weekend = st.checkbox("Is weekend", value=True)
        
        with col3:
            feat_day_of_year = st.checkbox("Day of year (1-365)", value=True)
            feat_week_of_year = st.checkbox("Week of year (1-52)", value=True)
        
        if st.button("➕ Create Temporal Features", use_container_width=True):
            features = {
                'day_of_week': feat_day_of_week,
                'month': feat_month,
                'quarter': feat_quarter,
                'is_weekend': feat_is_weekend,
                'day_of_year': feat_day_of_year,
                'week_of_year': feat_week_of_year
            }
            # ADD THIS CHECK:
            existing_features = [k for k, v in features.items() if v and k in data.columns]
            if existing_features:
                st.warning(f"⚠️ These features already exist: {existing_features}")
                features = {k: v for k, v in features.items() if k not in data.columns}
            
            data = create_temporal_features(data, features)
            st.session_state.working_data = data
            st.success(f"✅ Created {sum(features.values())} temporal features")
            st.rerun()
        
        # Show preview of temporal features if they exist
        temporal_cols = ['month', 'quarter', 'is_weekend', 'day_of_year', 'week_of_year']
        existing_temporal = [col for col in temporal_cols if col in data.columns]
        if existing_temporal:
            st.dataframe(data[['date'] + existing_temporal].head(5), use_container_width=True)
    
    # B. Lag Features
    with st.expander("⏮️ B. Lag Features", expanded=True):
        st.markdown("Create lagged values of the target variable:")
        
        default_lags = [1, 2, 3, 7, 14, 30]
        selected_lags = st.multiselect(
            "Select lag periods:",
            options=[1, 2, 3, 4, 5, 6, 7, 14, 21, 30, 60, 90],
            default=default_lags
        )
        
        if st.button("➕ Create Lag Features", use_container_width=True):
            if selected_lags:
                data = create_lag_features(data, 'bookings', selected_lags)
                st.session_state.working_data = data
                st.success(f"✅ Created {len(selected_lags)} lag features")
                st.warning(f"⚠️ First {max(selected_lags)} rows contain NaN values (will be dropped during modeling)")
                st.rerun()
            else:
                st.error("Please select at least one lag period")
        
        # Show preview of lag features
        lag_cols = [col for col in data.columns if col.startswith('bookings_lag_')]
        if lag_cols:
            preview_cols = ['date', 'bookings'] + sorted(lag_cols, key=lambda x: int(x.split('_')[-1]))[:5]
            st.dataframe(data[preview_cols].head(10), use_container_width=True)
    
    # C. Rolling Statistics
    with st.expander("📊 C. Rolling Statistics", expanded=True):
        st.markdown("Create rolling window statistics:")
        
        col1, col2 = st.columns(2)
        
        with col1:
            windows = st.multiselect(
                "Window sizes (days):",
                options=[3, 7, 14, 21, 30, 60, 90],
                default=[7, 14, 30]
            )
        
        with col2:
            stat_mean = st.checkbox("Mean", value=True)
            stat_std = st.checkbox("Std Dev", value=True)
            stat_min = st.checkbox("Min", value=False)
            stat_max = st.checkbox("Max", value=False)
        
        stats_selected = {
            'mean': stat_mean,
            'std': stat_std,
            'min': stat_min,
            'max': stat_max
        }
        
        if st.button("➕ Create Rolling Features", use_container_width=True):
            if windows and any(stats_selected.values()):
                data = create_rolling_features(data, 'bookings', windows, stats_selected)
                st.session_state.working_data = data
                n_features = len(windows) * sum(stats_selected.values())
                st.success(f"✅ Created {n_features} rolling features")
                st.rerun()
            else:
                st.error("Please select at least one window and one statistic")
        
        # Show preview of rolling features
        rolling_cols = [col for col in data.columns if col.startswith('bookings_rolling_')]
        if rolling_cols:
            preview_cols = ['date', 'bookings'] + sorted(rolling_cols)[:5]
            st.dataframe(data[preview_cols].head(10), use_container_width=True)
    
    # D. Transformations
    with st.expander("🔄 D. Transformations", expanded=True):
        st.markdown("Apply mathematical transformations to the target variable:")
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            transform_type = st.radio(
                "Transformation:",
                options=["None", "Log", "Box-Cox", "Differencing"],
                index=0
            )
        
        with col2:
            if transform_type != "None":
                original_series = data['bookings'].dropna()
                
                if transform_type == "Log":
                    transformed = np.log1p(original_series)
                    transform_col = 'bookings_log'
                elif transform_type == "Box-Cox":
                    transformed, lambda_param = boxcox(original_series + 1)
                    transform_col = 'bookings_boxcox'
                    st.info(f"Box-Cox λ = {lambda_param:.4f}")
                else:  # Differencing
                    transformed = apply_differencing(original_series, order=1)
                    transform_col = 'bookings_diff'
                
                # Create comparison plot
                fig_transform = create_transformation_plot(original_series, transformed, transform_type)
                st.plotly_chart(fig_transform, use_container_width=True)
                
                # Test stationarity on transformed data
                if transform_type in ["Log", "Differencing", "Box-Cox"]:
                    adf_result = adfuller(transformed.dropna())
                    if adf_result[1] < 0.05:
                        st.success(f"✓ After {transform_type}: ADF p-value = {adf_result[1]:.4f} → **Stationary**")
                    else:
                        st.warning(f"⚠ After {transform_type}: ADF p-value = {adf_result[1]:.4f} → Still non-stationary")
                
                if st.button(f"✅ Apply {transform_type} Transformation", use_container_width=True):
                    data[transform_col] = np.nan
                    data.loc[transformed.index, transform_col] = transformed
                    st.session_state.working_data = data
                    st.success(f"✅ {transform_type} transformation applied → '{transform_col}' column created")
                    st.rerun()
    
    st.markdown("---")
    
    # 3. Feature Summary
    st.subheader("3️⃣ Feature Engineering Summary")
    
    feature_summary = create_feature_summary(data)
    
    if len(feature_summary) > 0:
        st.dataframe(
            feature_summary.style.format({
                'missing_count': '{:.0f}',
                'correlation': '{:.3f}'
            }),
            use_container_width=True
        )
        st.info(f"**Total features created:** {len(feature_summary)} (excluding original columns)")
    else:
        st.info("No engineered features created yet. Use the sections above to create features.")
    
    st.markdown("---")
    
    # 4. Handle Missing Values
    st.subheader("4️⃣ Handle Missing Values")
    
    total_missing = data.isnull().sum().sum()
    st.write(f"**Current missing values:** {total_missing:,}")
    
    if total_missing > 0:
        col1, col2 = st.columns([2, 1])
        
        with col1:
            strategy = st.selectbox(
                "Missing value strategy:",
                options=[
                    "Drop rows with any NaN",
                    "Drop rows with NaN in lag/rolling features only",
                    "Forward fill",
                    "Backward fill",
                    "Interpolate"
                ]
            )
        
        with col2:
            if st.button("🔧 Apply Strategy", use_container_width=True):
                data_before = len(data)
                
                if strategy == "Drop rows with any NaN":
                    data = data.dropna()
                elif strategy == "Drop rows with NaN in lag/rolling features only":
                    lag_rolling_cols = [c for c in data.columns if 'lag_' in c or 'rolling_' in c or 'diff' in c]
                    data = data.dropna(subset=lag_rolling_cols)
                elif strategy == "Forward fill":
                    data = data.ffill()
                elif strategy == "Backward fill":
                    data = data.bfill()
                else:  # Interpolate
                    numeric_cols = data.select_dtypes(include=[np.number]).columns
                    data[numeric_cols] = data[numeric_cols].interpolate(method='linear')
                
                data_after = len(data)
                dropped = data_before - data_after
                
                st.session_state.working_data = data
                st.success(f"✅ Strategy applied: {dropped} rows dropped, {data_after:,} rows remain")
                st.rerun()
    else:
        st.success("✓ No missing values to handle")
    
    st.markdown("---")
    
    # 5. Final Preprocessed Data Preview
    st.subheader("5️⃣ Final Preprocessed Data")
    
    st.write(f"**Shape:** {data.shape[0]:,} rows × {data.shape[1]} columns")
    st.dataframe(data.head(10), use_container_width=True)
    
    st.markdown("---")
    
    # 6. Export and Navigation
    st.subheader("6️⃣ Save & Proceed")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        csv = data.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Preprocessed Data",
            data=csv,
            file_name="preprocessed_data.csv",
            mime="text/csv",
            use_container_width=True
        )
    
    with col3:
        if st.button("💾 Save & Proceed to Modeling", type="primary", use_container_width=True):
            save_to_session('preprocessed_data', data)
            save_to_session('working_data', data)
            st.success("✅ Data saved to session!")
            st.session_state.current_page = "Step 4 - Model Training"
            st.rerun()


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def train_test_split_timeseries(df, test_size=0.2):
    """Split time series data into train and test sets."""
    split_index = int(len(df) * (1 - test_size))
    train = df.iloc[:split_index].copy()
    test = df.iloc[split_index:].copy()
    split_date = test['date'].iloc[0]
    
    return train, test, split_date


def create_split_timeline(data, split_date):
    """Create visualization showing train/test split."""
    fig = go.Figure()
    
    train_data = data[data['date'] < split_date]
    test_data = data[data['date'] >= split_date]
    
    fig.add_trace(go.Scatter(
        x=train_data['date'],
        y=train_data['bookings'],
        mode='lines',
        name='Training Set',
        line=dict(color='#00c853', width=2),
        fill='tozeroy',
        fillcolor='rgba(0, 200, 83, 0.1)'
    ))
    
    fig.add_trace(go.Scatter(
        x=test_data['date'],
        y=test_data['bookings'],
        mode='lines',
        name='Test Set',
        line=dict(color='#ff9800', width=2),
        fill='tozeroy',
        fillcolor='rgba(255, 152, 0, 0.1)'
    ))
    
    fig.add_vline(
        x=split_date,
        line_dash="dash",
        line_color="red",
        annotation_text="Split Point",
        annotation_position="top"
    )
    
    fig.update_layout(
        title='Train/Test Split Visualization',
        xaxis_title='Date',
        yaxis_title='Bookings',
        height=350,
        template='plotly_white',
        hovermode='x unified'
    )
    
    return fig


def create_temporal_features(df, features_dict):
    """Create temporal features from date column."""
    df = df.copy()
    
    if features_dict.get('day_of_week') and 'day_of_week' not in df.columns:
        df['day_of_week'] = df['date'].dt.dayofweek
    
    if features_dict.get('month'):
        df['month'] = df['date'].dt.month
    
    if features_dict.get('quarter'):
        df['quarter'] = df['date'].dt.quarter
    
    if features_dict.get('is_weekend'):
        df['is_weekend'] = (df['date'].dt.dayofweek >= 5).astype(int)
    
    if features_dict.get('day_of_year'):
        df['day_of_year'] = df['date'].dt.dayofyear
    
    if features_dict.get('week_of_year'):
        df['week_of_year'] = df['date'].dt.isocalendar().week
    
    return df


def create_lag_features(df, column, lags):
    """Create lagged features."""
    df = df.copy()
    
    for lag in lags:
        df[f'{column}_lag_{lag}'] = df[column].shift(lag)
    
    return df


def create_rolling_features(df, column, windows, stats_dict):
    """Create rolling window statistics."""
    df = df.copy()
    
    for window in windows:
        if stats_dict.get('mean'):
            df[f'{column}_rolling_mean_{window}'] = df[column].rolling(window=window, min_periods=1).mean()
        
        if stats_dict.get('std'):
            df[f'{column}_rolling_std_{window}'] = df[column].rolling(window=window, min_periods=1).std()
        
        if stats_dict.get('min'):
            df[f'{column}_rolling_min_{window}'] = df[column].rolling(window=window, min_periods=1).min()
        
        if stats_dict.get('max'):
            df[f'{column}_rolling_max_{window}'] = df[column].rolling(window=window, min_periods=1).max()
    
    return df


def apply_differencing(series, order=1):
    """Apply differencing to a series."""
    result = series.copy()
    
    for _ in range(order):
        result = result.diff()
    
    return result


def create_transformation_plot(original, transformed, transform_type):
    """Create side-by-side plot of original vs transformed data."""
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Original', f'After {transform_type}')
    )
    
    fig.add_trace(
        go.Histogram(x=original, nbinsx=50, name='Original', marker_color='#1f77b4'),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Histogram(x=transformed, nbinsx=50, name='Transformed', marker_color='#ff7f0e'),
        row=1, col=2
    )
    
    fig.update_layout(
        height=300,
        showlegend=False,
        template='plotly_white'
    )
    
    fig.update_xaxes(title_text="Bookings", row=1, col=1)
    fig.update_xaxes(title_text="Transformed Value", row=1, col=2)
    fig.update_yaxes(title_text="Frequency", row=1, col=1)
    fig.update_yaxes(title_text="Frequency", row=1, col=2)
    
    return fig


def create_feature_summary(df):
    """Create summary table of engineered features."""
    original_cols = ['date', 'bookings', 'day_of_week', 'marketing_spend', 
                     'competitor_price_index', 'is_holiday', 'is_school_holiday', 
                     'weather_disruption_index']
    
    engineered_cols = [col for col in df.columns if col not in original_cols]
    
    if not engineered_cols:
        return pd.DataFrame()
    
    summary_data = []
    
    for col in engineered_cols:
        # Determine feature type
        if 'lag_' in col:
            feat_type = 'Lag'
        elif 'rolling_' in col:
            feat_type = 'Rolling'
        elif col in ['month', 'quarter', 'is_weekend', 'day_of_year', 'week_of_year']:
            feat_type = 'Temporal'
        elif 'log' in col or 'boxcox' in col or 'diff' in col:
            feat_type = 'Transform'
        else:
            feat_type = 'Other'
        
        # Calculate correlation with bookings if numeric
        if df[col].dtype in [np.float64, np.int64]:
            corr = df[['bookings', col]].corr().iloc[0, 1]
        else:
            corr = np.nan
        
        summary_data.append({
            'feature_name': col,
            'type': feat_type,
            'missing_count': df[col].isnull().sum(),
            'correlation': corr
        })
    
    summary_df = pd.DataFrame(summary_data)
    summary_df = summary_df.sort_values('correlation', ascending=False, key=abs)
    
    return summary_df


if __name__ == "__main__":
    show()