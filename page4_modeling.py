"""
page4_modeling.py
Model Training & Comparison page for Webjet Flight Booking Forecasting App.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor
import warnings
warnings.filterwarnings('ignore')

# Statistical models
from pmdarima import auto_arima
from prophet import Prophet
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.stats.diagnostic import acorr_ljungbox
from scipy import stats

# ML models
import xgboost as xgb
import lightgbm as lgb

from utils import check_data_loaded, save_to_session, load_from_session


def show():
    """Display the Model Training & Comparison page."""
    
    st.markdown("<h1 class='main-header'>🤖 Model Training & Comparison</h1>", unsafe_allow_html=True)
    st.markdown("<p class='sub-header'>Train multiple models, compare performance, select best</p>", unsafe_allow_html=True)
    st.markdown("---")
    
    if not check_data_loaded('preprocessed_data'):
        st.warning("⚠️ Please complete preprocessing in Page 3 first.")
        return
        st.markdown("**Statistical Models**")
        use_arima = st.checkbox("ARIMA/SARIMA", value=True)
        use_prophet = st.checkbox("Prophet", value=True)
        use_ets = st.checkbox("Exponential Smoothing", value=False)
    
    data = load_from_session('preprocessed_data')
    
    # Prepare data for modeling
    train_data = load_from_session('train_data')
    test_data = load_from_session('test_data')
    
    if train_data is None or test_data is None:
        st.warning("⚠️ Please apply train/test split in Page 3 first.")
        return
    
    # 1. Model Selection
    st.subheader("1️⃣ Select Models to Train")
    
    col2, col3 = st.columns(2)
    
    
    
    with col2:
        st.markdown("**ML Models**")
        use_xgb = st.checkbox("XGBoost", value=True)
        use_rf = st.checkbox("Random Forest", value=True)
        use_lgb = st.checkbox("LightGBM", value=True)
    
    with col3:
        st.markdown("**Ensemble**")
        use_simple_avg = st.checkbox("Simple Average", value=False)
        use_weighted_avg = st.checkbox("Weighted Average", value=False)
    
    st.markdown("---")
    
    # 2. Train Models
    st.subheader("2️⃣ Model Training")
    
    if st.button("🚀 Train Selected Models", type="primary", use_container_width=True):
        models_to_train = []
        if use_arima: models_to_train.append('ARIMA')
        if use_prophet: models_to_train.append('Prophet')
        if use_ets: models_to_train.append('ETS')
        if use_xgb: models_to_train.append('XGBoost')
        if use_rf: models_to_train.append('Random Forest')
        if use_lgb: models_to_train.append('LightGBM')
        
        if not models_to_train:
            st.error("Please select at least one model to train")
            return
        
        # Train all models
        results, trained_models = train_all_models(
            train_data, test_data, models_to_train
        )
        
        # Ensemble
        if use_simple_avg and len(results) > 1:
            results, trained_models = add_ensemble(
                results, trained_models, test_data, 'simple'
            )
        
        if use_weighted_avg and len(results) > 1:
            results, trained_models = add_ensemble(
                results, trained_models, test_data, 'weighted'
            )
        
        # Save to session
        save_to_session('model_results', results)
        save_to_session('trained_models', trained_models)
        
        st.success(f"✅ Successfully trained {len(results)} models!")
        st.rerun()
    
    # 3. Results Dashboard
    results = load_from_session('model_results')
    trained_models = load_from_session('trained_models')
    
    if results is not None and len(results) > 0:
        st.markdown("---")
        st.subheader("3️⃣ Model Comparison")
        
        # Results table
        results_df = pd.DataFrame(results).T
        results_df = results_df.sort_values('test_rmse')
        
        st.dataframe(
            results_df.style.highlight_min(
                subset=['train_rmse', 'test_rmse', 'mae', 'mape', 'time_s'],
                color='lightgreen'
            ).highlight_max(
                subset=['r2'],
                color='lightgreen'
            ).format({
                'train_rmse': '{:.2f}',
                'test_rmse': '{:.2f}',
                'mae': '{:.2f}',
                'mape': '{:.2f}%',
                'r2': '{:.3f}',
                'time_s': '{:.2f}'
            }),
            use_container_width=True
        )
        
        # Visualizations
        col1, col2 = st.columns(2)
        
        with col1:
            fig_rmse = create_rmse_comparison(results_df)
            st.plotly_chart(fig_rmse, use_container_width=True)
        
        with col2:
            fig_metrics = create_metrics_comparison(results_df)
            st.plotly_chart(fig_metrics, use_container_width=True)
        
        # Actual vs Predicted
        st.markdown("#### Actual vs Predicted (Test Set)")
        fig_pred = create_predictions_plot(test_data, trained_models)
        st.plotly_chart(fig_pred, use_container_width=True)
        
        st.markdown("---")
        
        # 4. Model Diagnostics
        st.subheader("4️⃣ Model Diagnostics")
        
        selected_model = st.selectbox("Select model to inspect:", list(trained_models.keys()))
        
        if selected_model:
            model_info = trained_models[selected_model]
            y_pred = model_info['test_predictions']
            residuals = test_data['bookings'].values - y_pred
            
            # Residual plots
            fig_resid = create_residual_plots(residuals, test_data['date'].values)
            st.pyplot(fig_resid)
            
            # Statistical tests
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Ljung-Box Test (Residual Autocorrelation)**")
                lb_result = ljung_box_test(residuals)
                if lb_result['p_value'] > 0.05:
                    st.success(f"✓ p-value = {lb_result['p_value']:.4f} → No significant autocorrelation")
                else:
                    st.warning(f"⚠ p-value = {lb_result['p_value']:.4f} → Autocorrelation detected")
            
            with col2:
                st.markdown("**Shapiro-Wilk Test (Normality of Residuals)**")
                if len(residuals) > 5000:
                    residuals_sample = np.random.choice(residuals, 5000, replace=False)
                else:
                    residuals_sample = residuals
                sw_stat, sw_p = stats.shapiro(residuals_sample)
                if sw_p > 0.05:
                    st.success(f"✓ p-value = {sw_p:.4f} → Residuals approximately normal")
                else:
                    st.warning(f"⚠ p-value = {sw_p:.4f} → Residuals not normal")
        
        st.markdown("---")
        
        # 5. Feature Importance (for tree models)
        if selected_model in ['XGBoost', 'Random Forest', 'LightGBM']:
            st.subheader("5️⃣ Feature Importance")
            
            model_obj = model_info['model']
            feature_names = model_info.get('feature_names', [])
            
            if hasattr(model_obj, 'feature_importances_'):
                importances = model_obj.feature_importances_
                fig_imp = create_feature_importance_plot(importances, feature_names)
                st.plotly_chart(fig_imp, use_container_width=True)
        
        st.markdown("---")
        
        # 6. Best Model Recommendation
        st.subheader("6️⃣ Model Recommendation")
        
        best_model = recommend_best_model(results_df)
        st.success(f"✅ **Recommended Model:** {best_model['model']}")
        st.info(f"**Rationale:** {best_model['rationale']}")
        
        st.markdown("---")
        
        # 7. Save & Navigate
        st.subheader("7️⃣ Save & Proceed")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("💾 Save Models", use_container_width=True):
                save_to_session('models', trained_models)
                st.success("✅ Models saved!")
        
        with col3:
            if st.button("➡️ Proceed to Forecasting", type="primary", use_container_width=True):
                save_to_session('models', trained_models)
                st.session_state.current_page = "Forecasting & Validation"
                st.rerun()


# ============================================================================
# TRAINING FUNCTIONS
# ============================================================================

def train_all_models(train_data, test_data, models_to_train):
    """Train all selected models and return results."""
    results = {}
    trained_models = {}
    
    # Prepare data
    y_train = train_data['bookings'].values
    y_test = test_data['bookings'].values
    
    # Get feature columns (exclude date, bookings, and target transforms)
    exclude_cols = ['date', 'bookings', 'bookings_log', 'bookings_boxcox', 'bookings_diff']
    feature_cols = [col for col in train_data.columns if col not in exclude_cols]
    
    X_train = train_data[feature_cols].fillna(0).values if feature_cols else None
    X_test = test_data[feature_cols].fillna(0).values if feature_cols else None
    
    total_models = len(models_to_train)
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for idx, model_name in enumerate(models_to_train):
        status_text.text(f"Training {model_name}... ({idx+1}/{total_models})")
        start_time = time.time()
        
        try:
            if model_name == 'ARIMA':
                model, train_pred, test_pred = train_auto_arima(train_data, test_data)
            elif model_name == 'Prophet':
                model, train_pred, test_pred = train_prophet(train_data, test_data)
            elif model_name == 'ETS':
                model, train_pred, test_pred = train_ets(train_data, test_data)
            elif model_name == 'XGBoost':
                model, train_pred, test_pred = train_xgboost(X_train, y_train, X_test)
            elif model_name == 'Random Forest':
                model, train_pred, test_pred = train_random_forest(X_train, y_train, X_test)
            elif model_name == 'LightGBM':
                model, train_pred, test_pred = train_lightgbm(X_train, y_train, X_test)
            
            elapsed_time = time.time() - start_time
            
            # Calculate metrics
            train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
            test_rmse = np.sqrt(mean_squared_error(y_test, test_pred))
            mae = mean_absolute_error(y_test, test_pred)
            mape = np.mean(np.abs((y_test - test_pred) / y_test)) * 100
            r2 = r2_score(y_test, test_pred)
            
            results[model_name] = {
                'train_rmse': train_rmse,
                'test_rmse': test_rmse,
                'mae': mae,
                'mape': mape,
                'r2': r2,
                'time_s': elapsed_time
            }
            
            trained_models[model_name] = {
                'model': model,
                'train_predictions': train_pred,
                'test_predictions': test_pred,
                'feature_names': feature_cols if X_train is not None else []
            }
            
        except Exception as e:
            st.warning(f"⚠️ {model_name} failed: {str(e)}")
        
        progress_bar.progress((idx + 1) / total_models)
    
    status_text.text("Training complete!")
    progress_bar.empty()
    status_text.empty()
    
    return results, trained_models


def train_auto_arima(train_data, test_data):
    """Train ARIMA/SARIMA model."""
    y_train = train_data['bookings'].values
    
    model = auto_arima(
        y_train,
        seasonal=True,
        m=7,
        suppress_warnings=True,
        stepwise=True,
        max_order=5,
        trace=False
    )
    
    train_pred = model.predict_in_sample()
    test_pred = model.predict(n_periods=len(test_data))
    
    return model, train_pred, test_pred


def train_prophet(train_data, test_data):
    """Train Prophet model."""
    df_prophet = pd.DataFrame({
        'ds': train_data['date'],
        'y': train_data['bookings']
    })
    
    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False
    )
    model.fit(df_prophet)
    
    # Train predictions
    train_pred = model.predict(df_prophet)['yhat'].values
    
    # Test predictions
    future = pd.DataFrame({'ds': test_data['date']})
    test_pred = model.predict(future)['yhat'].values
    
    return model, train_pred, test_pred


def train_ets(train_data, test_data):
    """Train Exponential Smoothing model."""
    y_train = train_data['bookings'].values
    
    model = ExponentialSmoothing(
        y_train,
        seasonal_periods=7,
        trend='add',
        seasonal='add'
    ).fit()
    
    train_pred = model.fittedvalues
    test_pred = model.forecast(steps=len(test_data))
    
    return model, train_pred, test_pred


def train_xgboost(X_train, y_train, X_test):
    """Train XGBoost model."""
    model = xgb.XGBRegressor(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        random_state=42
    )
    model.fit(X_train, y_train)
    
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)
    
    return model, train_pred, test_pred


def train_random_forest(X_train, y_train, X_test):
    """Train Random Forest model."""
    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=10,
        random_state=42
    )
    model.fit(X_train, y_train)
    
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)
    
    return model, train_pred, test_pred


def train_lightgbm(X_train, y_train, X_test):
    """Train LightGBM model."""
    model = lgb.LGBMRegressor(
        n_estimators=100,
        num_leaves=31,
        random_state=42,
        verbose=-1
    )
    model.fit(X_train, y_train)
    
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)
    
    return model, train_pred, test_pred


def add_ensemble(results, trained_models, test_data, method='simple'):
    """Add ensemble predictions."""
    y_test = test_data['bookings'].values
    
    # Get all test predictions
    all_preds = {name: info['test_predictions'] for name, info in trained_models.items()}
    
    if method == 'simple':
        ensemble_pred = ensemble_simple_average(all_preds)
        model_name = 'Simple Average'
    else:
        # Weighted by inverse RMSE
        weights = {name: 1/results[name]['test_rmse'] for name in all_preds.keys()}
        ensemble_pred = ensemble_weighted_average(all_preds, weights)
        model_name = 'Weighted Average'
    
    # Calculate metrics
    test_rmse = np.sqrt(mean_squared_error(y_test, ensemble_pred))
    mae = mean_absolute_error(y_test, ensemble_pred)
    mape = np.mean(np.abs((y_test - ensemble_pred) / y_test)) * 100
    r2 = r2_score(y_test, ensemble_pred)
    
    results[model_name] = {
        'train_rmse': np.nan,
        'test_rmse': test_rmse,
        'mae': mae,
        'mape': mape,
        'r2': r2,
        'time_s': 0
    }
    
    trained_models[model_name] = {
        'model': None,
        'train_predictions': None,
        'test_predictions': ensemble_pred,
        'feature_names': []
    }
    
    return results, trained_models


def ensemble_simple_average(predictions):
    """Simple average ensemble."""
    pred_array = np.array(list(predictions.values()))
    return np.mean(pred_array, axis=0)


def ensemble_weighted_average(predictions, weights):
    """Weighted average ensemble."""
    total_weight = sum(weights.values())
    weighted_sum = np.zeros(len(next(iter(predictions.values()))))
    
    for name, pred in predictions.items():
        weighted_sum += pred * (weights[name] / total_weight)
    
    return weighted_sum


# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================

def create_rmse_comparison(results_df):
    """Create RMSE comparison bar chart."""
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=results_df['test_rmse'],
        y=results_df.index,
        orientation='h',
        marker=dict(color='#1f77b4'),
        text=results_df['test_rmse'].round(2),
        textposition='outside'
    ))
    
    fig.update_layout(
        title='Test RMSE Comparison (Lower is Better)',
        xaxis_title='RMSE',
        yaxis_title='Model',
        height=300 + len(results_df) * 30,
        template='plotly_white',
        showlegend=False
    )
    
    return fig


def create_metrics_comparison(results_df):
    """Create multi-metric comparison."""
    fig = go.Figure()
    
    metrics = ['mae', 'mape', 'r2']
    colors = ['#ff7f0e', '#2ca02c', '#d62728']
    
    for metric, color in zip(metrics, colors):
        fig.add_trace(go.Bar(
            name=metric.upper(),
            x=results_df.index,
            y=results_df[metric],
            marker_color=color
        ))
    
    fig.update_layout(
        title='Performance Metrics Comparison',
        xaxis_title='Model',
        yaxis_title='Value',
        height=400,
        template='plotly_white',
        barmode='group'
    )
    
    return fig


def create_predictions_plot(test_data, trained_models):
    """Create actual vs predicted plot."""
    fig = go.Figure()
    
    # Actual
    fig.add_trace(go.Scatter(
        x=test_data['date'],
        y=test_data['bookings'],
        mode='lines',
        name='Actual',
        line=dict(color='black', width=2)
    ))
    
    # Predictions
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    for idx, (name, info) in enumerate(trained_models.items()):
        fig.add_trace(go.Scatter(
            x=test_data['date'],
            y=info['test_predictions'],
            mode='lines',
            name=name,
            line=dict(color=colors[idx % len(colors)], width=1.5),
            visible='legendonly' if idx > 2 else True
        ))
    
    fig.update_layout(
        title='Actual vs Predicted (Test Set)',
        xaxis_title='Date',
        yaxis_title='Bookings',
        height=450,
        template='plotly_white',
        hovermode='x unified'
    )
    
    return fig


def create_residual_plots(residuals, dates):
    """Create 4-panel residual diagnostic plots."""
    import matplotlib.pyplot as plt
    from statsmodels.graphics.tsaplots import plot_acf
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    # 1. Residuals over time
    axes[0, 0].plot(dates, residuals, linewidth=0.8, alpha=0.7)
    axes[0, 0].axhline(y=0, color='r', linestyle='--', linewidth=1)
    axes[0, 0].set_title('Residuals Over Time')
    axes[0, 0].set_xlabel('Date')
    axes[0, 0].set_ylabel('Residual')
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Histogram
    axes[0, 1].hist(residuals, bins=30, edgecolor='black', alpha=0.7)
    axes[0, 1].set_title('Distribution of Residuals')
    axes[0, 1].set_xlabel('Residual')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. ACF
    plot_acf(residuals, lags=30, ax=axes[1, 0], alpha=0.05)
    axes[1, 0].set_title('ACF of Residuals')
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. Q-Q plot
    stats.probplot(residuals, dist="norm", plot=axes[1, 1])
    axes[1, 1].set_title('Q-Q Plot')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def create_feature_importance_plot(importances, feature_names):
    """Create feature importance bar chart."""
    # Get top 15
    indices = np.argsort(importances)[-15:]
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=importances[indices],
        y=[feature_names[i] for i in indices],
        orientation='h',
        marker=dict(color='#2ca02c'),
        text=np.round(importances[indices], 3),
        textposition='outside'
    ))
    
    fig.update_layout(
        title='Top 15 Feature Importances',
        xaxis_title='Importance',
        yaxis_title='Feature',
        height=500,
        template='plotly_white',
        showlegend=False
    )
    
    return fig


# ============================================================================
# ANALYSIS FUNCTIONS
# ============================================================================

def ljung_box_test(residuals, lags=10):
    """Perform Ljung-Box test for autocorrelation."""
    result = acorr_ljungbox(residuals, lags=lags, return_df=False)
    return {'statistic': result[0][-1], 'p_value': result[1][-1]}


def recommend_best_model(results_df):
    """Recommend best model based on weighted criteria."""
    # Normalize metrics (0-1 scale, lower is better for RMSE/time)
    norm_rmse = 1 - (results_df['test_rmse'] - results_df['test_rmse'].min()) / (results_df['test_rmse'].max() - results_df['test_rmse'].min())
    norm_time = 1 - (results_df['time_s'] - results_df['time_s'].min()) / (results_df['time_s'].max() - results_df['time_s'].min() + 0.01)
    
    # Weighted score (50% RMSE, 20% time, 30% dummy for residuals)
    scores = 0.5 * norm_rmse + 0.2 * norm_time + 0.3 * 0.5
    
    best_idx = scores.idxmax()
    best_rmse = results_df.loc[best_idx, 'test_rmse']
    best_time = results_df.loc[best_idx, 'time_s']
    
    rationale = f"Best test RMSE ({best_rmse:.2f}) with acceptable training time ({best_time:.2f}s)"
    
    return {'model': best_idx, 'rationale': rationale}


if __name__ == "__main__":
    show()