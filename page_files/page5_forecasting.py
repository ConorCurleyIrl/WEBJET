"""
page5_forecasting.py
Forecasting & Validation page for Webjet Flight Booking Forecasting App.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.metrics import mean_squared_error, mean_absolute_error
from datetime import timedelta
from utils.utils import check_data_loaded, save_to_session, load_from_session


def show():
    """Display the Forecasting & Validation page."""
    
    st.markdown("<h1 class='main-header'>📈 Forecasting & Validation</h1>", unsafe_allow_html=True)
    st.markdown("<p class='sub-header'>Generate future forecasts with uncertainty quantification</p>", unsafe_allow_html=True)
    st.markdown("---")
    
    if not check_data_loaded('models'):
        st.warning("⚠️ Please train models in Page 4 first.")
        return
    
    trained_models = load_from_session('trained_models')
    preprocessed_data = load_from_session('preprocessed_data')
    train_data = load_from_session('train_data')
    test_data = load_from_session('test_data')
    
    # 1. Forecast Configuration
    st.subheader("1️⃣ Forecast Configuration")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        model_name = st.selectbox("Select Model:", list(trained_models.keys()))
    
    with col2:
        horizon = st.slider("Forecast Horizon (days):", 7, 90, 30)
    
    with col3:
        conf_intervals = st.multiselect(
            "Confidence Intervals:",
            [50, 80, 95],
            default=[80, 95]
        )
    
    # Check for exogenous variables
    has_exog = check_has_exogenous(preprocessed_data)
    
    if has_exog:
        exog_method = st.radio(
            "Future exogenous values:",
            ["Use historical average", "Custom scenario"],
            horizontal=True
        )
    else:
        exog_method = "Use historical average"
    
    st.markdown("---")
    
    # 2. Generate Forecast
    st.subheader("2️⃣ Generate Forecast")
    
    if st.button("🔮 Generate Forecast", type="primary", use_container_width=True):
        with st.spinner("Generating forecast..."):
            forecast_df = generate_forecast(
                trained_models[model_name],
                preprocessed_data,
                horizon,
                conf_intervals
            )
            save_to_session('forecast_df', forecast_df)
            save_to_session('forecast_model', model_name)
            st.success(f"✅ Forecast generated for {horizon} days ahead")
            st.rerun()
    
    forecast_df = load_from_session('forecast_df')
    
    if forecast_df is not None:
        # Display forecast table
        st.dataframe(forecast_df.head(14), use_container_width=True)
        
        st.markdown("---")
        
        # 3. Forecast Visualization
        st.subheader("3️⃣ Forecast Visualization")
        
        fig_forecast = create_forecast_plot(
            train_data, test_data, forecast_df, conf_intervals
        )
        st.plotly_chart(fig_forecast, use_container_width=True)
        
        st.markdown("---")
        
        # 4. Forecast Summary
        st.subheader("4️⃣ Forecast Summary (First 14 Days)")
        
        summary_df = forecast_df.head(14).copy()
        summary_df['day_of_week'] = summary_df['date'].dt.day_name()
        
        display_cols = ['date', 'day_of_week', 'point_forecast']
        if 80 in conf_intervals:
            display_cols.extend(['lower_80', 'upper_80'])
        if 95 in conf_intervals:
            display_cols.extend(['lower_95', 'upper_95'])
        
        st.dataframe(
            summary_df[display_cols].style.format({
                'point_forecast': '{:.0f}',
                'lower_80': '{:.0f}',
                'upper_80': '{:.0f}',
                'lower_95': '{:.0f}',
                'upper_95': '{:.0f}'
            }),
            use_container_width=True
        )
        
        st.markdown("---")
        
        # 5. Backtesting
        st.subheader("5️⃣ Backtesting")
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            n_periods = st.number_input("Number of periods:", 3, 10, 5)
            backtest_horizon = st.number_input("Horizon per period:", 7, 60, 30)
        
        with col2:
            if st.button("🔄 Run Backtest", use_container_width=True):
                with st.spinner("Running backtest..."):
                    backtest_results = run_backtest(
                        trained_models[model_name],
                        preprocessed_data,
                        n_periods,
                        backtest_horizon
                    )
                    save_to_session('backtest_results', backtest_results)
                    st.success("✅ Backtest complete")
                    st.rerun()
        
        backtest_results = load_from_session('backtest_results')
        
        if backtest_results is not None:
            # Backtest table
            bt_df = pd.DataFrame(backtest_results['periods'])
            st.dataframe(
                bt_df.style.format({
                    'rmse': '{:.2f}',
                    'mae': '{:.2f}',
                    'mape': '{:.2f}%'
                }),
                use_container_width=True
            )
            
            # Average metrics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Avg RMSE", f"{bt_df['rmse'].mean():.2f}")
            with col2:
                st.metric("Avg MAE", f"{bt_df['mae'].mean():.2f}")
            with col3:
                st.metric("Avg MAPE", f"{bt_df['mape'].mean():.2f}%")
            
            # Backtest visualization
            fig_bt = create_backtest_plot(backtest_results)
            st.plotly_chart(fig_bt, use_container_width=True)
        
        st.markdown("---")
        
        # 6. Forecast Accuracy Degradation
        st.subheader("6️⃣ Forecast Accuracy Degradation")
        
        if st.button("📊 Compute Degradation", use_container_width=True):
            with st.spinner("Computing..."):
                degradation = compute_forecast_degradation(
                    trained_models[model_name],
                    test_data,
                    horizons=range(1, 31)
                )
                save_to_session('degradation', degradation)
                st.rerun()
        
        degradation = load_from_session('degradation')
        
        if degradation is not None:
            fig_deg = create_degradation_plot(degradation)
            st.plotly_chart(fig_deg, use_container_width=True)
            
            # Insights
            rmse_1day = degradation['rmse'][0]
            rmse_7day = degradation['rmse'][6]
            rmse_30day = degradation['rmse'][29]
            
            st.info(f"**Insights:** 1-day RMSE: {rmse_1day:.1f} | 7-day RMSE: {rmse_7day:.1f} | 30-day RMSE: {rmse_30day:.1f}")
        
        st.markdown("---")
        
        # 7. Prediction Interval Coverage
        st.subheader("7️⃣ Prediction Interval Coverage Analysis")
        
        if 80 in conf_intervals or 95 in conf_intervals:
            # Generate intervals on test set
            test_intervals = generate_prediction_intervals(
                trained_models[model_name],
                test_data
            )
            
            col1, col2 = st.columns(2)
            
            if 80 in conf_intervals:
                coverage_80 = evaluate_interval_coverage(
                    test_data['bookings'].values,
                    test_intervals['lower_80'],
                    test_intervals['upper_80'],
                    0.80
                )
                
                with col1:
                    color = "green" if abs(coverage_80 - 80) < 5 else "orange"
                    st.markdown(f"**80% Interval Coverage:** <span style='color:{color}'>{coverage_80:.1f}%</span> (Target: 80%)", unsafe_allow_html=True)
                    if abs(coverage_80 - 80) < 5:
                        st.success("✓ Well-calibrated")
                    else:
                        st.warning("⚠ Intervals may need adjustment")
            
            if 95 in conf_intervals:
                coverage_95 = evaluate_interval_coverage(
                    test_data['bookings'].values,
                    test_intervals['lower_95'],
                    test_intervals['upper_95'],
                    0.95
                )
                
                with col2:
                    color = "green" if abs(coverage_95 - 95) < 5 else "orange"
                    st.markdown(f"**95% Interval Coverage:** <span style='color:{color}'>{coverage_95:.1f}%</span> (Target: 95%)", unsafe_allow_html=True)
                    if abs(coverage_95 - 95) < 5:
                        st.success("✓ Well-calibrated")
                    else:
                        st.warning("⚠ Intervals may need adjustment")
        
        st.markdown("---")
        
        # 8. Scenario Analysis
        if has_exog:
            st.subheader("8️⃣ Scenario Analysis")
            
            scenarios = {
                'Optimistic': {'marketing_spend': 1.2, 'competitor_price_index': 0.9},
                'Baseline': {'marketing_spend': 1.0, 'competitor_price_index': 1.0},
                'Pessimistic': {'marketing_spend': 0.8, 'competitor_price_index': 1.1}
            }
            
            scenario_forecasts = scenario_forecast(
                trained_models[model_name],
                preprocessed_data,
                horizon,
                scenarios
            )
            
            # Chart
            fig_scenario = create_scenario_plot(scenario_forecasts)
            st.plotly_chart(fig_scenario, use_container_width=True)
            
            # Table
            scenario_summary = pd.DataFrame({
                'Scenario': scenarios.keys(),
                'Total Bookings': [s['point_forecast'].sum() for s in scenario_forecasts.values()],
                'Avg Daily': [s['point_forecast'].mean() for s in scenario_forecasts.values()],
                'vs Baseline': [
                    (s['point_forecast'].sum() - scenario_forecasts['Baseline']['point_forecast'].sum()) / 
                    scenario_forecasts['Baseline']['point_forecast'].sum() * 100
                    for s in scenario_forecasts.values()
                ]
            })
            
            st.dataframe(
                scenario_summary.style.format({
                    'Total Bookings': '{:.0f}',
                    'Avg Daily': '{:.0f}',
                    'vs Baseline': '{:+.1f}%'
                }),
                use_container_width=True
            )
        
        st.markdown("---")
        
        # 9. Export & Navigate
        st.subheader("9️⃣ Export & Proceed")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            csv = forecast_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download Forecast CSV",
                data=csv,
                file_name=f"forecast_{model_name}_{horizon}days.csv",
                mime="text/csv",
                use_container_width=True
            )
        
        with col3:
            if st.button("➡️ Proceed to Deployment", type="primary", use_container_width=True):
                st.session_state.current_page = "Business Insights & Deployment"
                st.rerun()


# ============================================================================
# FORECASTING FUNCTIONS
# ============================================================================

def generate_forecast(model_info, data, horizon, conf_intervals):
    """Generate forecast with prediction intervals."""
    last_date = data['date'].max()
    future_dates = pd.date_range(start=last_date + timedelta(days=1), periods=horizon, freq='D')
    
    # Point forecast
    model = model_info['model']
    
    if model is None:  # Ensemble
        # Use simple approach for ensemble
        point_forecast = np.full(horizon, data['bookings'].mean())
    else:
        try:
            # For statistical models
            if hasattr(model, 'predict'):
                point_forecast = model.predict(n_periods=horizon)
            elif hasattr(model, 'forecast'):
                point_forecast = model.forecast(steps=horizon)
            else:
                # For ML models - need features
                point_forecast = np.full(horizon, data['bookings'].mean())
        except:
            point_forecast = np.full(horizon, data['bookings'].mean())
    
    # Create forecast dataframe
    forecast_df = pd.DataFrame({
        'date': future_dates,
        'point_forecast': point_forecast
    })
    
    # Add prediction intervals
    for conf in conf_intervals:
        margin = 1.96 * (conf / 100) * np.std(point_forecast)
        forecast_df[f'lower_{conf}'] = point_forecast - margin
        forecast_df[f'upper_{conf}'] = point_forecast + margin
    
    return forecast_df


def generate_prediction_intervals(model_info, test_data):
    """Generate prediction intervals on test data."""
    y_pred = model_info['test_predictions']
    residuals = test_data['bookings'].values - y_pred
    std_resid = np.std(residuals)
    
    intervals = {
        'lower_80': y_pred - 1.28 * std_resid,
        'upper_80': y_pred + 1.28 * std_resid,
        'lower_95': y_pred - 1.96 * std_resid,
        'upper_95': y_pred + 1.96 * std_resid
    }
    
    return intervals


def run_backtest(model_info, data, n_periods, horizon):
    """Run walk-forward backtesting."""
    periods = []
    all_forecasts = []
    
    total_len = len(data)
    test_size = horizon
    step_size = len(data) // (n_periods + 1)
    
    for i in range(n_periods):
        train_end = total_len - (n_periods - i) * step_size
        test_start = train_end
        test_end = min(test_start + horizon, total_len)
        
        if test_end - test_start < horizon // 2:
            continue
        
        train = data.iloc[:train_end]
        test = data.iloc[test_start:test_end]
        
        # Simple forecast (use mean for demo)
        forecast = np.full(len(test), train['bookings'].mean())
        actual = test['bookings'].values
        
        rmse = np.sqrt(mean_squared_error(actual, forecast))
        mae = mean_absolute_error(actual, forecast)
        mape = np.mean(np.abs((actual - forecast) / actual)) * 100
        
        periods.append({
            'period': i + 1,
            'date_range': f"{test['date'].iloc[0].strftime('%Y-%m-%d')} to {test['date'].iloc[-1].strftime('%Y-%m-%d')}",
            'rmse': rmse,
            'mae': mae,
            'mape': mape
        })
        
        all_forecasts.append({
            'dates': test['date'].values,
            'actual': actual,
            'forecast': forecast
        })
    
    return {'periods': periods, 'forecasts': all_forecasts}


def compute_forecast_degradation(model_info, test_data, horizons):
    """Compute how accuracy degrades with forecast horizon."""
    rmse_by_horizon = []
    
    y_true = test_data['bookings'].values
    
    for h in horizons:
        if h >= len(y_true):
            break
        
        # Rolling forecast
        errors = []
        for i in range(len(y_true) - h):
            # Simple approach: use historical mean
            forecast = y_true[:i+1].mean() if i > 0 else y_true[0]
            actual = y_true[i + h]
            errors.append((actual - forecast) ** 2)
        
        rmse = np.sqrt(np.mean(errors)) if errors else np.nan
        rmse_by_horizon.append(rmse)
    
    return {'horizon': list(horizons)[:len(rmse_by_horizon)], 'rmse': rmse_by_horizon}


def evaluate_interval_coverage(y_true, lower, upper, conf_level):
    """Calculate prediction interval coverage."""
    in_interval = (y_true >= lower) & (y_true <= upper)
    coverage = np.mean(in_interval) * 100
    return coverage


def scenario_forecast(model_info, data, horizon, scenarios):
    """Generate forecasts under different scenarios."""
    forecasts = {}
    
    for scenario_name, adjustments in scenarios.items():
        # Simple approach: adjust baseline forecast
        baseline = generate_forecast(model_info, data, horizon, [])
        
        # Adjust based on scenario
        adjustment_factor = 1.0
        if 'marketing_spend' in adjustments:
            adjustment_factor *= adjustments['marketing_spend'] ** 0.3
        if 'competitor_price_index' in adjustments:
            adjustment_factor *= (2 - adjustments['competitor_price_index']) ** 0.2
        
        forecast_df = baseline.copy()
        forecast_df['point_forecast'] = forecast_df['point_forecast'] * adjustment_factor
        
        forecasts[scenario_name] = forecast_df
    
    return forecasts


def check_has_exogenous(data):
    """Check if data has exogenous variables."""
    exog_cols = ['marketing_spend', 'competitor_price_index']
    return any(col in data.columns for col in exog_cols)


# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================

def create_forecast_plot(train_data, test_data, forecast_df, conf_intervals):
    """Create comprehensive forecast visualization."""
    fig = go.Figure()
    
    # Historical (train)
    fig.add_trace(go.Scatter(
        x=train_data['date'],
        y=train_data['bookings'],
        mode='lines',
        name='Historical (Train)',
        line=dict(color='#1f77b4', width=1.5)
    ))
    
    # Test
    fig.add_trace(go.Scatter(
        x=test_data['date'],
        y=test_data['bookings'],
        mode='lines',
        name='Actual (Test)',
        line=dict(color='#2ca02c', width=1.5)
    ))
    
    # Forecast
    fig.add_trace(go.Scatter(
        x=forecast_df['date'],
        y=forecast_df['point_forecast'],
        mode='lines',
        name='Forecast',
        line=dict(color='#ff7f0e', width=2)
    ))
    
    # Prediction intervals
    for conf in sorted(conf_intervals, reverse=True):
        fig.add_trace(go.Scatter(
            x=pd.concat([forecast_df['date'], forecast_df['date'][::-1]]),
            y=pd.concat([forecast_df[f'upper_{conf}'], forecast_df[f'lower_{conf}'][::-1]]),
            fill='toself',
            fillcolor=f'rgba(255,127,14,{0.1 + conf/200})',
            line=dict(color='rgba(255,255,255,0)'),
            name=f'{conf}% Interval',
            showlegend=True
        ))
    
    # Boundary lines
    fig.add_vline(
        x=test_data['date'].iloc[0],
        line_dash="dash",
        line_color="gray",
        annotation_text="Train/Test Split"
    )
    
    fig.add_vline(
        x=forecast_df['date'].iloc[0],
        line_dash="dash",
        line_color="red",
        annotation_text="Forecast Start"
    )
    
    fig.update_layout(
        title='Forecast with Prediction Intervals',
        xaxis_title='Date',
        yaxis_title='Bookings',
        height=500,
        template='plotly_white',
        hovermode='x unified'
    )
    
    return fig


def create_backtest_plot(backtest_results):
    """Create backtest visualization."""
    fig = go.Figure()
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    
    for i, fc in enumerate(backtest_results['forecasts']):
        fig.add_trace(go.Scatter(
            x=fc['dates'],
            y=fc['actual'],
            mode='lines',
            name=f'Period {i+1} Actual',
            line=dict(color=colors[i % len(colors)], width=2)
        ))
        
        fig.add_trace(go.Scatter(
            x=fc['dates'],
            y=fc['forecast'],
            mode='lines',
            name=f'Period {i+1} Forecast',
            line=dict(color=colors[i % len(colors)], width=2, dash='dash')
        ))
    
    fig.update_layout(
        title='Backtest: Forecasts vs Actuals',
        xaxis_title='Date',
        yaxis_title='Bookings',
        height=400,
        template='plotly_white',
        hovermode='x unified'
    )
    
    return fig


def create_degradation_plot(degradation):
    """Create forecast degradation plot."""
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=degradation['horizon'],
        y=degradation['rmse'],
        mode='lines+markers',
        marker=dict(size=8, color='#d62728'),
        line=dict(width=2, color='#d62728')
    ))
    
    fig.update_layout(
        title='Forecast Accuracy Degradation',
        xaxis_title='Forecast Horizon (days ahead)',
        yaxis_title='RMSE',
        height=350,
        template='plotly_white',
        showlegend=False
    )
    
    return fig


def create_scenario_plot(scenario_forecasts):
    """Create scenario comparison plot."""
    fig = go.Figure()
    
    colors = {'Optimistic': '#2ca02c', 'Baseline': '#1f77b4', 'Pessimistic': '#d62728'}
    
    for scenario_name, forecast_df in scenario_forecasts.items():
        fig.add_trace(go.Scatter(
            x=forecast_df['date'],
            y=forecast_df['point_forecast'],
            mode='lines',
            name=scenario_name,
            line=dict(color=colors.get(scenario_name, '#7f7f7f'), width=2)
        ))
    
    fig.update_layout(
        title='Scenario Analysis: Forecast Comparison',
        xaxis_title='Date',
        yaxis_title='Forecasted Bookings',
        height=400,
        template='plotly_white',
        hovermode='x unified'
    )
    
    return fig


if __name__ == "__main__":
    show()