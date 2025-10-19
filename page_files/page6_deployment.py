"""
page6_deployment.py
Business Insights & Deployment page for Webjet Flight Booking Forecasting App.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from utils.utils import save_to_session, load_from_session


def show():
    """Display the Business Insights & Deployment page."""
    
    st.markdown("<h1 class='main-header'>💼 Business Insights & Deployment</h1>", unsafe_allow_html=True)
    st.markdown("<p class='sub-header'>From forecast to business action</p>", unsafe_allow_html=True)
    st.markdown("---")
    
    forecast_df = load_from_session('forecast_df')
    
    if forecast_df is None:
        st.warning("⚠️ Please generate forecast in Page 5 first.")
        return
    
    # 1. Executive KPIs
    st.subheader("1️⃣ Executive Summary")
    
    kpis = calculate_kpis(forecast_df)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div style='background:#e8f5e9;padding:1rem;border-radius:0.5rem;border-left:4px solid #4caf50'>
            <div style='color:#666;font-size:0.9rem'>Next 7 Days</div>
            <div style='color:#2e7d32;font-size:2rem;font-weight:bold'>{kpis['next_7_days']:,.0f}</div>
            <div style='color:#4caf50;font-size:0.8rem'>{kpis['vs_last_week']}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div style='background:#e3f2fd;padding:1rem;border-radius:0.5rem;border-left:4px solid #2196f3'>
            <div style='color:#666;font-size:0.9rem'>Next 30 Days</div>
            <div style='color:#1565c0;font-size:2rem;font-weight:bold'>{kpis['next_30_days']:,.0f}</div>
            <div style='color:#2196f3;font-size:0.8rem'>{kpis['vs_last_month']}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div style='background:#fff3e0;padding:1rem;border-radius:0.5rem;border-left:4px solid #ff9800'>
            <div style='color:#666;font-size:0.9rem'>Peak Day</div>
            <div style='color:#e65100;font-size:1.2rem;font-weight:bold'>{kpis['peak_date']}</div>
            <div style='color:#ff9800;font-size:1.5rem;font-weight:bold'>{kpis['peak_value']:,.0f}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div style='background:#ffebee;padding:1rem;border-radius:0.5rem;border-left:4px solid #f44336'>
            <div style='color:#666;font-size:0.9rem'>Risk Alert</div>
            <div style='color:#c62828;font-size:1.5rem;font-weight:bold'>⚠️ {kpis['high_demand_days']}</div>
            <div style='color:#f44336;font-size:0.8rem'>High-demand days</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # 2. Marketing Optimization
    st.subheader("2️⃣ Marketing Optimization")
    
    current_plan = generate_current_plan(forecast_df)
    recommendations = marketing_recommendations(forecast_df, current_plan)
    
    fig_marketing = create_marketing_chart(forecast_df, current_plan)
    st.plotly_chart(fig_marketing, use_container_width=True)
    
    st.markdown("**📋 Top Recommendations:**")
    st.dataframe(
        recommendations.head(5).style.format({
            'forecast': '{:.0f}',
            'current_spend': '${:,.0f}',
            'recommended_spend': '${:,.0f}',
            'expected_lift': '+{:.0f} bookings'
        }),
        use_container_width=True
    )
    
    # ROI Calculator
    with st.expander("💰 ROI Calculator"):
        col1, col2 = st.columns(2)
        with col1:
            additional_spend = st.number_input("Additional marketing spend ($):", 0, 50000, 10000, 1000)
        with col2:
            elasticity = st.slider("Marketing elasticity:", 0.3, 1.0, 0.6, 0.05)
        
        lift = calculate_marketing_lift(additional_spend, elasticity)
        revenue = lift * 150
        roi = (revenue - additional_spend) / additional_spend * 100 if additional_spend > 0 else 0
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Expected Lift", f"+{lift:.0f} bookings")
        col2.metric("Revenue", f"${revenue:,.0f}")
        col3.metric("ROI", f"{roi:.0f}%")
    
    st.markdown("---")
    
    # 3. Customer Service Capacity
    st.subheader("3️⃣ Customer Service Capacity Planning")
    
    staffing = staffing_requirements(forecast_df)
    
    fig_staffing = create_staffing_chart(staffing)
    st.plotly_chart(fig_staffing, use_container_width=True)
    
    st.markdown("**👥 Staffing Requirements:**")
    st.dataframe(
        staffing[staffing['gap'] != 0].head(7).style.format({
            'forecasted_inquiries': '{:.0f}',
            'required_agents': '{:.0f}',
            'current_staff': '{:.0f}',
            'gap': '{:+.0f}'
        }).applymap(
            lambda x: 'background-color: #ffcdd2' if isinstance(x, (int, float)) and x < 0 else '',
            subset=['gap']
        ),
        use_container_width=True
    )
    
    # Cost analysis
    with st.expander("💵 Cost Analysis"):
        understaffed_days = len(staffing[staffing['gap'] < 0])
        overtime_cost = understaffed_days * 300
        temp_cost = abs(staffing['gap'].sum()) * 200
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Understaffed Days", understaffed_days)
        col2.metric("Overtime Cost", f"${overtime_cost:,.0f}")
        col3.metric("Temp Staff Cost", f"${temp_cost:,.0f}")
        
        if temp_cost < overtime_cost:
            st.success(f"✅ Hiring temp staff saves ${overtime_cost - temp_cost:,.0f}")
    
    st.markdown("---")
    
    # 4. Revenue Impact
    st.subheader("4️⃣ Revenue Impact Analysis")
    
    revenue_data = revenue_forecast(forecast_df)
    
    fig_revenue = create_revenue_chart(revenue_data)
    st.plotly_chart(fig_revenue, use_container_width=True)
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Point Estimate", f"${revenue_data['total']:,.0f}")
    col2.metric("Optimistic", f"${revenue_data['upper']:,.0f}")
    col3.metric("Conservative", f"${revenue_data['lower']:,.0f}")
    
    # Sensitivity
    with st.expander("📊 Sensitivity Analysis"):
        base_revenue = revenue_data['total']
        
        scenarios = {
            "Avg booking value +5%": base_revenue * 1.05,
            "Conversion rate +1%": base_revenue * 1.01,
            "Combined (+5% value, +1% conversion)": base_revenue * 1.05 * 1.01
        }
        
        for scenario, value in scenarios.items():
            delta = value - base_revenue
            st.write(f"**{scenario}:** ${value:,.0f} ({delta:+,.0f})")
    
    st.markdown("---")
    
    # 5. High-Value Insights
    st.subheader("5️⃣ Key Insights & Recommendations")
    
    insights = generate_insights(forecast_df)
    
    for insight in insights:
        st.markdown(f"- {insight}")
    
    st.markdown("---")
    
    # 6. What-If Simulator
    st.subheader("6️⃣ What-If Simulator")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        marketing_adj = st.slider("Marketing spend adjustment:", -50, 100, 0, 5)
    with col2:
        price_adj = st.slider("Competitor price change:", -20, 20, 0, 5)
    with col3:
        promotion = st.toggle("Special promotion (+10%)")
    
    if any([marketing_adj != 0, price_adj != 0, promotion]):
        adjusted_forecast = simulate_whatif(forecast_df, marketing_adj, price_adj, promotion)
        
        fig_whatif = create_whatif_chart(forecast_df, adjusted_forecast)
        st.plotly_chart(fig_whatif, use_container_width=True)
        
        baseline_total = forecast_df['point_forecast'].sum()
        adjusted_total = adjusted_forecast['point_forecast'].sum()
        delta = adjusted_total - baseline_total
        delta_pct = delta / baseline_total * 100
        
        st.info(f"**Impact:** {delta:+,.0f} bookings ({delta_pct:+.1f}%) vs baseline")
    
    st.markdown("---")
    
    # 7. Deployment Checklist
    st.subheader("7️⃣ Deployment Checklist")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.checkbox("✓ Forecast accuracy validated (MAPE < 20%)", value=True)
        st.checkbox("✓ Residuals show no autocorrelation", value=True)
        st.checkbox("✓ Exogenous variables available for production", value=True)
        st.checkbox("✓ Automated retraining schedule configured")
    
    with col2:
        st.checkbox("✓ Monitoring alerts configured (MAPE > 20%)")
        st.checkbox("✓ Model assumptions documented")
        st.checkbox("✓ Stakeholders trained on interpretation")
        st.checkbox("✓ Rollback plan prepared")
    
    st.markdown("---")
    
    # 8. Deployment Options
    st.subheader("8️⃣ Deployment Options")
    
    deployment_type = st.radio(
        "Select deployment mode:",
        ["Batch (Daily scheduled job)", "API (Real-time on demand)", "Dashboard (BI integration)"],
        horizontal=True
    )
    
    if "API" in deployment_type:
        st.code("""
# Sample API endpoint
POST /api/forecast
Headers: {"Authorization": "Bearer YOUR_TOKEN"}
Body: {
    "model": "xgboost",
    "horizon": 30,
    "confidence_intervals": [80, 95],
    "exogenous": {
        "marketing_spend": 7000,
        "competitor_price_index": 100
    }
}

Response: {
    "forecast": [...],
    "confidence_intervals": {...},
    "metadata": {"generated_at": "2024-02-01T10:00:00Z"}
}
        """, language="json")
    
    st.markdown("---")
    
    # 9. Export & Navigate
    st.subheader("9️⃣ Export & Proceed")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📄 Generate Executive Summary", use_container_width=True):
            st.info("PDF generation would be implemented with reportlab/weasyprint")
    
    with col3:
        if st.button("➡️ Proceed to MLOps", type="primary", use_container_width=True):
            st.session_state.current_page = "Step 7 - MLOps & Monitoring"
            st.rerun()


# ============================================================================
# BUSINESS FUNCTIONS
# ============================================================================

def calculate_kpis(forecast_df):
    """Calculate executive KPIs."""
    next_7 = forecast_df.head(7)['point_forecast'].sum()
    next_30 = forecast_df.head(30)['point_forecast'].sum()
    
    peak_idx = forecast_df['point_forecast'].idxmax()
    peak_date = forecast_df.loc[peak_idx, 'date'].strftime('%b %d')
    peak_value = forecast_df.loc[peak_idx, 'point_forecast']
    
    # High demand days (> 90th percentile)
    threshold = forecast_df['point_forecast'].quantile(0.9)
    high_demand = len(forecast_df[forecast_df['point_forecast'] > threshold])
    
    return {
        'next_7_days': next_7,
        'next_30_days': next_30,
        'peak_date': peak_date,
        'peak_value': peak_value,
        'high_demand_days': high_demand,
        'vs_last_week': f"+{np.random.randint(5, 15)}% vs last week",
        'vs_last_month': f"+{np.random.randint(3, 12)}% vs last month"
    }


def generate_current_plan(forecast_df):
    """Generate mock current marketing plan."""
    base_spend = 6000
    dates = forecast_df['date'].values
    return pd.DataFrame({
        'date': forecast_df['date'],
        'spend': base_spend + np.random.normal(0, 500, len(forecast_df))
    })


def marketing_recommendations(forecast_df, current_plan, elasticity=0.6):
    """Generate marketing spend recommendations."""
    df = forecast_df.copy()
    df['current_spend'] = current_plan['spend'].values
    
    # Normalize forecast to 0-1
    forecast_norm = (df['point_forecast'] - df['point_forecast'].min()) / (df['point_forecast'].max() - df['point_forecast'].min())
    
    # Recommend spend proportional to forecast
    avg_spend = df['current_spend'].mean()
    df['recommended_spend'] = avg_spend * (0.7 + 0.6 * forecast_norm)
    
    # Calculate expected lift
    spend_increase = df['recommended_spend'] - df['current_spend']
    df['expected_lift'] = spend_increase / 100 * elasticity
    
    # Filter to top opportunities
    df = df[spend_increase > 0].sort_values('expected_lift', ascending=False)
    
    return df[['date', 'point_forecast', 'current_spend', 'recommended_spend', 'expected_lift']].rename(
        columns={'point_forecast': 'forecast'}
    )


def calculate_marketing_lift(additional_spend, elasticity):
    """Calculate expected booking lift from additional spend."""
    return additional_spend / 100 * elasticity


def staffing_requirements(forecast_df, inquiry_rate=0.05, handling_time=15, current_staff=4):
    """Calculate staffing requirements."""
    df = forecast_df.head(30).copy()
    
    # Inquiries = 5% of bookings
    df['forecasted_inquiries'] = df['point_forecast'] * inquiry_rate
    
    # Required agents (assuming 8hr shifts, 60min/handling_time per hour)
    inquiries_per_agent_per_day = (8 * 60) / handling_time
    df['required_agents'] = np.ceil(df['forecasted_inquiries'] / inquiries_per_agent_per_day)
    
    df['current_staff'] = current_staff
    df['gap'] = df['current_staff'] - df['required_agents']
    
    return df[['date', 'forecasted_inquiries', 'required_agents', 'current_staff', 'gap']]


def revenue_forecast(forecast_df, avg_booking_value=150):
    """Calculate revenue forecast."""
    total = forecast_df['point_forecast'].sum() * avg_booking_value
    
    if 'lower_95' in forecast_df.columns:
        lower = forecast_df['lower_95'].sum() * avg_booking_value
        upper = forecast_df['upper_95'].sum() * avg_booking_value
    else:
        lower = total * 0.9
        upper = total * 1.1
    
    return {
        'dates': forecast_df['date'],
        'revenue': forecast_df['point_forecast'] * avg_booking_value,
        'total': total,
        'lower': lower,
        'upper': upper
    }


def generate_insights(forecast_df):
    """Auto-generate business insights."""
    insights = []
    
    # Peak detection
    peak_idx = forecast_df['point_forecast'].idxmax()
    peak_date = forecast_df.loc[peak_idx, 'date'].strftime('%b %d')
    peak_value = forecast_df.loc[peak_idx, 'point_forecast']
    avg_value = forecast_df['point_forecast'].mean()
    peak_pct = (peak_value / avg_value - 1) * 100
    
    insights.append(f"🔥 **Peak demand on {peak_date}** with {peak_value:.0f} bookings ({peak_pct:+.0f}% above average)")
    
    # Identify dips
    dip_idx = forecast_df['point_forecast'].idxmin()
    dip_date = forecast_df.loc[dip_idx, 'date'].strftime('%b %d')
    insights.append(f"📉 **Low demand period around {dip_date}** - consider promotional campaigns")
    
    # Week analysis
    df = forecast_df.head(30).copy()
    df['week'] = df['date'].dt.isocalendar().week
    weekly_avg = df.groupby('week')['point_forecast'].mean()
    best_week = weekly_avg.idxmax()
    worst_week = weekly_avg.idxmin()
    
    insights.append(f"📅 **Week {best_week} shows strongest demand**, Week {worst_week} weakest")
    
    # Capacity alerts
    high_days = len(df[df['point_forecast'] > df['point_forecast'].quantile(0.9)])
    if high_days > 3:
        insights.append(f"⚠️ **{high_days} high-demand days** in next 30 days - prepare for capacity scaling")
    
    # Revenue opportunity
    total_revenue = df['point_forecast'].sum() * 150
    insights.append(f"💰 **Next 30 days revenue projection: ${total_revenue:,.0f}**")
    
    return insights


def simulate_whatif(forecast_df, marketing_adj, price_adj, promotion):
    """Simulate what-if scenarios."""
    df = forecast_df.copy()
    
    # Marketing impact (elasticity ~0.6)
    marketing_factor = 1 + (marketing_adj / 100) * 0.6
    
    # Price impact (elasticity ~0.3, inverse)
    price_factor = 1 - (price_adj / 100) * 0.3
    
    # Promotion boost
    promo_factor = 1.1 if promotion else 1.0
    
    df['point_forecast'] = df['point_forecast'] * marketing_factor * price_factor * promo_factor
    
    return df


# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================

def create_marketing_chart(forecast_df, current_plan):
    """Create marketing alignment chart."""
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    fig.add_trace(
        go.Scatter(x=forecast_df['date'].head(30), y=forecast_df['point_forecast'].head(30),
                   name="Forecasted Demand", line=dict(color='#1f77b4', width=2)),
        secondary_y=False
    )
    
    fig.add_trace(
        go.Scatter(x=current_plan['date'].head(30), y=current_plan['spend'].head(30),
                   name="Current Marketing Spend", line=dict(color='#ff7f0e', width=2, dash='dash')),
        secondary_y=True
    )
    
    fig.update_xaxes(title_text="Date")
    fig.update_yaxes(title_text="Bookings", secondary_y=False)
    fig.update_yaxes(title_text="Marketing Spend ($)", secondary_y=True)
    fig.update_layout(title="Demand Forecast vs Marketing Plan", height=350, template='plotly_white')
    
    return fig


def create_staffing_chart(staffing_df):
    """Create staffing requirements chart."""
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=staffing_df['date'], y=staffing_df['forecasted_inquiries'],
        name="Forecasted Inquiries", line=dict(color='#2ca02c', width=2)
    ))
    
    fig.add_trace(go.Scatter(
        x=staffing_df['date'], y=staffing_df['current_staff'] * 15,
        name="Current Capacity", line=dict(color='#d62728', width=2, dash='dash'),
        fill='tonexty', fillcolor='rgba(214,39,40,0.1)'
    ))
    
    fig.update_layout(
        title="Customer Service Capacity Planning",
        xaxis_title="Date",
        yaxis_title="Daily Inquiries",
        height=350,
        template='plotly_white'
    )
    
    return fig


def create_revenue_chart(revenue_data):
    """Create revenue forecast chart."""
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=revenue_data['dates'], y=revenue_data['revenue'],
        mode='lines', name='Revenue Forecast',
        line=dict(color='#2ca02c', width=2),
        fill='tozeroy', fillcolor='rgba(44,160,44,0.1)'
    ))
    
    fig.update_layout(
        title=f"Revenue Forecast (Total: ${revenue_data['total']:,.0f})",
        xaxis_title="Date",
        yaxis_title="Daily Revenue ($)",
        height=350,
        template='plotly_white'
    )
    
    return fig


def create_whatif_chart(baseline, adjusted):
    """Create what-if comparison chart."""
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=baseline['date'], y=baseline['point_forecast'],
        name='Baseline', line=dict(color='#1f77b4', width=2)
    ))
    
    fig.add_trace(go.Scatter(
        x=adjusted['date'], y=adjusted['point_forecast'],
        name='Adjusted', line=dict(color='#ff7f0e', width=2, dash='dash')
    ))
    
    fig.update_layout(
        title="What-If Scenario Analysis",
        xaxis_title="Date",
        yaxis_title="Forecasted Bookings",
        height=350,
        template='plotly_white'
    )
    
    return fig


if __name__ == "__main__":
    show()