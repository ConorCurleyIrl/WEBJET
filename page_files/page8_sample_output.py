"""
page8_sample_output.py
Sample Output - Executive Business Dashboard for Webjet Flight Booking Forecasting
Audience: Senior Management & Marketing Leadership
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from datetime import datetime, timedelta
import time
from utils.utils import save_to_session, load_from_session


def show():
    """Display the most engaging Sample Output business dashboard."""
    
    # Dynamic CSS with animations and premium styling
    st.markdown("""
    <style>
    @keyframes pulse {
        0% { transform: scale(1); }
        50% { transform: scale(1.02); }
        100% { transform: scale(1); }
    }
    
    @keyframes slideInUp {
        from { transform: translateY(30px); opacity: 0; }
        to { transform: translateY(0); opacity: 1; }
    }
    
    @keyframes countUp {
        from { opacity: 0; transform: scale(0.8); }
        to { opacity: 1; transform: scale(1); }
    }
    
    .hero-container {
        background: linear-gradient(135deg, #FF6600 0%, #FF8C42 50%, #FFB84D 100%);
        padding: 3rem 2rem;
        border-radius: 20px;
        margin-bottom: 2rem;
        text-align: center;
        box-shadow: 0 20px 40px rgba(255, 102, 0, 0.3);
        position: relative;
        overflow: hidden;
    }
    
    .hero-container::before {
        content: '';
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: repeating-linear-gradient(
            45deg,
            transparent,
            transparent 10px,
            rgba(255, 255, 255, 0.05) 10px,
            rgba(255, 255, 255, 0.05) 20px
        );
        animation: pulse 4s ease-in-out infinite;
    }
    
    .hero-title {
        color: white;
        font-size: 3.5rem;
        margin: 0;
        font-weight: 900;
        text-shadow: 3px 3px 6px rgba(0,0,0,0.3);
        position: relative;
        z-index: 2;
    }
    
    .hero-subtitle {
        color: white;
        font-size: 1.4rem;
        margin: 1rem 0 0 0;
        opacity: 0.95;
        position: relative;
        z-index: 2;
    }
    
    .hero-impact {
        color: #FFD700;
        font-size: 1.1rem;
        margin: 0.5rem 0 0 0;
        font-weight: bold;
        position: relative;
        z-index: 2;
    }
    
    .mega-metric {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        padding: 3rem 2rem;
        border-radius: 25px;
        color: white;
        text-align: center;
        margin: 2rem 0;
        box-shadow: 0 25px 50px rgba(30, 60, 114, 0.4);
        animation: slideInUp 1s ease-out;
        position: relative;
        overflow: hidden;
    }
    
    .mega-metric::after {
        content: '🚀';
        position: absolute;
        top: 20px;
        right: 30px;
        font-size: 3rem;
        opacity: 0.2;
    }
    
    .mega-value {
        font-size: 5rem;
        font-weight: 900;
        margin: 1rem 0;
        text-shadow: 3px 3px 6px rgba(0,0,0,0.3);
        animation: countUp 1.5s ease-out;
        background: linear-gradient(45deg, #FFD700, #FFF);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    .mega-label {
        font-size: 1.3rem;
        opacity: 0.9;
        margin-bottom: 0.5rem;
    }
    
    .mega-impact {
        font-size: 1.2rem;
        margin-top: 1rem;
        padding: 1rem;
        background: rgba(255, 255, 255, 0.1);
        border-radius: 15px;
        border: 1px solid rgba(255, 255, 255, 0.2);
    }
    
    .premium-kpi {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 20px;
        color: white;
        text-align: center;
        margin-bottom: 1.5rem;
        box-shadow: 0 15px 35px rgba(102, 126, 234, 0.3);
        transition: transform 0.3s ease;
        animation: slideInUp 0.8s ease-out;
    }
    
    .premium-kpi:hover {
        transform: translateY(-5px);
        box-shadow: 0 20px 40px rgba(102, 126, 234, 0.4);
    }
    
    .kpi-icon {
        font-size: 2.5rem;
        margin-bottom: 1rem;
        display: block;
    }
    
    .kpi-value {
        font-size: 2.8rem;
        font-weight: 900;
        margin: 1rem 0;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    
    .kpi-label {
        font-size: 1rem;
        opacity: 0.9;
        line-height: 1.4;
    }
    
    .insight-card {
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        border-left: 6px solid #28a745;
        padding: 2rem;
        margin: 2rem 0;
        border-radius: 15px;
        box-shadow: 0 10px 25px rgba(0,0,0,0.1);
        animation: slideInUp 1s ease-out;
    }
    
    .insight-title {
        color: #155724;
        font-size: 1.3rem;
        font-weight: bold;
        margin-bottom: 1rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    .comparison-card {
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1rem 0;
        box-shadow: 0 8px 20px rgba(0,0,0,0.1);
        transition: transform 0.3s ease;
    }
    
    .comparison-card:hover {
        transform: scale(1.02);
    }
    
    .current-state {
        background: linear-gradient(135deg, #fff3cd 0%, #ffeaa7 100%);
        border-left: 5px solid #ffc107;
    }
    
    .enhanced-state {
        background: linear-gradient(135deg, #d4edda 0%, #a8e6cf 100%);
        border-left: 5px solid #28a745;
    }
    
    .state-title {
        font-weight: bold;
        margin-bottom: 1rem;
        font-size: 1.1rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    .metric-highlight {
        background: rgba(255, 255, 255, 0.8);
        padding: 0.8rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    }
    
    .strategic-summary {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 3rem 2rem;
        border-radius: 20px;
        margin: 3rem 0;
        box-shadow: 0 20px 40px rgba(102, 126, 234, 0.3);
        position: relative;
        overflow: hidden;
    }
    
    .strategic-summary::before {
        content: '⭐';
        position: absolute;
        top: 20px;
        right: 30px;
        font-size: 4rem;
        opacity: 0.2;
    }
    
    .action-button {
        background: linear-gradient(135deg, #FF6600 0%, #FF8C42 100%);
        color: white;
        border: none;
        padding: 1rem 2rem;
        border-radius: 15px;
        font-size: 1.1rem;
        font-weight: bold;
        cursor: pointer;
        transition: all 0.3s ease;
        box-shadow: 0 8px 20px rgba(255, 102, 0, 0.3);
    }
    
    .action-button:hover {
        transform: translateY(-3px);
        box-shadow: 0 12px 25px rgba(255, 102, 0, 0.4);
    }
    
    .disclaimer-box {
        background: linear-gradient(135deg, #e9ecef 0%, #f8f9fa 100%);
        border: 2px solid #dee2e6;
        border-radius: 15px;
        padding: 2rem;
        margin: 3rem 0;
        box-shadow: 0 8px 20px rgba(0,0,0,0.1);
    }
    
    .value-card {
        background: linear-gradient(135deg, #ffffff 0%, #f8f9fa 100%);
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 8px 20px rgba(0,0,0,0.1);
        margin: 1rem 0;
        border-left: 4px solid #FF6600;
    }
    
    .progress-ring {
        width: 120px;
        height: 120px;
        margin: 1rem auto;
    }
    
    .section-divider {
        height: 4px;
        background: linear-gradient(90deg, #FF6600 0%, #FF8C42 50%, #FFB84D 100%);
        border-radius: 2px;
        margin: 3rem 0;
        animation: pulse 2s ease-in-out infinite;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Hero Section with Dynamic Elements
    st.markdown("""
    <div class='hero-container'>
        <h1 class='hero-title'>🎯 AI Forecasting Revolution</h1>
        <p class='hero-subtitle'>
            Transforming Webjet's $1.5B TTV • Precision Intelligence • Market Leadership
        </p>
        <p class='hero-impact'>
            ⚡ From Reactive to Predictive • 12-Month Strategic Impact Analysis • FY30 Growth Acceleration
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Mega Hero Metric with Animation
    st.markdown("""
    <div class='mega-metric'>
        <div class='mega-label'>Forecast Accuracy Transformation</div>
        <div class='mega-value'>15% → 28%</div>
        <div class='mega-impact'>
            <strong>🎯 13 percentage point breakthrough</strong><br>
            Unlocking $18.5M annual value across entire operation<br>
            <em>Supporting $3.2B+ TTV target by FY30</em>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Generate realistic data
    with st.spinner("🔄 Generating real-time analytics..."):
        time.sleep(0.5)  # Brief pause for dramatic effect
        forecast_data, current_performance, enhanced_performance = generate_webjet_data()
    
    # Dynamic KPI Dashboard
    st.markdown("## 🏆 Executive Performance Dashboard: Current vs AI-Enhanced")
    
    kpis = calculate_webjet_kpis(forecast_data, current_performance, enhanced_performance)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class='premium-kpi' style='background: linear-gradient(135deg, #28a745 0%, #20c997 100%);'>
            <span class='kpi-icon'>💰</span>
            <div class='kpi-label'>Revenue Acceleration</div>
            <div class='kpi-value'>${kpis['revenue_boost']:.1f}M</div>
            <div class='kpi-label'>Annual incremental growth<br>🚀 Supporting FY30 strategy</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class='premium-kpi' style='background: linear-gradient(135deg, #007bff 0%, #0056b3 100%);'>
            <span class='kpi-icon'>🎯</span>
            <div class='kpi-label'>Marketing ROI Explosion</div>
            <div class='kpi-value'>+{kpis['marketing_roi']:.0f}%</div>
            <div class='kpi-label'>Precision targeting impact<br>📈 International growth focus</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class='premium-kpi' style='background: linear-gradient(135deg, #fd7e14 0%, #e55a4e 100%);'>
            <span class='kpi-icon'>📊</span>
            <div class='kpi-label'>EBITDA Margin Leap</div>
            <div class='kpi-value'>+{kpis['margin_expansion']:.1f}%</div>
            <div class='kpi-label'>From 43.0% to 46.8%<br>⚡ Operational excellence</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class='premium-kpi' style='background: linear-gradient(135deg, #6610f2 0%, #6f42c1 100%);'>
            <span class='kpi-icon'>⚡</span>
            <div class='kpi-label'>Cost Optimization</div>
            <div class='kpi-value'>${kpis['cost_reduction']:.1f}M</div>
            <div class='kpi-label'>Annual savings unlock<br>🎯 Efficiency breakthrough</div>
        </div>
        """, unsafe_allow_html=True)

    # Section Divider
    st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)

    # Interactive Use Case 1: Marketing Revolution
    st.markdown("## 🚀 Marketing Revolution: From Seasonal to AI-Powered Precision")
    
    # Create tabs for better engagement
    tab1, tab2, tab3 = st.tabs(["📊 Performance Analysis", "🎯 Strategic Impact", "💡 Key Insights"])
    
    with tab1:
        col1, col2 = st.columns([3, 2])
        
        with col1:
            fig_marketing = create_dynamic_marketing_chart(current_performance, enhanced_performance)
            st.plotly_chart(fig_marketing, use_container_width=True, key="marketing_chart")
        
        with col2:
            st.markdown("""
            <div class='comparison-card current-state'>
                <div class='state-title'>🔄 Current State (FY25 Baseline)</div>
                <div class='metric-highlight'>
                    <strong>Marketing Budget:</strong> ~$25M annually<br>
                    <strong>Efficiency:</strong> Quarterly adjustments<br>
                    <strong>International Mix:</strong> 21% of bookings<br>
                    <strong>Response Time:</strong> 2-3 week cycles
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("""
            <div class='comparison-card enhanced-state'>
                <div class='state-title'>🚀 AI-Enhanced Future</div>
                <div class='metric-highlight'>
                    <strong>Smart Allocation:</strong> Real-time optimization<br>
                    <strong>Efficiency:</strong> +22% ROI improvement<br>
                    <strong>International Target:</strong> 28-30% by FY27<br>
                    <strong>Response Time:</strong> 4-hour adjustments
                </div>
            </div>
            """, unsafe_allow_html=True)
    
    with tab2:
        marketing_impact = calculate_marketing_impact()
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Annual Value Add", f"${marketing_impact['value']:,.0f}", f"+{marketing_impact['growth']:.1f}%")
        with col2:
            st.metric("International Growth", "21% → 28%", "+7 percentage points")
        with col3:
            st.metric("Campaign Efficiency", f"+{marketing_impact['efficiency']:.0f}%", "Real-time optimization")
    
    with tab3:
        st.markdown("""
        <div class='insight-card'>
            <div class='insight-title'>🎯 Strategic Marketing Intelligence</div>
            <ul style='font-size: 1.1rem; line-height: 1.6;'>
                <li><strong>Micro-Seasonality Detection:</strong> Identify 48-hour demand patterns for $2.1M additional capture</li>
                <li><strong>REX Closure Recovery:</strong> AI predicted capacity gaps 30 days ahead, captured 15% displaced bookings</li>
                <li><strong>International Acceleration:</strong> Dynamic content optimization for 28-30% international mix by FY27</li>
                <li><strong>NDC Partnership Leverage:</strong> Optimize unique airline content with demand-driven positioning</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    # Section Divider
    st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)

    # Interactive Use Case 2: Revenue Optimization Revolution
    st.markdown("## 💎 Revenue Optimization: ABV Journey from $1,046 to $1,180")
    
    tab1, tab2, tab3 = st.tabs(["📈 Revenue Dynamics", "🔬 Deep Analysis", "🎯 Strategic Wins"])
    
    with tab1:
        col1, col2 = st.columns([3, 2])
        
        with col1:
            fig_revenue = create_revenue_optimization_chart(current_performance, enhanced_performance)
            st.plotly_chart(fig_revenue, use_container_width=True, key="revenue_chart")
        
        with col2:
            revenue_metrics = calculate_revenue_metrics()
            
            st.markdown(f"""
            <div class='value-card'>
                <h4 style='color: #FF6600; margin-top: 0;'>🎯 Revenue Transformation</h4>
                <div style='font-size: 1.1rem;'>
                    <strong>Current ABV:</strong> $1,046 (FY25 actual)<br>
                    <strong>AI-Enhanced ABV:</strong> $1,180 target<br>
                    <strong>Improvement:</strong> +$134 per booking<br>
                    <strong>Annual Impact:</strong> ${revenue_metrics['annual_lift']:,.0f}
                </div>
            </div>
            """, unsafe_allow_html=True)
    
    with tab2:
        col1, col2, col3, col4 = st.columns(4)
        
        metrics = [
            ("Cross-Sell Rate", "34% → 42%", "+8%"),
            ("Ancillary Revenue", "$355 → $472", "+$117"),
            ("Package Attach", "12% → 18%", "+6%"),
            ("Premium Upgrade", "8% → 13%", "+5%")
        ]
        
        for i, (label, value, change) in enumerate(metrics):
            with [col1, col2, col3, col4][i]:
                st.metric(label, value, change)
    
    with tab3:
        st.markdown("""
        <div class='insight-card'>
            <div class='insight-title'>💰 Revenue Excellence Insights</div>
            <ul style='font-size: 1.1rem; line-height: 1.6;'>
                <li><strong>Seat Selection Revolution:</strong> Now live for 18 airlines (vs 1 in FY24) - AI optimizes pricing by route demand</li>
                <li><strong>Hotel Cross-Sell Mastery:</strong> 28% first-time customer conversion with predictive recommendations</li>
                <li><strong>Package Premium Capture:</strong> New partnerships + brand refresh + AI demand forecasting = 25% uplift</li>
                <li><strong>International Revenue Premium:</strong> 15% higher margins on international routes with AI optimization</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    # Section Divider
    st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)

    # Use Case 3: Operational Excellence
    st.markdown("## ⚡ Operational Excellence: EBITDA Margin Transformation")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        fig_operations = create_operations_excellence_chart()
        st.plotly_chart(fig_operations, use_container_width=True, key="operations_chart")
    
    with col2:
        st.markdown("""
        <div class='comparison-card current-state'>
            <div class='state-title'>📊 FY25 Performance</div>
            <div class='metric-highlight'>
                <strong>EBITDA Margin:</strong> 43.0% (-170bps YoY)<br>
                <strong>Contact Volume:</strong> -24% improvement<br>
                <strong>Cost per Call:</strong> -10% reduction<br>
                <strong>NPS Score:</strong> +12 points improvement
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class='comparison-card enhanced-state'>
            <div class='state-title'>🚀 AI-Enhanced Target</div>
            <div class='metric-highlight'>
                <strong>Target Margin:</strong> 46.8% (+380bps)<br>
                <strong>Contact Reduction:</strong> -35% total<br>
                <strong>Cost Efficiency:</strong> -25% per interaction<br>
                <strong>NPS Target:</strong> +20 points (industry leading)
            </div>
        </div>
        """, unsafe_allow_html=True)

    # Section Divider
    st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)

    # Competitive Intelligence & Market Leadership
    st.markdown("## 🏆 Market Leadership: Competitive Intelligence Revolution")
    
    col1, col2 = st.columns([3, 2])
    
    with col1:
        fig_competitive = create_competitive_intelligence_chart()
        st.plotly_chart(fig_competitive, use_container_width=True, key="competitive_chart")
    
    with col2:
        st.markdown("""
        <div class='insight-card' style='margin: 0;'>
            <div class='insight-title'>🥇 Market Dominance Strategy</div>
            <div style='font-size: 1rem; line-height: 1.5;'>
                <strong>🎯 Jetstar Response:</strong> Predict promotional cycles 5 days ahead<br><br>
                <strong>✈️ Virgin Capacity:</strong> Monitor route changes for market share protection<br><br>
                <strong>🛫 Rex Recovery:</strong> Captured 15% of displaced bookings through AI prediction<br><br>
                <strong>🌏 International Growth:</strong> NZ gateway optimization with demand forecasting
            </div>
        </div>
        """, unsafe_allow_html=True)

    # Section Divider
    st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)

    # Ultimate Business Impact Summary
    st.markdown("## 🎯 Ultimate Business Impact: 12-Month Value Creation")
    
    total_impact = calculate_total_impact()
    
    # Interactive value creation chart
    col1, col2 = st.columns([3, 2])
    
    with col1:
        fig_waterfall = create_value_waterfall_chart(total_impact)
        st.plotly_chart(fig_waterfall, use_container_width=True, key="waterfall_chart")
    
    with col2:
        # ROI Calculator with real-time updates
        st.markdown("""
        <div class='value-card'>
            <h3 style='color: #FF6600; margin-top: 0;'>💎 Investment Summary</h3>
        </div>
        """, unsafe_allow_html=True)
        
        col_a, col_b = st.columns(2)
        with col_a:
            st.metric("Total Value", f"${total_impact['total_value']:.1f}M", "Annual benefit")
            st.metric("Implementation", f"${total_impact['investment']:.1f}M", "One-time cost")
        with col_b:
            st.metric("Net ROI", f"{total_impact['roi']:.0f}%", "First year")
            st.metric("Payback", f"{total_impact['payback']} months", "Break-even")

    # Strategic Roadmap
    st.markdown("""
    <div class='strategic-summary'>
        <h3 style='margin-top: 0; color: white; font-size: 2rem;'>🚀 Strategic Implementation Roadmap</h3>
        <div style='font-size: 1.2rem; line-height: 1.8; margin-top: 2rem;'>
            <strong>🎯 Immediate Opportunity:</strong> Transform Webjet's forecasting accuracy from 15% to 28%, unlocking $18.5M annual value to accelerate FY30 TTV target of $3.2B+
        </div>
        <div style='margin-top: 2rem; font-size: 1.1rem; line-height: 1.6;'>
            <div style='background: rgba(255,255,255,0.1); padding: 1.5rem; border-radius: 15px; margin: 1rem 0;'>
                <strong>🚀 Phase 1 (Q1-Q2 FY26):</strong> Marketing & Pricing AI Integration - $8.2M impact<br>
                <em>Align with announced $15M growth investment strategy</em>
            </div>
            <div style='background: rgba(255,255,255,0.1); padding: 1.5rem; border-radius: 15px; margin: 1rem 0;'>
                <strong>⚡ Phase 2 (Q3-Q4 FY26):</strong> Operational Excellence & Customer Service AI - $6.1M impact<br>
                <em>Build on award-winning service achievements</em>
            </div>
            <div style='background: rgba(255,255,255,0.1); padding: 1.5rem; border-radius: 15px; margin: 1rem 0;'>
                <strong>🏆 Phase 3 (FY27-FY30):</strong> Market Leadership & International Expansion - $4.2M ongoing<br>
                <em>Support 25-30% international booking target</em>
            </div>
        </div>
        <div style='margin-top: 2rem; padding: 1.5rem; background: rgba(255,215,0,0.2); border-radius: 15px; border: 2px solid #FFD700;'>
            <strong style='font-size: 1.3rem;'>🎯 Success Metric: Maintain #1 OTA position while expanding EBITDA margin from 43% to 47%+ by FY30</strong>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Action Center
    st.markdown("### 🚀 Executive Action Center")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📧 Email Complete Business Case", key="email_case", help="Send detailed analysis to executive team"):
            st.success("✅ Comprehensive AI forecasting business case sent to Board, CEO, and C-suite executives")
            st.balloons()
    
    with col2:
        if st.button("📊 Download FY30 Strategic Plan", key="download_plan", help="Get full integration roadmap"):
            st.success("✅ 89-page strategic integration plan with FY30 alignment downloading...")
            # Simulate download progress
            progress = st.progress(0)
            for i in range(100):
                time.sleep(0.01)
                progress.progress(i + 1)
            st.success("Download complete! 📄")
    
    with col3:
        if st.button("🎯 Schedule Investment Committee", key="schedule_meeting", help="Book presentation with investment committee"):
            st.success("✅ Board investment committee presentation scheduled for next Wednesday 2:00 PM")
            st.info("📅 Calendar invite sent to all attendees with pre-read materials")

    # Interactive Demo Section
    with st.expander("🔬 Interactive Demo: See AI Forecasting in Action", expanded=False):
        st.markdown("### Real-Time Forecasting Simulation")
        
        col1, col2 = st.columns(2)
        
        with col1:
            scenario = st.selectbox(
                "Select Business Scenario:",
                ["Summer Peak Demand", "Competitor Price War", "New Route Launch", "Marketing Campaign"]
            )
            
            impact_multiplier = st.slider("Scenario Impact Level", 0.5, 2.0, 1.0, 0.1)
        
        with col2:
            # Real-time calculation
            demo_value = calculate_demo_impact(scenario, impact_multiplier)
            
            st.metric("Predicted Revenue Impact", f"${demo_value['revenue']:,.0f}", f"{demo_value['change']:+.1f}%")
            st.metric("Forecast Confidence", f"{demo_value['confidence']:.1f}%", "AI-powered")
            
        # Mini chart
        demo_chart = create_demo_forecast_chart(scenario, impact_multiplier)
        st.plotly_chart(demo_chart, use_container_width=True, key="demo_chart")

    # Professional Disclaimer with Style
    st.markdown("""
    <div class='disclaimer-box'>
        <h4 style='color: #495057; margin-top: 0; display: flex; align-items: center; gap: 0.5rem;'>
            📋 Professional Disclaimer
        </h4>
        <div style='font-size: 1rem; line-height: 1.6; color: #495057;'>
            <strong>For Strategic Demonstration Purposes:</strong> This executive dashboard leverages Webjet Group Limited's 
            publicly available FY25 financial results to provide realistic context for AI forecasting impact analysis. 
            All enhancement projections, ROI calculations, and business value scenarios are hypothetical models created 
            for demonstration purposes only.
            <br><br>
            <strong>Investment Consideration:</strong> Actual results will vary based on implementation approach, 
            market conditions, and organizational factors. This analysis should be evaluated alongside comprehensive 
            due diligence and professional consultation.
            <br><br>
            <em style='font-size: 0.9rem;'>📚 Source: Webjet Group Limited FY25 Results - Investor Briefing, 21 May 2025</em>
        </div>
    </div>
    """, unsafe_allow_html=True)


# ============================================================================
# DYNAMIC DATA GENERATION & CALCULATION FUNCTIONS
# ============================================================================

@st.cache_data
def generate_webjet_data():
    """Generate engaging, realistic Webjet performance data."""
    np.random.seed(42)
    
    dates = pd.date_range(start='2024-04-01', periods=365, freq='D')
    
    # Webjet-scale demand (1.254M annual bookings)
    base_demand = 3438  # Daily average
    
    # Enhanced seasonality with Australian travel patterns
    seasonal_pattern = (
        np.sin(2 * np.pi * np.arange(365) / 365 + np.pi/6) * 0.3 +  # Annual cycle
        np.sin(4 * np.pi * np.arange(365) / 365) * 0.1 +  # Semi-annual
        1
    )
    
    # Weekly patterns with strong weekend bias
    weekly_pattern = np.array([0.82, 0.87, 0.94, 1.28, 1.35, 1.18, 0.89])
    weekly_multiplier = np.array([weekly_pattern[d.weekday()] for d in dates])
    
    # Special events and holidays
    event_multiplier = np.ones(365)
    for i, date in enumerate(dates):
        # Major holiday periods
        if date.month == 12 and date.day > 18:  # Christmas rush
            event_multiplier[i] = 1.75
        elif date.month == 1 and date.day < 12:  # New Year
            event_multiplier[i] = 1.55
        elif date.month == 4 and 10 <= date.day <= 17:  # Easter
            event_multiplier[i] = 1.4
        elif date.month == 7 and 1 <= date.day <= 16:  # Winter holidays
            event_multiplier[i] = 1.3
        elif date.month == 10 and 1 <= date.day <= 8:  # Spring break
            event_multiplier[i] = 1.25
    
    # True demand with realistic volatility
    true_demand = base_demand * seasonal_pattern * weekly_multiplier * event_multiplier
    
    # Current vs Enhanced forecasting (15% vs 28% accuracy)
    current_error = np.random.normal(0, 0.35, 365)  # Higher error
    enhanced_error = np.random.normal(0, 0.12, 365)  # Lower error
    
    current_forecast = true_demand * (1 + current_error)
    enhanced_forecast = true_demand * (1 + enhanced_error)
    
    # Ensure positive values
    current_forecast = np.maximum(current_forecast, true_demand * 0.3)
    enhanced_forecast = np.maximum(enhanced_forecast, true_demand * 0.75)
    
    forecast_data = pd.DataFrame({
        'date': dates,
        'true_demand': true_demand.astype(int),
        'seasonal_pattern': seasonal_pattern,
        'weekly_pattern': weekly_multiplier,
        'events': event_multiplier
    })
    
    current_performance = pd.DataFrame({
        'date': dates,
        'forecast': current_forecast.astype(int),
        'accuracy': np.abs(current_forecast - true_demand) / true_demand,
        'marketing_efficiency': np.random.normal(0.82, 0.03, 365),
        'abv': np.random.normal(1046, 25, 365)
    })
    
    enhanced_performance = pd.DataFrame({
        'date': dates,
        'forecast': enhanced_forecast.astype(int),
        'accuracy': np.abs(enhanced_forecast - true_demand) / true_demand,
        'marketing_efficiency': np.random.normal(0.97, 0.02, 365),
        'abv': np.random.normal(1180, 15, 365)
    })
    
    return forecast_data, current_performance, enhanced_performance


def calculate_webjet_kpis(forecast_data, current_performance, enhanced_performance):
    """Calculate compelling KPIs with Webjet context."""
    
    # Revenue boost calculation
    annual_bookings = 1254000  # FY25 actual
    abv_improvement = enhanced_performance['abv'].mean() - current_performance['abv'].mean()
    revenue_boost = (abv_improvement * annual_bookings) / 1_000_000
    
    # Marketing ROI improvement
    current_efficiency = current_performance['marketing_efficiency'].mean()
    enhanced_efficiency = enhanced_performance['marketing_efficiency'].mean()
    marketing_roi = ((enhanced_efficiency - current_efficiency) / current_efficiency) * 100
    
    # EBITDA margin expansion
    current_margin = 43.0  # FY25 actual
    target_margin = 46.8   # AI target
    margin_expansion = target_margin - current_margin
    
    # Cost reduction from efficiency gains
    cost_reduction = 2.8  # Millions from operational efficiency
    
    return {
        'revenue_boost': revenue_boost,
        'marketing_roi': marketing_roi,
        'margin_expansion': margin_expansion,
        'cost_reduction': cost_reduction
    }


def calculate_marketing_impact():
    """Calculate dynamic marketing impact metrics."""
    return {
        'value': 4_500_000,  # Annual value add
        'growth': 22.0,      # Growth percentage
        'efficiency': 18.5   # Efficiency improvement
    }


def calculate_revenue_metrics():
    """Calculate revenue transformation metrics."""
    return {
        'current_abv': 1046,
        'target_abv': 1180,
        'improvement': 134,
        'annual_lift': 168_036_000  # 134 * 1,254,000 bookings
    }


def calculate_total_impact():
    """Calculate total business impact for waterfall chart."""
    return {
        'total_value': 18.5,      # Total annual value in millions
        'investment': 2.8,        # Implementation cost
        'roi': 561,              # ROI percentage
        'payback': 10,           # Payback in months
        'marketing': 4.5,        # Marketing value
        'revenue': 8.9,          # Revenue optimization
        'operations': 3.8,       # Operational efficiency
        'competitive': 1.3       # Competitive advantage
    }


# ============================================================================
# DYNAMIC VISUALIZATION FUNCTIONS
# ============================================================================

def create_dynamic_marketing_chart(current_performance, enhanced_performance):
    """Create engaging marketing performance chart."""
    sample_data = current_performance.head(90).copy()  # 3 months of data
    
    # Marketing efficiency over time
    current_eff = sample_data['marketing_efficiency'] * 100
    enhanced_eff = enhanced_performance.head(90)['marketing_efficiency'] * 100
    
    fig = go.Figure()
    
    # Current performance
    fig.add_trace(go.Scatter(
        x=sample_data['date'],
        y=current_eff,
        name="Current Marketing Efficiency",
        line=dict(color='#dc3545', width=4, dash='dot'),
        fill='tonexty',
        fillcolor='rgba(220, 53, 69, 0.1)'
    ))
    
    # Enhanced performance
    fig.add_trace(go.Scatter(
        x=sample_data['date'],
        y=enhanced_eff,
        name="AI-Enhanced Efficiency",
        line=dict(color='#28a745', width=4),
        fill='tonexty',
        fillcolor='rgba(40, 167, 69, 0.2)'
    ))
    
    # Add annotations for key insights
    fig.add_annotation(
        x=sample_data['date'].iloc[30],
        y=enhanced_eff.iloc[30],
        text="🚀 22% ROI<br>Improvement",
        showarrow=True,
        arrowcolor="#28a745",
        arrowwidth=3,
        arrowhead=2,
        bgcolor="rgba(40, 167, 69, 0.8)",
        font=dict(color="white", size=12)
    )
    
    fig.update_layout(
        title="🎯 Marketing Revolution: Seasonal → AI-Powered Precision Targeting",
        height=500,
        template='plotly_white',
        hovermode='x unified',
        yaxis_title="Marketing Efficiency (%)",
        xaxis_title="Timeline",
        font=dict(size=14),
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    return fig


def create_revenue_optimization_chart(current_performance, enhanced_performance):
    """Create dynamic revenue optimization visualization."""
    sample_data = current_performance.head(60).copy()  # 2 months
    
    current_abv = sample_data['abv']
    enhanced_abv = enhanced_performance.head(60)['abv']
    
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=("Average Booking Value ($)", "Revenue Impact ($M)"),
        vertical_spacing=0.12,
        specs=[[{"secondary_y": False}], [{"secondary_y": False}]]
    )
    
    # ABV comparison
    fig.add_trace(
        go.Scatter(
            x=sample_data['date'],
            y=current_abv,
            name="Current ABV ($1,046)",
            line=dict(color='#ffc107', width=3),
            mode='lines'
        ),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=sample_data['date'],
            y=enhanced_abv,
            name="AI-Enhanced ABV ($1,180)",
            line=dict(color='#28a745', width=4),
            mode='lines',
            fill='tonexty',
            fillcolor='rgba(40, 167, 69, 0.2)'
        ),
        row=1, col=1
    )
    
    # Daily revenue impact
    daily_bookings = 3438  # Average daily bookings
    daily_impact = (enhanced_abv - current_abv) * daily_bookings / 1_000_000  # Convert to millions
    
    fig.add_trace(
        go.Bar(
            x=sample_data['date'],
            y=daily_impact,
            name="Daily Revenue Lift",
            marker_color='rgba(255, 102, 0, 0.7)'
        ),
        row=2, col=1
    )
    
    fig.update_layout(
        title="💎 Revenue Optimization: ABV Transformation Journey",
        height=600,
        template='plotly_white',
        hovermode='x unified',
        font=dict(size=13),
        showlegend=True
    )
    
    return fig


def create_operations_excellence_chart():
    """Create operational excellence transformation chart."""
    months = ['Apr 25', 'May 25', 'Jun 25', 'Jul 25', 'Aug 25', 'Sep 25', 
              'Oct 25', 'Nov 25', 'Dec 25', 'Jan 26', 'Feb 26', 'Mar 26']
    
    # EBITDA margin progression
    current_margins = [43.0, 42.8, 43.1, 43.2, 43.4, 43.3, 43.5, 43.6, 43.8, 43.9, 44.0, 44.1]
    ai_margins = [43.0, 43.8, 44.5, 45.0, 45.4, 45.7, 46.0, 46.2, 46.4, 46.5, 46.7, 46.8]
    
    # Customer satisfaction scores
    nps_current = [72, 71, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82]
    nps_ai = [72, 75, 78, 81, 83, 85, 87, 88, 89, 90, 91, 92]
    
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=("EBITDA Margin Evolution (%)", "Customer Satisfaction (NPS)"),
        vertical_spacing=0.15
    )
    
    # EBITDA margins
    fig.add_trace(
        go.Scatter(
            x=months, y=current_margins,
            name="Current Trajectory", 
            line=dict(color='#ffc107', width=3, dash='dash'),
            mode='lines+markers'
        ),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=months, y=ai_margins,
            name="AI-Enhanced Performance",
            line=dict(color='#28a745', width=4),
            mode='lines+markers',
            fill='tonexty',
            fillcolor='rgba(40, 167, 69, 0.2)'
        ),
        row=1, col=1
    )
    
    # NPS scores
    fig.add_trace(
        go.Scatter(
            x=months, y=nps_current,
            name="Current NPS",
            line=dict(color='#17a2b8', width=3, dash='dash'),
            mode='lines+markers',
            showlegend=False
        ),
        row=2, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=months, y=nps_ai,
            name="AI-Enhanced NPS",
            line=dict(color='#007bff', width=4),
            mode='lines+markers',
            fill='tonexty',
            fillcolor='rgba(0, 123, 255, 0.2)',
            showlegend=False
        ),
        row=2, col=1
    )
    
    fig.update_layout(
        title="⚡ Operational Excellence: EBITDA & Customer Satisfaction Transformation",
        height=550,
        template='plotly_white',
        font=dict(size=13)
    )
    
    return fig


def create_competitive_intelligence_chart():
    """Create competitive market position chart."""
    quarters = ['Q4 FY24', 'Q1 FY25', 'Q2 FY25', 'Q3 FY25', 'Q4 FY25', 'Q1 FY26', 'Q2 FY26']
    
    # Market share evolution
    webjet_share = [34.2, 34.0, 33.8, 34.1, 34.3, 34.8, 35.2]
    jetstar_share = [28.5, 28.3, 28.1, 27.9, 27.8, 27.5, 27.2]
    virgin_share = [22.8, 22.6, 22.7, 22.5, 22.4, 22.3, 22.1]
    
    fig = go.Figure()
    
    # Webjet - enhanced with AI
    fig.add_trace(go.Scatter(
        x=quarters, y=webjet_share,
        name="Webjet (AI-Enhanced)",
        line=dict(color='#FF6600', width=5),
        mode='lines+markers',
        marker=dict(size=10, symbol='star')
    ))
    
    # Competitors
    fig.add_trace(go.Scatter(
        x=quarters, y=jetstar_share,
        name="Jetstar",
        line=dict(color='#ff8c00', width=3, dash='dash'),
        mode='lines+markers'
    ))
    
    fig.add_trace(go.Scatter(
        x=quarters, y=virgin_share,
        name="Virgin Australia", 
        line=dict(color='#dc143c', width=3, dash='dot'),
        mode='lines+markers'
    ))
    
    # Add AI intervention point
    fig.add_vline(
        x=4.5, line_dash="solid", line_color="green", line_width=3,
        annotation_text="🚀 AI Implementation"
    )
    
    fig.update_layout(
        title="🏆 Market Leadership: Competitive Intelligence & Share Growth",
        height=450,
        template='plotly_white',
        yaxis_title="Market Share (%)",
        xaxis_title="Quarter",
        font=dict(size=14),
        hovermode='x unified'
    )
    
    return fig


def create_value_waterfall_chart(impact_data):
    """Create engaging value creation waterfall chart."""
    
    categories = [
        "FY25 Baseline",
        "Marketing AI",
        "Revenue Optimization", 
        "Operational Excellence",
        "Competitive Edge",
        "Implementation Cost",
        "Net Value Created"
    ]
    
    values = [
        0,  # Baseline (relative)
        impact_data['marketing'],
        impact_data['revenue'],
        impact_data['operations'],
        impact_data['competitive'],
        -impact_data['investment'],
        0  # Total (will be calculated)
    ]
    
    # Calculate running totals for waterfall
    running_totals = []
    current_total = 0
    
    for i, value in enumerate(values):
        if i == 0:  # Baseline
            running_totals.append(0)
        elif i == len(values) - 1:  # Final total
            running_totals.append(current_total)
        else:
            if value < 0:  # Cost
                running_totals.append(current_total + value)
            else:  # Benefit
                running_totals.append(current_total)
            current_total += value
    
    fig = go.Figure(go.Waterfall(
        name="Value Creation",
        orientation="v",
        measure=["relative", "relative", "relative", "relative", "relative", "relative", "total"],
        x=categories,
        textposition="outside",
        text=[f"${v:+.1f}M" if v != 0 else "Baseline" for v in values[:-1]] + [f"${current_total:.1f}M"],
        y=values[:-1] + [current_total],
        connector={"line": {"color": "rgba(63, 63, 63, 0.5)"}},
        increasing={"marker": {"color": "rgba(40, 167, 69, 0.8)"}},
        decreasing={"marker": {"color": "rgba(220, 53, 69, 0.8)"}},
        totals={"marker": {"color": "rgba(255, 102, 0, 0.8)"}}
    ))
    
    fig.update_layout(
        title="💰 Annual Value Creation: AI Forecasting Business Impact Waterfall",
        height=500,
        template='plotly_white',
        yaxis_title="Value Impact ($M)",
        font=dict(size=14),
        showlegend=False
    )
    
    return fig


def calculate_demo_impact(scenario, multiplier):
    """Calculate demo values for interactive section."""
    base_impacts = {
        "Summer Peak Demand": {"revenue": 125000, "confidence": 87.2},
        "Competitor Price War": {"revenue": -45000, "confidence": 91.5},
        "New Route Launch": {"revenue": 89000, "confidence": 78.9},
        "Marketing Campaign": {"revenue": 156000, "confidence": 93.1}
    }
    
    base = base_impacts[scenario]
    adjusted_revenue = base["revenue"] * multiplier
    change_pct = ((multiplier - 1) * 100)
    
    return {
        "revenue": adjusted_revenue,
        "confidence": base["confidence"],
        "change": change_pct
    }


def create_demo_forecast_chart(scenario, multiplier):
    """Create mini demo forecast chart."""
    days = pd.date_range(start='2025-01-01', periods=30, freq='D')
    base_demand = 3400
    
    # Scenario-specific patterns
    if scenario == "Summer Peak Demand":
        demand_pattern = base_demand * (1 + 0.3 * multiplier * np.sin(np.arange(30) / 5))
    elif scenario == "Competitor Price War":
        demand_pattern = base_demand * (0.9 + 0.1 * multiplier * np.random.random(30))
    else:
        demand_pattern = base_demand * (1 + 0.2 * multiplier * np.random.random(30))
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=days,
        y=demand_pattern,
        name=f"AI Forecast: {scenario}",
        line=dict(color='#FF6600', width=3),
        mode='lines+markers',
        marker=dict(size=6)
    ))
    
    fig.update_layout(
        title=f"📊 {scenario} - AI Prediction",
        height=250,
        template='plotly_white',
        yaxis_title="Bookings",
        xaxis_title="Date",
        font=dict(size=11),
        showlegend=False,
        margin=dict(l=50, r=50, t=50, b=50)
    )
    
    return fig


if __name__ == "__main__":
    show()
    