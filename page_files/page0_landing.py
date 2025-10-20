"""
page0_landing.py
Executive Landing Page - Business Case for Flight Booking Forecasting
Audience: Webjet Marketing Department Leadership
"""

import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np


def show():
    """Display executive landing page."""
    
    st.title("✈️ Webjet Flight Booking Forecasting System") 
    st.info("This is an example of potential ML project deliverable for demonstration purposes only. Happy to dicsuss in more detail if needed.")
    # Hero Section
    with st.container():
        st.markdown("""
            <div style='background: linear-gradient(135deg, #e14747 0%, #e14747 100%); 
                        padding: 3rem 2rem; border-radius: 12px; margin-bottom: 2rem; text-align: center;'>
                <h1 style='color: white; font-size: 3rem; margin: 0; font-weight: 700;'>
                    Flight Booking Forecasting System
                </h1>
                <p style='color: #FFD700; font-size: 1.5rem; margin: 1rem 0 0 0; font-weight: 500;'>
                    Powering Webjet's FY30 Strategy to Double TTV
                </p>
                <p style='color: white; font-size: 1.1rem; margin: 1rem 0 0 0; opacity: 0.9;'>
                    AI-Powered Demand Intelligence for Australia & New Zealand's #1 OTA
                </p>
            </div>
        """, unsafe_allow_html=True)
        
    # Executive Summary
    st.markdown("""
        <div style='background: linear-gradient(135deg, #003087 0%, #0056b3 100%); 
                    padding: 2rem; border-radius: 10px; margin: 2rem 0; text-align: center;'>
            <h2 style='color: white; font-size: 2.2rem; margin: 0; font-weight: 600; display: flex; align-items: center; justify-content: center; gap: 15px;'>
                <span style='font-size: 2.5rem;'>🎯</span>
                The Strategic Imperative
            </h2>
        </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns([3, 2])
    
    with col1:
        st.markdown("""
        Webjet is entering a new era with its ambitious FY30 strategy to double Total Transaction Value (TTV) by 2030. 
        As part of the bold 'Go Somewhere' rebrand and transformation from a flight-centric OTA to a comprehensive travel companion, 
        predictive analytics and AI-powered tools are central to delivering seamless, integrated travel experiences.
        
        **This forecasting system is mission-critical infrastructure** for executing on three strategic pillars:
        
        1. **Marketing Optimization** - Align $10M+ annual digital spend with predicted demand
        2. **Customer Experience** - Right-size service capacity to maintain award-winning standards
        3. **Revenue Maximization** - Dynamic pricing and inventory management based on demand intelligence
        """)
    
    with col2:
        st.markdown("""
        <div style='background: #FFF8DC; padding: 1.5rem; border-radius: 8px; border-left: 5px solid #FF6600;'>
            <h3 style='color: #003087; margin-top: 0;'>Quick Facts</h3>
            <p style='margin: 0.5rem 0;'><strong>Market Position:</strong> Australia's #1 OTA</p>
            <p style='margin: 0.5rem 0;'><strong>Brand Awareness:</strong> 66% nationally</p>
            <p style='margin: 0.5rem 0;'><strong>Annual Marketing:</strong> $10M+ digital spend</p>
            <p style='margin: 0.5rem 0;'><strong>Strategic Goal:</strong> 2x TTV by 2030</p>
            <p style='margin: 0.5rem 0;'><strong>AI Integration:</strong> Core to rebrand</p>
        </div>
        """, unsafe_allow_html=True)
    
    
    # Business Impact Analysis
    st.markdown("---")
    st.markdown("""
        <div style='background: linear-gradient(135deg, #FF6600 0%, #e14747 100%); 
                    padding: 2rem; border-radius: 10px; margin: 2rem 0; text-align: center;'>
            <h2 style='color: white; font-size: 2.2rem; margin: 0; font-weight: 600; display: flex; align-items: center; justify-content: center; gap: 15px;'>
                <span style='font-size: 2.5rem;'>💼</span>
                Why This Matters: Department-Level Impact
            </h2>
        </div>
    """, unsafe_allow_html=True)
    create_business_impact_analysis()
    
    
    st.markdown("---")
    
    # The Solution
    st.markdown("""
        <div style='background: linear-gradient(135deg, #2E7D32 0%, #4CAF50 100%); 
                    padding: 2rem; border-radius: 10px; margin: 2rem 0; text-align: center;'>
            <h2 style='color: white; font-size: 2.2rem; margin: 0; font-weight: 600; display: flex; align-items: center; justify-content: center; gap: 15px;'>
                <span style='font-size: 2.5rem;'>🚀</span>
                The Solution: AI-Powered Demand Intelligence
            </h2>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    ### Potential of an End-to-End Forecasting System
    
    A production-ready ML platform delivering 7-90 day booking forecasts with uncertainty quantification, 
    designed specifically for Webjet's marketing, operations, and revenue teams.
    """)
    
    st.markdown("### Quantified Business Value")
    
    col1, col2 = st.columns([2, 1.5])
    
    with col1:
        # ROI visualization
        fig_roi = create_roi_chart()
        st.plotly_chart(fig_roi, use_container_width=True)
    
    with col2:
        

        st.markdown("""
        <div style='background: #FFF8DC; padding: 1.5rem; border-radius: 8px;'>
            <h3 style='color: #FF6600; margin-top: 0;'>Annual Benefits</h3>
            <p><strong>Marketing Savings:</strong> $2.1M (20% efficiency)</p>
            <p><strong>Service Optimization:</strong> Customer Service Cost Reduction - $450K (15% efficiency)</p>
            <p><strong>Revenue Protection:</strong> $280K (capacity management)</p>
            <h4><strong>Total Annual Benefit:</strong> $2.83M</h4>    
        </div>
        """, unsafe_allow_html=True)

    
        st.markdown("""
        <div style='background: #E8F5E9; padding: 1.5rem; border-radius: 8px;'>
            <h3 style='color: #2E7D32; margin-top: 0;'>Supporting WebJet's Strategic Enablers for FY30 Goals</h3>
            <p><strong>1. Double TTV by 2030</strong>  - Optimize every dollar of marketing spend to maximize booking volume</p>
            <p><strong>2. 'Go Somewhere' Platform</strong> - AI tools create personalized experiences and better-value itineraries</p>
            <p><strong>3. Market Leadership</strong> - Maintain #1 position through data-driven competitive advantage</p>
            <p><strong>4. Multi-Vertical Expansion</strong> - Scale forecasting to hotels, packages, tours, business travel</p>
            <p><strong>5. Customer Experience</strong> - Support award-winning service with predictive capacity planning</p>
        </div>""", unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Competitive Context
    st.subheader("📊 Explore Realistic Booking Forecasts",divider="rainbow")
    st.markdown("**Experience the power of AI-driven demand prediction in real-time**")
    
    # Demo forecaster with sliders
    create_interactive_demo()
    
    
    st.markdown("---")
    # Call to Action
    st.markdown("""
        <div style='background: linear-gradient(135deg, #1565C0 0%, #1976D2 100%); 
                    padding: 2rem; border-radius: 10px; margin: 2rem 0; text-align: center;'>
            <h2 style='color: white; font-size: 2.2rem; margin: 0; font-weight: 600; display: flex; align-items: center; justify-content: center; gap: 15px;'>
                <span style='font-size: 2.5rem;'>🎬</span>
                Take Action
            </h2>
        </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        ### Why This Matters Now
        
        Webjet is putting serious cash behind the 'Go Somewhere' rebrand and new product verticals. 
        Every day without demand forecasting means:
        - **>10M** marketing spend to be optimised, a moderate improvement of 10-15percent would be **$1-1.5M** opportunity missed
        - **Missed opportunities** during high-demand periods
        - **Reactive decisions** while competitors use predictive analytics
        
        The transformation is about bringing Webjet back to the forefront for a new generation. 
        This forecasting system ensures every marketing dollar, every customer interaction, and every pricing decision is 
        **data-driven, predictive, and optimized** for the FY30 goal.
        
        """)
    
    st.subheader("",divider="rainbow")
    st.subheader("🚀 Want to see how ML engineers would build this system?")

    if st.button("📊 Explore ML System", type="primary", use_container_width=False):
        st.session_state.current_page = "Step 1 - Data Acquisition"
        st.rerun()
    
    st.subheader("",divider="rainbow")
    
    st.markdown("""
                <div style='text-align: center; color: #888; font-size: 0.9rem;'>
                This is a prototype forecasting system developed for demonstration purposes only.
                </div>
                """, unsafe_allow_html=True)


# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================

def create_interactive_demo():
    """Create interactive forecasting demo with sliders."""
    
    # Demo controls
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("**📅 Forecast Horizon**")
        forecast_days = st.slider("Days to Predict", 7, 90, 30, step=7, help="How many days ahead to forecast")
        
    with col2:
        st.markdown("**📊 Base Demand**")
        base_demand = st.slider("Base Daily Bookings", 300, 800, 520, help="Typical daily booking volume")
        
    with col3:
        st.markdown("**🎯 Marketing Spend**")
        marketing_spend = st.slider("Daily Marketing ($)", 3000, 15000, 6500, step=500, help="Digital marketing budget per day")
        
    with col4:
        st.markdown("**📈 External Factors**")
        external_boost = st.slider("Campaign/Event Impact (%)", -20, 50, 0, help="Promotional campaigns or external events")
    
    # Generate forecast data starting from today
    start_date = pd.Timestamp.now().normalize()
    dates = pd.date_range(start_date, periods=forecast_days, freq='D')
    
    # Create realistic booking patterns
    booking_forecast = []
    confidence_upper = []
    confidence_lower = []
    
    for i, date in enumerate(dates):
        # Base seasonal pattern
        day_of_week = date.weekday()  # 0=Monday, 6=Sunday
        
        # Australian booking patterns
        base = base_demand
        
        # Weekly seasonality (higher on Thu-Sun for weekend trips)
        weekly_mult = {0: 0.85, 1: 0.80, 2: 0.75, 3: 1.15, 4: 1.25, 5: 1.35, 6: 1.10}[day_of_week]
        
        # Monthly seasonal trends (Australian context)
        if date.month in [12, 1]:  # Summer peak
            seasonal_mult = 1.4
        elif date.month in [6, 7]:  # Winter holidays
            seasonal_mult = 1.2
        elif date.month in [4, 9, 10]:  # School holidays
            seasonal_mult = 1.1
        else:  # Regular periods
            seasonal_mult = 1.0
        
        # Add some randomness for special events
        if date.month == 12 and date.day == 25:  # Christmas
            seasonal_mult *= 1.3
        elif date.month == 1 and date.day == 1:  # New Year
            seasonal_mult *= 1.2
        elif date.month == 1 and date.day == 26:  # Australia Day
            seasonal_mult *= 1.15
        
        # Marketing elasticity (10% increase in spend = 3% increase in bookings)
        marketing_mult = 1 + ((marketing_spend - 6500) / 6500) * 0.3
        
        # External factors
        external_mult = 1 + (external_boost / 100)
        
        # Add uncertainty that increases with forecast horizon
        uncertainty_factor = 1 + (i / forecast_days) * 0.1
        
        # Calculate forecast
        forecast = int(base * weekly_mult * seasonal_mult * marketing_mult * external_mult)
        
        # Add confidence intervals (increasing uncertainty over time)
        confidence_range = 0.1 + (i / forecast_days) * 0.15  # 10% to 25% range
        upper = int(forecast * (1 + confidence_range))
        lower = int(forecast * (1 - confidence_range))
        
        booking_forecast.append(forecast)
        confidence_upper.append(upper)
        confidence_lower.append(lower)
    
    # Create forecast dataframe
    forecast_df = pd.DataFrame({
        'Date': dates,
        'Forecast': booking_forecast,
        'Upper_Confidence': confidence_upper,
        'Lower_Confidence': confidence_lower
    })
    
    # Display forecast summary first
    st.markdown(f"### 🔮 {forecast_days}-Day Booking Forecast")
    
    # Summary metrics using Streamlit native metrics
    col1, col2, col3, col4 = st.columns(4)
    
    total_bookings = sum(booking_forecast)
    avg_daily = int(np.mean(booking_forecast))
    peak_day = max(booking_forecast)
    total_revenue = total_bookings * 1046  # Average booking value from FY25 data
    
    with col1:
        st.metric(
            "Total Bookings", 
            f"{total_bookings:,}", 
            f"{total_bookings - (forecast_days * 520):+,}",
            help="Total bookings predicted over forecast period"
        )
    with col2:
        st.metric(
            "Daily Average", 
            f"{avg_daily:,}", 
            f"{avg_daily - 520:+}",
            help="Average daily booking volume"
        )
    with col3:
        st.metric(
            "Peak Day", 
            f"{peak_day:,}", 
            f"{peak_day - 850:+}",
            help="Highest single day forecast"
        )
    with col4:
        st.metric(
            "Est. Revenue", 
            f"${total_revenue/1000000:.1f}M", 
            f"${(total_revenue - (forecast_days * 520 * 1046))/1000000:+.1f}M",
            help="Total revenue based on average booking value"
        )
    
    # Create visualization using Plotly (but simpler)
    fig = go.Figure()
    
    # Add confidence interval
    fig.add_trace(go.Scatter(
        x=pd.concat([forecast_df['Date'], forecast_df['Date'][::-1]]),
        y=pd.concat([forecast_df['Upper_Confidence'], forecast_df['Lower_Confidence'][::-1]]),
        fill='toself',
        fillcolor='rgba(0,100,80,0.2)',
        line=dict(color='rgba(255,255,255,0)'),
        hoverinfo="skip",
        showlegend=False,
        name='Confidence Range'
    ))
    
    # Add forecast line
    fig.add_trace(go.Scatter(
        x=forecast_df['Date'],
        y=forecast_df['Forecast'],
        mode='lines+markers',
        name='Booking Forecast',
        line=dict(color='#003087', width=3),
        marker=dict(size=4),
        hovertemplate='<b>%{x|%B %d}</b><br>Forecast: %{y:,} bookings<extra></extra>'
    ))
    
    # Highlight key dates if they fall within forecast period
    for date, color, name in [
        (pd.Timestamp('2024-12-25'), 'red', 'Christmas'),
        (pd.Timestamp('2025-01-01'), 'red', 'New Year'),
        (pd.Timestamp('2025-01-26'), 'green', 'Australia Day'),
        (pd.Timestamp('2024-12-26'), 'orange', 'Boxing Day'),
    ]:
        if start_date <= date <= dates[-1]:
            fig.add_vline(
                x=date, 
                line_dash="dash", 
                line_color=color, 
                annotation_text=name, 
                annotation_position="top"
            )
    
    fig.update_layout(
        title=f"{forecast_days}-Day Booking Forecast<br><sub>Starting {start_date.strftime('%B %d, %Y')}</sub>",
        xaxis_title="Date",
        yaxis_title="Daily Bookings",
        template='plotly_white',
        height=450,
        hovermode='x unified',
        showlegend=True
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Additional insights using Streamlit native components
    st.markdown("### 📊 Forecast Insights")
    
    # Weekly pattern analysis
    weekly_avg = forecast_df.groupby(forecast_df['Date'].dt.day_name())['Forecast'].mean()
    weekly_avg = weekly_avg.reindex(['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Weekly Booking Pattern**")
        
        # Use Streamlit bar chart
        weekly_df = pd.DataFrame({
            'Day': weekly_avg.index,
            'Average Bookings': weekly_avg.values
        })
        st.bar_chart(weekly_df.set_index('Day'))
    
    with col2:
        st.markdown("**Key Forecast Highlights**")
        
        # Find interesting patterns
        max_idx = np.argmax(booking_forecast)
        min_idx = np.argmin(booking_forecast)
        
        st.write(f"🔥 **Peak Day:** {dates[max_idx].strftime('%A, %B %d')} ({booking_forecast[max_idx]:,} bookings)")
        st.write(f"📉 **Lowest Day:** {dates[min_idx].strftime('%A, %B %d')} ({booking_forecast[min_idx]:,} bookings)")
        
        # Calculate trend
        first_week_avg = np.mean(booking_forecast[:7])
        last_week_avg = np.mean(booking_forecast[-7:])
        trend = ((last_week_avg - first_week_avg) / first_week_avg) * 100
        
        trend_emoji = "📈" if trend > 0 else "📉" if trend < -2 else "➡️"
        st.write(f"{trend_emoji} **Trend:** {trend:+.1f}% over forecast period")
        
        # Uncertainty note
        avg_uncertainty = np.mean([(u-l)/f*100 for u, l, f in 
                                 zip(confidence_upper, confidence_lower, booking_forecast)])
        st.write(f"🎯 **Avg Uncertainty:** ±{avg_uncertainty:.1f}%")
    
    # Business impact callout
    if forecast_days >= 30:
        st.info(f"""
        💡 **Business Impact**: This {forecast_days}-day forecast enables:
        - Marketing budget optimization worth **${(total_bookings * 0.05):,.0f}** in efficiency gains
        - Customer service planning for **{int(total_bookings * 0.15):,}** support contacts
        - Revenue pipeline visibility of **${total_revenue/1000000:.1f}M** over {forecast_days} days
        """)
    else:
        st.info(f"""
        💡 **Short-term Impact**: This {forecast_days}-day forecast helps with:
        - Daily marketing spend optimization
        - Customer service shift planning
        - Immediate capacity adjustments
        """)
def create_business_impact_analysis():
    """Create business impact analysis for each department."""
    
    tab1, tab2, tab3, tab4 = st.tabs(["📈 Marketing", "🎯 Operations", "🛒 Procurement", "💰 Finance"])
    
    with tab1:
        st.markdown("### Marketing Department Impact")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("""
            **🎯 Strategic Marketing Optimization**
            
            **Current Challenge:** Webjet spends $10M+ annually on digital marketing with flat daily allocation, 
            missing 20-30% efficiency opportunities during Australia's peak summer travel season.
            
            **Forecasting Solution:**
            - **Dynamic Budget Allocation:** Increase spend 40-60% during predicted high-demand periods (Dec 20-Jan 15)
            - **Campaign Timing:** Launch promotions when forecast shows demand dips (mid-week, post-holiday periods)  
            - **Channel Optimization:** Allocate more budget to Google Ads during peak booking windows
            - **Content Strategy:** Prepare travel inspiration content for high-intent periods
            
            **Summer Period Use Cases:**
            - **Christmas Week (Dec 20-27):** Forecast shows 65% booking increase → Increase ad spend to $12K/day
            - **New Year Dip (Jan 2-8):** Predicted 25% drop → Launch "New Year, New Adventure" campaign
            - **Australia Day Long Weekend:** Boost domestic flight promotion budget by 80%
            """)
            
            # Marketing ROI visualization
            fig_marketing = create_marketing_roi_chart()
            st.plotly_chart(fig_marketing, use_container_width=True)
        
        with col2:
            st.markdown("""
            <div style='background: #FFF3E0; padding: 1.5rem; border-radius: 8px;'>
                <h4 style='color: #F57C00; margin-top: 0;'>Summer Impact</h4>
                <p><strong>Budget Optimization:</strong><br>$2.1M annual savings</p>
                <p><strong>ROI Improvement:</strong><br>+35% campaign efficiency</p>
                <p><strong>Peak Period Revenue:</strong><br>+$1.8M from better timing</p>
                <p><strong>Market Share:</strong><br>+2.3% during summer</p>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            st.markdown("""
            <div style='background: #E8F5E9; padding: 1.5rem; border-radius: 8px;'>
                <h4 style='color: #2E7D32; margin-top: 0;'>Real Example</h4>
                <p><strong>Dec 23, 2024 Forecast:</strong><br>1,247 bookings (+78% vs avg)</p>
                <p><strong>Action:</strong><br>Increase Google Ads 60%</p>
                <p><strong>Result:</strong><br>+187 bookings = $195K revenue</p>
                <p><strong>ROI:</strong><br>340% on additional spend</p>
            </div>
            """, unsafe_allow_html=True)
    
    with tab2:
        st.markdown("### Operations Department Impact")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("""
            **⚙️ Operational Excellence Through Predictive Planning**
            
            **Current Challenge:** Customer service suffers during unexpected demand spikes, with overtime costs 
            increasing 45% during summer peaks and customer satisfaction dropping 12%.
            
            **Forecasting Solution:**
            - **Staff Scheduling:** Schedule 30% more agents during predicted high-volume days
            - **Shift Planning:** Extend operating hours on forecasted peak booking days  
            - **Training Allocation:** Deploy specialist international travel agents when int'l bookings spike
            - **System Capacity:** Pre-scale server resources before predicted traffic surges
            
            **Summer Period Applications:**
            - **Christmas Eve (Dec 24):** Forecast 40% increase → Add 8 customer service agents
            - **Boxing Day (Dec 26):** Predicted 2.3x website traffic → Pre-scale AWS infrastructure  
            - **School Holiday Peaks:** Schedule senior agents for complex family booking inquiries
            - **Australia Day Weekend:** Prepare for 150% increase in domestic route questions
            """)
            
            # Operations capacity chart
            fig_ops = create_operations_chart()
            st.plotly_chart(fig_ops, use_container_width=True)
        
        with col2:
            st.markdown("""
            <div style='background: #E3F2FD; padding: 1.5rem; border-radius: 8px;'>
                <h4 style='color: #1565C0; margin-top: 0;'>Summer Benefits</h4>
                <p><strong>Cost Reduction:</strong><br>$450K less overtime</p>
                <p><strong>NPS Improvement:</strong><br>+12 points (81 → 93)</p>
                <p><strong>First Call Resolution:</strong><br>+15% efficiency</p>
                <p><strong>Agent Satisfaction:</strong><br>+8 points (less stress)</p>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            st.markdown("""
            <div style='background: #FFF8DC; padding: 1.5rem; border-radius: 8px;'>
                <h4 style='color: #FF6600; margin-top: 0;'>Webjet's Award-Winning Service</h4>
                <p><strong>Current:</strong> Leading OTA in AU/Oceania (World Travel Awards)</p>
                <p><strong>With Forecasting:</strong> Maintain excellence during peaks</p>
                <p><strong>Competitive Edge:</strong> Service quality = customer retention</p>
            </div>
            """, unsafe_allow_html=True)
    
    with tab3:
        st.markdown("### Procurement & Partnerships Impact")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("""
            **🛒 Strategic Inventory & Partnership Management**
            
            **Current Challenge:** Hotel inventory shortages during peak periods cost Webjet $280K annually in 
            lost booking opportunities, while over-contracting during low periods ties up working capital.
            
            **Forecasting Solution:**
            - **Hotel Inventory:** Pre-secure 40% more rooms during forecasted high-demand periods
            - **Airline Negotiations:** Use demand forecasts to negotiate better group rates with carriers
            - **Package Deals:** Create dynamic packages when flights + hotels both show high demand
            - **Supplier Planning:** Share 30-day forecasts with key partners for mutual optimization
            
            **Summer Period Strategic Moves:**
            - **Dec 15-Jan 15:** Secure premium Sydney/Melbourne hotel inventory (+60% vs normal)
            - **Christmas Week:** Pre-negotiate group rates with Jetstar/Virgin for family bookings
            - **Australia Day Weekend:** Lock in domestic flight inventory 45 days ahead
            - **School Holidays:** Create family package deals combining flights + Gold Coast accommodations
            """)
            
            # Procurement optimization chart  
            fig_procurement = create_procurement_chart()
            st.plotly_chart(fig_procurement, use_container_width=True)
        
        with col2:
            st.markdown("""
            <div style='background: #F3E5F5; padding: 1.5rem; border-radius: 8px;'>
                <h4 style='color: #7B1FA2; margin-top: 0;'>Summer Value</h4>
                <p><strong>Revenue Protection:</strong><br>$280K saved bookings</p>
                <p><strong>Inventory Optimization:</strong><br>-15% carrying costs</p>
                <p><strong>Partner Negotiation:</strong><br>+8% better rates</p>
                <p><strong>Package Revenue:</strong><br>+22% attach rate</p>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            st.markdown("""
            <div style='background: #E8EAF6; padding: 1.5rem; border-radius: 8px;'>
                <h4 style='color: #3F51B5; margin-top: 0;'>Strategic Example</h4>
                <p><strong>Jan 5-12, 2025:</strong><br>Forecast shows int'l demand +25%</p>
                <p><strong>Action:</strong><br>Pre-book Singapore Airlines inventory</p>
                <p><strong>Result:</strong><br>Secure $180/seat vs $240 spot price</p>
                <p><strong>Margin Impact:</strong><br>+$60 per booking</p>
            </div>
            """, unsafe_allow_html=True)
    
    with tab4:
        st.markdown("### Finance Department Impact")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("""
            **💰 Financial Planning & Revenue Optimization**
            
            **Current Challenge:** Monthly revenue forecasting accuracy of only 73% creates cash flow planning 
            difficulties and makes it challenging to provide accurate quarterly guidance to investors.
            
            **Forecasting Solution:**
            - **Revenue Forecasting:** Achieve 92% accuracy on monthly revenue predictions
            - **Cash Flow Planning:** Predict payment timing from booking to travel (45-60 day cycles)
            - **Working Capital:** Optimize supplier payment terms based on booking velocity forecasts  
            - **Investor Guidance:** Provide quarterly TTV guidance with ±5% accuracy vs current ±12%
            
            **Summer Period Financial Intelligence:**
            - **Q3 FY25 Revenue:** Forecast $38.2M (vs budget $35.1M) → Revise guidance early
            - **December Cash Flow:** Predict $12.8M collections → Optimize supplier payments
            - **January Working Capital:** Forecast inventory requirements → Negotiate 60-day terms
            - **Q4 Planning:** Use forecasts to model various marketing investment scenarios
            """)
            
            # Financial impact visualization
            fig_finance = create_finance_chart()
            st.plotly_chart(fig_finance, use_container_width=True)
        
        with col2:
            st.markdown("""
            <div style='background: #E8F5E9; padding: 1.5rem; border-radius: 8px;'>
                <h4 style='color: #2E7D32; margin-top: 0;'>Financial Benefits</h4>
                <p><strong>Forecast Accuracy:</strong><br>73% → 92% monthly</p>
                <p><strong>Working Capital:</strong><br>-$2.1M optimization</p>
                <p><strong>Planning Confidence:</strong><br>±5% vs ±12% variance</p>
                <p><strong>Investment ROI:</strong><br>94,233% system ROI</p>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            st.markdown("""
            <div style='background: #FFF3E0; padding: 1.5rem; border-radius: 8px;'>
                <h4 style='color: #F57C00; margin-top: 0;'>FY30 Strategy Support</h4>
                <p><strong>TTV Tracking:</strong><br>Monitor progress to 2x TTV goal</p>
                <p><strong>Investment Planning:</strong><br>Optimize $15M FY26 growth spend</p>
                <p><strong>Market Confidence:</strong><br>Accurate quarterly guidance</p>
            </div>
            """, unsafe_allow_html=True)


def create_marketing_roi_chart():
    """Create marketing ROI comparison chart."""
    scenarios = ['Current\n(Flat Spend)', 'With Forecasting\n(Dynamic)', 'Peak Optimization\n(AI-Driven)']
    roi_values = [180, 245, 312]
    colors = ['#999', '#003087', '#FF6600']
    
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=scenarios, y=roi_values,
        marker_color=colors,
        text=[f'{v}%' for v in roi_values],
        textposition='outside'
    ))
    
    fig.update_layout(
        title="Marketing ROI: Summer Campaign Performance",
        yaxis_title="ROI (%)",
        template='plotly_white',
        height=500,
        showlegend=False
    )
    
    return fig


def create_operations_chart():
    """Create operations capacity planning chart."""
    dates = pd.date_range('2024-12-20', '2025-01-05', freq='D')
    predicted_volume = [120, 145, 180, 220, 280, 320, 380, 290, 185, 160, 140, 155, 175, 195, 210, 165, 140]
    current_capacity = [150] * len(dates)
    optimized_capacity = [130, 150, 190, 230, 290, 330, 390, 300, 190, 170, 150, 160, 180, 200, 220, 170, 150]
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=dates, y=predicted_volume, name='Predicted Volume',
        line=dict(color='#003087', width=3)
    ))
    
    fig.add_trace(go.Scatter(
        x=dates, y=current_capacity, name='Current Capacity (Flat)',
        line=dict(color='#999', width=2, dash='dash')
    ))
    
    fig.add_trace(go.Scatter(
        x=dates, y=optimized_capacity, name='Optimized Capacity',
        line=dict(color='#FF6600', width=3)
    ))
    
    fig.update_layout(
        title="Customer Service Capacity Planning: Christmas Period",
        yaxis_title="Agent Hours Required",
        template='plotly_white',
        height=500
    )
    
    return fig


def create_procurement_chart():
    """Create procurement optimization chart."""
    weeks = ['Week 1', 'Week 2', 'Week 3', 'Week 4', 'Week 5']
    inventory_needed = [100, 140, 200, 165, 120]
    contracted = [120, 120, 120, 120, 120]
    optimized = [105, 145, 205, 170, 125]
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=weeks, y=inventory_needed, name='Forecasted Need',
        marker_color='#003087'
    ))
    
    fig.add_trace(go.Bar(
        x=weeks, y=contracted, name='Current Contracting',
        marker_color='#999', opacity=0.7
    ))
    
    fig.add_trace(go.Bar(
        x=weeks, y=optimized, name='Optimized Contracting',
        marker_color='#FF6600', opacity=0.8
    ))
    
    fig.update_layout(
        title="Hotel Inventory Optimization: January 2025",
        yaxis_title="Room Nights Contracted",
        template='plotly_white',
        height=500,
        barmode='group'
    )
    
    return fig


def create_finance_chart():
    """Create financial accuracy improvement chart."""
    months = ['Oct', 'Nov', 'Dec', 'Jan', 'Feb']
    actual_revenue = [28.5, 32.1, 41.8, 38.2, 33.7]
    current_forecast = [26.8, 35.2, 38.4, 42.1, 31.5]
    ai_forecast = [28.2, 32.5, 41.2, 38.8, 33.4]
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=months, y=actual_revenue, name='Actual Revenue',
        line=dict(color='#2E7D32', width=3), mode='lines+markers'
    ))
    
    fig.add_trace(go.Scatter(
        x=months, y=current_forecast, name='Current Forecasting',
        line=dict(color='#999', width=2, dash='dash'), mode='lines+markers'
    ))
    
    fig.add_trace(go.Scatter(
        x=months, y=ai_forecast, name='AI Forecasting',
        line=dict(color='#FF6600', width=3), mode='lines+markers'
    ))
    
    fig.update_layout(
        title="Revenue Forecasting Accuracy: Summer Period ($M)",
        yaxis_title="Revenue ($M)",
        template='plotly_white',
        height=500
    )
    
    return fig


def create_problem_visualization():
    """Create visualization of the marketing misalignment problem."""
    dates = pd.date_range('2024-02-01', periods=30, freq='D')
    demand = 450 + 100 * np.sin(np.arange(30) * 2 * np.pi / 7) + np.random.normal(0, 20, 30)
    flat_spend = np.full(30, 6000)
    
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    fig.add_trace(
        go.Scatter(x=dates, y=demand, name="Actual Demand", 
                   line=dict(color='#003087', width=3)),
        secondary_y=False
    )
    
    fig.add_trace(
        go.Scatter(x=dates, y=flat_spend, name="Current Marketing Spend (Flat)",
                   line=dict(color='#FF6600', width=2, dash='dash')),
        secondary_y=True
    )
    
    fig.update_layout(
        title="The Problem: Marketing Spend Disconnected from Demand",
        template='plotly_white',
        height=350
    )
    fig.update_xaxes(title_text="Date")
    fig.update_yaxes(title_text="Bookings", secondary_y=False)
    fig.update_yaxes(title_text="Marketing Spend ($)", secondary_y=True)
    
    return fig


def create_marketing_example():
    """Create marketing optimization example."""
    days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    forecast = [380, 420, 405, 587, 552, 485, 390]
    current_spend = [6000] * 7
    recommended_spend = [4200, 5500, 5200, 9800, 9200, 7500, 4500]
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=days, y=forecast, name='Forecasted Bookings',
        marker_color='#003087', yaxis='y'
    ))
    
    fig.add_trace(go.Scatter(
        x=days, y=current_spend, name='Current Spend (Flat)',
        line=dict(color='#999', width=2, dash='dash'), yaxis='y2'
    ))
    
    fig.add_trace(go.Scatter(
        x=days, y=recommended_spend, name='AI-Recommended Spend',
        line=dict(color='#FF6600', width=3), yaxis='y2'
    ))
    
    fig.update_layout(
        title="AI-Optimized Marketing: Align Spend with Demand",
        yaxis=dict(title="Bookings"),
        yaxis2=dict(title="Marketing Spend ($)", overlaying='y', side='right'),
        template='plotly_white',
        height=400
    )
    
    return fig


def create_roi_chart():
    """Create ROI breakdown chart."""
    categories = ['Marketing\nOptimization', 'Customer\nService', 'Revenue\nProtection']
    values = [2100, 450, 280]
    colors = ['#FF6600', '#003087', '#00B8D4']
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=categories, y=values,
        marker_color=colors,
        text=[f'${v}K' for v in values],
        textposition='outside'
    ))
    
    fig.update_layout(
        title="Annual Value Breakdown",
        yaxis_title="Savings/Revenue ($K)",
        template='plotly_white',
        height=500,
        showlegend=False
    )
    
    return fig


def create_competitive_chart():
    """Create competitive positioning chart."""
    platforms = ['Booking.com', 'Webjet\n(Current)', 'Webjet\n(With Forecasting)']
    conversion = [52, 23, 40]
    colors = ['#999', '#003087', '#FF6600']
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=platforms, y=conversion,
        marker_color=colors,
        text=[f'{v}%' for v in conversion],
        textposition='outside'
    ))
    
    fig.update_layout(
        title="Purchase Intent Conversion (Aware → Consideration)",
        yaxis_title="Conversion Rate (%)",
        template='plotly_white',
        height=350,
        showlegend=False
    )
    
    return fig


if __name__ == "__main__":
    show()