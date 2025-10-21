import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime
from utils.data_generator import generate_synthetic_data
from utils.utils import (
    save_to_session,
    load_from_session,
    validate_data_quality,
    detect_outliers,
    get_status_badge
)


def show():
    """Display the Data Acquisition page."""
    st.info("👈 Open sidebar to see the process steps")
    # Header
    col1,col2,col3 = st.columns([1,4,1])
    with col1:
        st.write("")
        
    with col2:
        # Header
        with st.container():
            st.markdown("""
                <div style='background: linear-gradient(135deg, #e14747 0%, #e14747 100%); 
                            padding: 1rem 1rem; border-radius: 12px; margin-bottom: 1rem; text-align: center;'>
                    <h1 style='color: white; font-size: 3rem; margin: 0; font-weight: 500;'>
                        Step 1: 📥 Data Acquisition
                    </h1>

                </div>
            """, unsafe_allow_html=True)
            st.markdown("""
            - In this step, you will either generate synthetic flight booking data or upload your own historical bookings data in CSV format.
            - The synthetic data simulates real-world patterns including seasonality, holidays, marketing effects, competitor pricing, and weather disruptions.
            - After loading the data, you will see a preview, summary statistics, and quick visualizations to understand its structure and quality.
                    """, unsafe_allow_html=True)
    st.markdown("---")
    
    # Data Source Selector
    st.subheader("1️⃣ Select Data Source")
    
    data_source = st.radio(
        "Choose data source:",
        options=["Generate Synthetic Data", "Upload CSV File"],
        index=0,
        horizontal=True
    )
    
    st.markdown("---")
    
    # Generate or Upload Data
    if data_source == "Generate Synthetic Data":
        st.subheader("2️⃣ Configure Data Generation")
        
        col1, col2, col3 = st.columns([2, 2, 1])
        
        with col1:
            start_date = st.date_input(
                "Start Date",
                value=datetime(2022, 10, 20),
                min_value=datetime(2020, 1, 1),
                max_value=datetime(2022, 10, 20)
            )
        
        with col2:
            end_date = st.date_input(
                "End Date",
                value=datetime(2025, 10, 19),
                min_value=datetime(2020, 1, 1),
                max_value=datetime(2025, 10, 19)
            )
        
        with col3:
            seed = st.number_input("Random Seed", value=42, min_value=0, max_value=9999)
        
        if start_date >= end_date:
            st.error("❌ End date must be after start date")
        else:
            st.info(f"📅 Will generate **{(end_date - start_date).days + 1}** days of data")
        
        if st.button("🎲 Generate Data", type="primary", use_container_width=True):
            with st.spinner("Generating synthetic data..."):
                try:
                    df = generate_synthetic_data(
                        start_date=start_date.strftime('%Y-%m-%d'),
                        end_date=end_date.strftime('%Y-%m-%d'),
                        seed=seed
                    )
                    
                    save_to_session('data', df)
                    save_to_session('raw_data', df.copy())
                    save_to_session('data_source', 'synthetic')
                    
                    st.success(f"✅ Successfully generated {len(df):,} rows of data!")
                    st.rerun()
                    
                except Exception as e:
                    st.error(f"❌ Error generating data: {str(e)}")
    
    else:  # Upload CSV
        st.subheader("2️⃣ Upload CSV File")
        
        uploaded_file = st.file_uploader(
            "Choose a CSV file",
            type=['csv'],
            help="File must contain 'date' and 'bookings' columns"
        )
        
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                
                if 'date' in df.columns:
                    df['date'] = pd.to_datetime(df['date'])
                
                save_to_session('data', df)
                save_to_session('raw_data', df.copy())
                save_to_session('data_source', 'uploaded')
                
                st.success(f"✅ Successfully loaded {len(df):,} rows from file!")
                st.rerun()
                
            except Exception as e:
                st.error(f"❌ Error loading file: {str(e)}")
    
    st.markdown("---")
    
    # Display data if loaded
    data = load_from_session('data')
    
    if data is not None:
        # Data Preview
        st.subheader("3️⃣ Data Preview (First 20 Rows)")
        st.dataframe(data.head(20), use_container_width=True, height=400)
        
        csv = data.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Data as CSV",
            data=csv,
            file_name=f"webjet_bookings_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )
        
        st.markdown("---")
        
        # Summary Statistics
        st.subheader("4️⃣ Summary Statistics")
        
        numeric_cols = data.select_dtypes(include=['int64', 'float64']).columns
        stats_df = data[numeric_cols].describe().T
        stats_df['missing'] = data[numeric_cols].isnull().sum()
        stats_df['missing %'] = (data[numeric_cols].isnull().sum() / len(data) * 100).round(2)
        
        st.dataframe(
            stats_df.style.format({
                'count': '{:.0f}',
                'mean': '{:.2f}',
                'std': '{:.2f}',
                'min': '{:.2f}',
                '25%': '{:.2f}',
                '50%': '{:.2f}',
                '75%': '{:.2f}',
                'max': '{:.2f}',
                'missing': '{:.0f}',
                'missing %': '{:.2f}'
            }),
            use_container_width=True
        )
        
        st.markdown("---")
        
        # Quick Visualization
        st.subheader("5️⃣ Quick Visualization")
        
        fig = create_time_series_plot(data)
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("---")
        
        # Data Quality Checks
        st.subheader("6️⃣ Data Quality Checks")
        
        validation_results = validate_data_quality(data)
        outliers = detect_outliers(data, 'bookings', method='zscore', threshold=3)
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            if validation_results['checks'].get('no_duplicate_dates', True):
                st.markdown(get_status_badge('success', '✓ No Duplicate Dates'), unsafe_allow_html=True)
            else:
                dupes = validation_results['summary'].get('duplicate_dates', 0)
                st.markdown(get_status_badge('error', f'✗ {dupes} Duplicate Dates'), unsafe_allow_html=True)
        
        with col2:
            if validation_results['checks'].get('continuous_dates', True):
                st.markdown(get_status_badge('success', '✓ Continuous Date Range'), unsafe_allow_html=True)
            else:
                missing = validation_results['summary'].get('missing_days', 0)
                st.markdown(get_status_badge('warning', f'⚠ {missing} Days Missing'), unsafe_allow_html=True)
        
        with col3:
            total_missing = data.isnull().sum().sum()
            if total_missing == 0:
                st.markdown(get_status_badge('success', '✓ No Missing Values'), unsafe_allow_html=True)
            else:
                st.markdown(get_status_badge('warning', f'⚠ {total_missing} Missing Values'), unsafe_allow_html=True)
        
        with col4:
            if len(outliers) == 0:
                st.markdown(get_status_badge('success', '✓ No Outliers'), unsafe_allow_html=True)
            else:
                st.markdown(get_status_badge('warning', f'⚠ {len(outliers)} Outliers'), unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Action Buttons
        st.subheader("7️⃣ Next Steps")
        
        col1, col2, col3 = st.columns([1, 1, 1])
        
        with col1:
            if st.button("💾 Save Data to Session", use_container_width=True):
                save_to_session('data', data)
                st.success("✅ Data saved to session!")
        
        with col3:
            if st.button("➡️ Proceed to EDA", type="primary", use_container_width=True):
                st.session_state.current_page = "Step 2 - Exploratory Data Analysis"
                st.rerun()
        
    else:
        st.info("👆 Please generate synthetic data or upload a CSV file to begin.")


# Helper Functions
def create_time_series_plot(data):
    """Create interactive time series plot."""
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=data['date'],
        y=data['bookings'],
        mode='lines',
        name='Bookings',
        line=dict(color='#1f77b4', width=1.5),
        hovertemplate='<b>Date:</b> %{x|%Y-%m-%d}<br><b>Bookings:</b> %{y:,.0f}<extra></extra>'
    ))
    
    fig.update_layout(
        title='Daily Flight Bookings Over Time',
        xaxis_title='Date',
        yaxis_title='Number of Bookings',
        hovermode='x unified',
        height=400,
        template='plotly_white',
        showlegend=False
    )
    
    return fig


if __name__ == "__main__":
    show()