import streamlit as st
from utils import load_from_session, save_to_session
import importlib

# Page configuration
st.set_page_config(
    page_title="Webjet Flight Booking Forecasting",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional styling
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        margin-bottom: 2rem;
    }
    .stat-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .stat-label {
        font-size: 0.9rem;
        color: #666;
        margin-bottom: 0.25rem;
    }
    .stat-value {
        font-size: 1.5rem;
        font-weight: bold;
        color: #1f77b4;
    }
    .progress-indicator {
        font-size: 1.1rem;
        font-weight: 600;
        color: #1f77b4;
        padding: 0.5rem;
        background-color: #e8f4f8;
        border-radius: 0.5rem;
        text-align: center;
        margin-bottom: 1rem;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if 'current_page' not in st.session_state:
    st.session_state.current_page = "Data Acquisition"

# Page mapping
PAGES = {
    "Data Acquisition": "page1_data_acquisition",
    "Exploratory Data Analysis": "page2_eda",
    "Preprocessing & Feature Engineering": "page3_preprocessing",
    "Model Training": "page4_modeling",
    "Forecasting & Validation": "page5_forecasting",
    "Business Insights & Deployment": "page6_deployment",
    "MLOps & Monitoring": "page7_mlops"
}

# Sidebar
with st.sidebar:
    # App logo and title
    st.markdown("""
        <div style='text-align: center; padding: 1rem 0;'>
            <h1 style='color: #1f77b4; margin: 0;'>✈️ Webjet</h1>
            <p style='color: #666; margin: 0.25rem 0 0 0;'>Flight Booking Forecasting</p>
        </div>
        <hr style='margin: 1rem 0;'>
    """, unsafe_allow_html=True)
    
    # Progress indicator
    page_num = list(PAGES.keys()).index(st.session_state.current_page) + 1
    st.markdown(f"""
        <div class='progress-indicator'>
            Step {page_num} / 7
        </div>
    """, unsafe_allow_html=True)
    
    # Page navigation
    st.subheader("Navigation")
    selected_page = st.radio(
        "Select a step:",
        options=list(PAGES.keys()),
        index=page_num - 1,
        label_visibility="collapsed"
    )
    
    st.session_state.current_page = selected_page
    
    st.markdown("<hr style='margin: 1.5rem 0;'>", unsafe_allow_html=True)
    
    # Quick stats summary
    st.subheader("Quick Stats")
    data = load_from_session('raw_data')
    
    if data is not None:
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown(f"""
                <div class='stat-card'>
                    <div class='stat-label'>Date Range</div>
                    <div class='stat-value' style='font-size: 0.9rem;'>
                        {data['date'].min().strftime('%Y-%m-%d')}<br>to<br>{data['date'].max().strftime('%Y-%m-%d')}
                    </div>
                </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
                <div class='stat-card'>
                    <div class='stat-label'>Observations</div>
                    <div class='stat-value'>{len(data):,}</div>
                </div>
            """, unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Additional stats
        avg_bookings = data['bookings'].mean()
        total_bookings = data['bookings'].sum()
        
        st.markdown(f"""
            <div class='stat-card'>
                <div class='stat-label'>Avg Daily Bookings</div>
                <div class='stat-value'>{avg_bookings:.0f}</div>
            </div>
        """, unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        st.markdown(f"""
            <div class='stat-card'>
                <div class='stat-label'>Total Bookings</div>
                <div class='stat-value'>{total_bookings:,.0f}</div>
            </div>
        """, unsafe_allow_html=True)
    else:
        st.info("📊 Load data to see statistics")
    
    st.markdown("<hr style='margin: 1.5rem 0;'>", unsafe_allow_html=True)
    
    # Reset button
    if st.button("🔄 Reset All", use_container_width=True, type="secondary"):
        for key in list(st.session_state.keys()):
            if key != 'current_page':
                del st.session_state[key]
        st.success("✅ All data cleared!")
        st.rerun()
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Footer
    st.markdown("""
        <div style='text-align: center; color: #999; font-size: 0.8rem; margin-top: 2rem;'>
            <hr style='margin-bottom: 1rem;'>
            Built with Streamlit<br>
            © 2024 Webjet Analytics
        </div>
    """, unsafe_allow_html=True)

# Main content area
try:
    # Dynamically import and run the selected page
    page_module = importlib.import_module(PAGES[selected_page])
    page_module.show()
    
except Exception as e:
    st.error(f"Error loading page: {str(e)}")
    st.info("Please ensure all page modules are in the same directory as main.py")
