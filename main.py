import streamlit as st
from utils.utils import load_from_session, save_to_session
import importlib

# Page configuration
st.set_page_config(
    page_title="Webjet Flight Booking Forecasting",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state defaults
if 'current_page' not in st.session_state:
    st.session_state.current_page = "Business Case Overview"

# Prevent stale data crashes
if st.sidebar.button("🔄 Reset & Restart"):
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    st.rerun()

# Page mapping with logical grouping
BUSINESS_PAGES = {
    "Business Case Overview": "page0_landing"
}

ML_PAGES = {
    "Step 1 - Data Acquisition": "page1_data_acquisition", 
    "Step 2 - Exploratory Data Analysis": "page2_eda",
    "Step 3 - Preprocessing & Feature Engineering": "page3_preprocessing",
    "Step 4 - Model Training": "page4_modeling",
    "Step 5 - Forecasting & Validation": "page5_forecasting",
    "Step 6 - Business Insights & Deployment": "page6_deployment",
    "Step 7 - MLOps & Monitoring": "page7_mlops"
}

# Combined pages for navigation logic
ALL_PAGES = {**BUSINESS_PAGES, **ML_PAGES}

# Sidebar
with st.sidebar:
    # App logo and title
    st.markdown("""
    <style>
    .main-header {
        font-size: 1.5rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 1rem;
    }
    .section-header {
        font-size: 1.1rem;
        font-weight: bold;
        color: #FF6600;
        margin: 1.5rem 0 0.5rem 0;
        border-bottom: 2px solid #FF6600;
        padding-bottom: 0.3rem;
    }
    .progress-indicator {
        font-size: 0.9rem;
        color: #666;
        text-align: center;
        margin-bottom: 1rem;            
    }
    .audience-tag {
        background: linear-gradient(135deg, #FF6600 0%, #FF8C42 100%);
        color: white;
        padding: 0.3rem 0.6rem;
        border-radius: 15px;
        font-size: 0.8rem;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .ml-tag {
        background: linear-gradient(135deg, #007bff 0%, #0056b3 100%);
        color: white;
        padding: 0.3rem 0.6rem;
        border-radius: 15px;
        font-size: 0.8rem;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    </style>
    """, unsafe_allow_html=True)
    
    st.markdown("<div class='main-header'>Webjet Flight Booking Forecasting</div>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([2, 1, 2])
    with col2:
        try:
            st.image("docs/logo.jpeg", use_column_width=True)
        except:
            st.markdown("### ✈️")
    
    # Progress indicator based on current section
    if st.session_state.current_page in BUSINESS_PAGES:
        page_num = list(BUSINESS_PAGES.keys()).index(st.session_state.current_page) + 1
        total_pages = len(BUSINESS_PAGES)
        section = "Business Overview"
    else:
        page_num = list(ML_PAGES.keys()).index(st.session_state.current_page) + 1
        total_pages = len(ML_PAGES)
        section = "ML Walkthrough"
    
    st.markdown(f"""
        <div class='progress-indicator'>
            {section}: {page_num} / {total_pages}
        </div>
    """, unsafe_allow_html=True)
    
    # Business Overview Section
    st.subheader("🎯 Business Stakeholders ",divider="rainbow")
    for page_name in BUSINESS_PAGES.keys():
        is_selected = st.session_state.current_page == page_name
        if st.button(
            page_name, 
            use_container_width=True,
            type="primary" if is_selected else "secondary"
        ):
            st.session_state.current_page = page_name
            st.rerun()
    
    # ML System Walkthrough Section
    st.subheader("ML Engineers",divider="rainbow")
    for page_name in ML_PAGES.keys():
        is_selected = st.session_state.current_page == page_name
        if st.button(
            page_name, 
            use_container_width=True,
            type="primary" if is_selected else "secondary"
        ):
            st.session_state.current_page = page_name
            st.rerun()
    
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
            Built by Conor Curley (<a href='https://www.conorcurley.com' target='_blank'>Website</a> | <a href='https://www.linkedin.com/in/conor-curley/' target='_blank'>LinkedIn</a>)
        </div>
    """, unsafe_allow_html=True)

# Main content area
try:
    page_module = importlib.import_module(f".{ALL_PAGES[st.session_state.current_page]}", package="page_files")
    page_module.show()
except ImportError as e:
    st.error(f"⚠️ Page module not found: {ALL_PAGES[st.session_state.current_page]}.py")
    st.info("Please ensure all page files are in the page_files/ directory.")
    if st.checkbox("Show technical details"):
        st.exception(e)
except Exception as e:
    st.error(f"❌ An error occurred: {str(e)}")
    st.warning("Try refreshing the page or clicking 'Reset All' in the sidebar.")
    if st.checkbox("Show technical details"):
        st.exception(e)