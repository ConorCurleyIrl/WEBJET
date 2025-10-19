# Webjet Flight Booking Demand Forecasting Application

A production-ready Streamlit application for end-to-end time series forecasting of flight booking demand.

## 🚀 Quick Start

### Installation

1. **Clone or create project directory:**
```bash
mkdir webjet_forecasting
cd webjet_forecasting
```

2. **Create virtual environment:**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

### Running the Application

```bash
streamlit run main.py
```

The app will open in your browser at `http://localhost:8501`

## 📁 Project Structure

```
webjet_forecasting/
├── main.py                          # Main Streamlit app with navigation
├── page1_data_acquisition.py       # ✅ Data loading & quality checks
├── page2_eda.py                     # 🚧 Exploratory data analysis
├── page3_preprocessing.py           # 🚧 Feature engineering
├── page4_modeling.py                # 🚧 Model training
├── page5_forecasting.py             # 🚧 Forecast generation
├── page6_deployment.py              # 🚧 Business insights
├── page7_mlops.py                   # 🚧 Monitoring & maintenance
├── utils.py                         # Shared utility functions
├── data_generator.py                # Synthetic data generation
├── requirements.txt                 # Python dependencies
└── README.md                        # This file
```

## ✨ Features (Phase 1 - Complete)

### Main Application (`main.py`)
- ✅ Clean, professional sidebar navigation
- ✅ Progress indicator (Step X/7)
- ✅ Quick stats summary panel
- ✅ Reset all functionality
- ✅ Dynamic page loading
- ✅ Custom CSS styling

### Data Generator (`data_generator.py`)
- ✅ Realistic daily booking data (3+ years)
- ✅ Multiple seasonality patterns:
  - Weekly (Thu/Fri business travel peaks)
  - Yearly (Dec/Jan holiday peaks)
- ✅ COVID-19 impact simulation
- ✅ School holiday effects (AU/NZ calendar)
- ✅ Exogenous variables:
  - Marketing spend
  - Competitor price index
  - Weather disruption index
- ✅ Configurable outliers and missing values
- ✅ Reproducible (seed-based)

### Page 1: Data Acquisition
- ✅ Generate synthetic data with custom date ranges
- ✅ Upload CSV files
- ✅ Data preview (first 20 rows)
- ✅ Comprehensive summary statistics
- ✅ Quick visualization (time series plot)
- ✅ Data quality checks:
  - Duplicate dates detection
  - Date continuity validation
  - Missing value analysis
  - Outlier detection
  - Data type validation
- ✅ Download processed data
- ✅ Navigation to next page

### Utilities (`utils.py`)
- ✅ Session state management
- ✅ Data quality validation
- ✅ Outlier detection (Z-score & IQR methods)
- ✅ Number formatting utilities
- ✅ HTML component generators
- ✅ Statistical calculations
- ✅ Data export functions

## 🎯 Business Context

**Use Case:** Webjet B2C flight booking demand forecasting

**Objectives:**
- Optimize $10M+ annual digital marketing spend
- Negotiate better airline partnerships based on volume forecasts
- Staff customer service appropriately (5% inquiry rate)
- Enable dynamic pricing strategies

**Success Metrics:**
- Forecast accuracy: MAPE < 15%
- Business impact: 20% reduction in marketing waste
- 15% reduction in customer service overtime

## 📊 Data Schema

| Column | Type | Description |
|--------|------|-------------|
| `date` | datetime | Daily observation date |
| `bookings` | int | Total flight bookings for the day |
| `day_of_week` | int | Day of week (0=Monday, 6=Sunday) |
| `marketing_spend` | float | Daily marketing expenditure ($) |
| `competitor_price_index` | float | Competitor pricing index (100=baseline) |
| `is_holiday` | bool | Public holiday indicator |
| `is_school_holiday` | bool | School holiday period indicator |
| `weather_disruption_index` | float | Weather impact score (0-10) |

## 🔄 Development Phases

### ✅ Phase 1: Foundation (COMPLETE)
- Main app structure
- Data generation
- Data acquisition page
- Utility functions
- Placeholder pages

### 🚧 Phase 2: EDA & Preprocessing (NEXT)
- Page 2: Exploratory Data Analysis
  - Time series decomposition
  - Seasonality analysis
  - Outlier visualization
  - Correlation analysis
  - Stationarity tests
- Page 3: Preprocessing & Feature Engineering
  - Train/test split
  - Temporal features
  - Lag features
  - Rolling statistics
  - Transformations

### 🔜 Phase 3: Modeling & Forecasting
- Page 4: Model Training
  - ARIMA/SARIMA
  - Prophet
  - XGBoost, Random Forest, LightGBM
  - Ensemble methods
  - Model comparison
- Page 5: Forecasting
  - Generate forecasts
  - Uncertainty quantification
  - Backtesting
  - Scenario analysis

### 🔜 Phase 4: Deployment & MLOps
- Page 6: Business Insights
  - Marketing optimization
  - Staffing recommendations
  - Revenue forecasts
  - What-if simulator
- Page 7: MLOps
  - Performance monitoring
  - Drift detection
  - Retraining triggers
  - A/B testing

## 🛠️ Technical Stack

- **Framework:** Streamlit 1.31.0
- **Data Processing:** Pandas, NumPy
- **Visualization:** Plotly, Matplotlib, Seaborn
- **Statistical Models:** Statsmodels, Prophet
- **ML Models:** Scikit-learn, XGBoost, LightGBM
- **Time Series:** scipy, statsmodels

## 📈 Usage Examples

### Generate Synthetic Data

```python
from data_generator import generate_synthetic_data

# Generate 3 years of data
df = generate_synthetic_data(
    start_date='2021-01-01',
    end_date='2024-01-31',
    seed=42
)

print(f"Generated {len(df)} days of booking data")
print(df.head())
```

### Validate Data Quality

```python
from utils import validate_data_quality

results = validate_data_quality(df)

if results['is_valid']:
    print("✓ Data passed all validation checks")
else:
    print("✗ Issues found:")
    for issue in results['issues']:
        print(f"  - {issue}")
```

### Detect Outliers

```python
from utils import detect_outliers

# Z-score method
outliers_z = detect_outliers(df, 'bookings', method='zscore', threshold=3)
print(f"Found {len(outliers_z)} outliers using Z-score")

# IQR method
outliers_iqr = detect_outliers(df, 'bookings', method='iqr', threshold=1.5)
print(f"Found {len(outliers_iqr)} outliers using IQR")
```

## 🎨 Customization

### Modify Data Generation Parameters

Edit `data_generator.py` to customize:
- Seasonality patterns
- Trend strength
- COVID impact period
- Holiday effects
- Noise level
- Outlier frequency

### Change Styling

Modify the CSS in `main.py`:
```python
st.markdown("""
    <style>
    .main-header {
        color: #YOUR_COLOR;
        # ... customize
    }
    </style>
""", unsafe_allow_html=True)
```

## 🐛 Troubleshooting

### Import Errors

If you see module import errors, ensure all page files are in the same directory:
```bash
ls -la *.py
# Should show: main.py, page1_data_acquisition.py, page2_eda.py, etc.
```

### Data Not Persisting

Check session state:
```python
# In Streamlit sidebar
st.write(st.session_state.keys())
```

### Missing Dependencies

Reinstall requirements:
```bash
pip install -r requirements.txt --force-reinstall
```

## 📝 Development Notes

### Session State Keys

| Key | Type | Description |
|-----|------|-------------|
| `raw_data` | DataFrame | Original loaded data |
| `data_source` | str | 'synthetic' or 'uploaded' |
| `generation_params` | dict | Data generation parameters |
| `current_page` | str | Active page name |

### Adding New Pages

1. Create `pageX_name.py`
2. Implement `show()` function
3. Add to `PAGES` dict in `main.py`
4. Update navigation

## 🤝 Contributing

This is a demonstration project. For production use:
1. Add comprehensive error handling
2. Implement logging
3. Add unit tests
4. Set up CI/CD
5. Configure environment variables
6. Add authentication if needed

## 📄 License

This project is for educational and demonstration purposes.

## 🙏 Acknowledgments

- Built with Streamlit
- Inspired by real-world travel tech forecasting needs
- Data generation mimics Australian/New Zealand travel patterns

---

**Current Status:** ✅ Phase 1 Complete | 🚧 Phase 2 In Progress

For questions or issues, please refer to the detailed specification document.
