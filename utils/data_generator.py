import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta


@st.cache_data(ttl=3600)  # Cache for 1 hour
def generate_synthetic_data(start_date='2022-09-01', end_date='2025-09-30', seed=42):
    """
    Generate realistic daily flight booking data for Webjet.
    
    Parameters:
    -----------
    start_date : str
        Start date in YYYY-MM-DD format
    end_date : str
        End date in YYYY-MM-DD format
    seed : int
        Random seed for reproducibility
    
    Returns:
    --------
    pd.DataFrame
        DataFrame with daily booking data and features
    """
    np.random.seed(seed)
    
    # Create date range
    dates = pd.date_range(start=start_date, end=end_date, freq='D')
    n_days = len(dates)
    
    # Initialize DataFrame
    df = pd.DataFrame({'date': dates})
    df['day_of_week'] = df['date'].dt.dayofweek  # 0=Monday, 6=Sunday
    df['month'] = df['date'].dt.month
    df['year'] = df['date'].dt.year
    df['day_of_year'] = df['date'].dt.dayofyear
    
    # Base booking level (starting point)
    base_bookings = 400
    
    # 1. TREND COMPONENT (growth over time)
    # Linear growth from 2021 to 2024
    days_from_start = (df['date'] - df['date'].min()).dt.days
    trend = base_bookings + (days_from_start / 365) * 30  # 30 bookings/year growth
    
    # 2. WEEKLY SEASONALITY (Thu/Fri peaks for business travel)
    weekly_pattern = np.array([0.85, 0.90, 0.95, 1.15, 1.20, 0.95, 0.80])  # Mon-Sun
    weekly_seasonality = np.array([weekly_pattern[d % 7] for d in range(n_days)])
    
    # 3. YEARLY SEASONALITY (Dec/Jan peaks for holidays)
    yearly_seasonality = 1 + 0.3 * np.sin(2 * np.pi * df['day_of_year'] / 365 - np.pi/2)
    # Boost for December-January (summer holidays in AU/NZ)
    yearly_seasonality = np.where(
        (df['month'] == 12) | (df['month'] == 1),
        yearly_seasonality * 1.2,
        yearly_seasonality
    )
    
    # 4. COVID IMPACT (dramatic drop Mar 2020 - Nov 2021)
    covid_impact = np.ones(n_days)
    covid_start = pd.Timestamp('2020-03-15')
    covid_end = pd.Timestamp('2021-11-30')
    
    covid_mask = (df['date'] >= covid_start) & (df['date'] <= covid_end)
    # Gradual recovery curve
    covid_days = (df.loc[covid_mask, 'date'] - covid_start).dt.days
    covid_recovery = 0.2 + 0.8 * (covid_days / (covid_end - covid_start).days) ** 1.5
    covid_impact[covid_mask] = covid_recovery
    
    # 5. SCHOOL HOLIDAYS (AU/NZ calendar)
    # 5. SCHOOL HOLIDAYS (AU/NZ calendar) - Enhanced boost
    school_holidays = []
    for year in range(2021, 2025):
        school_holidays.extend([
            (f'{year}-01-01', f'{year}-02-05', 1.35),  # Summer holidays - peak
            (f'{year}-04-01', f'{year}-04-18', 1.20),  # Autumn break
            (f'{year}-06-24', f'{year}-07-10', 1.30),  # Winter break - ski season
            (f'{year}-09-23', f'{year}-10-09', 1.15),  # Spring break
            (f'{year}-12-15', f'{year}-12-31', 1.40),  # Christmas break - peak
        ])

    df['is_school_holiday'] = False
    df['holiday_boost_factor'] = 1.0
    for start, end, boost in school_holidays:
        try:
            mask = (df['date'] >= start) & (df['date'] <= end)
            df.loc[mask, 'is_school_holiday'] = True
            df.loc[mask, 'holiday_boost_factor'] = boost
        except:
            pass

    school_holiday_boost = df['holiday_boost_factor'].values
    # 6. PUBLIC HOLIDAYS (simplified)
    public_holidays = []
    for year in range(2021, 2025):
        public_holidays.extend([
            f'{year}-01-01',  # New Year
            f'{year}-01-26',  # Australia Day
            f'{year}-04-25',  # ANZAC Day
            f'{year}-12-25',  # Christmas
            f'{year}-12-26',  # Boxing Day
        ])
    
    df['is_holiday'] = df['date'].astype(str).isin(public_holidays)
    holiday_boost = np.where(df['is_holiday'], 0.7, 1.0)  # Lower on holidays themselves
    
    # 7. COMBINE ALL COMPONENTS
    bookings = (
        trend * 
        weekly_seasonality * 
        yearly_seasonality * 
        covid_impact * 
        school_holiday_boost * 
        holiday_boost
    )
    
    # 8. ADD RANDOM NOISE (±10%)
    noise = np.random.normal(1.0, 0.10, n_days)
    bookings = bookings * noise
    
    # 9. ADD OCCASIONAL OUTLIERS (1-2%)
    outlier_mask = np.random.random(n_days) < 0.015
    outlier_multiplier = np.random.choice([0.5, 1.8], size=n_days, p=[0.4, 0.6])
    bookings = np.where(outlier_mask, bookings * outlier_multiplier, bookings)
    
    # Round to integers
    df['bookings'] = np.round(bookings).astype(int)
    
    # 10. GENERATE EXOGENOUS VARIABLES
    # Marketing spend by channel (correlated with bookings + seasonality)
    base_marketing = 6000
    marketing_seasonality = 1 + 0.2 * np.sin(2 * np.pi * df['day_of_year'] / 365)
    marketing_trend = base_marketing + (days_from_start / 365) * 500

    # Total marketing spend
    df['marketing_spend'] = (
        marketing_trend * 
        marketing_seasonality * 
        np.random.normal(1.0, 0.15, n_days)
    )

    # Channel breakdown (percentages that sum to 1)
    channel_mix = np.random.dirichlet([3, 2, 1.5, 1, 0.5], n_days)  # Weighted toward search/social
    df['marketing_search'] = df['marketing_spend'] * channel_mix[:, 0]
    df['marketing_social'] = df['marketing_spend'] * channel_mix[:, 1]
    df['marketing_display'] = df['marketing_spend'] * channel_mix[:, 2]
    df['marketing_email'] = df['marketing_spend'] * channel_mix[:, 3]
    df['marketing_affiliate'] = df['marketing_spend'] * channel_mix[:, 4]

    df['marketing_spend'] = np.round(df['marketing_spend'], 2)
    df['marketing_search'] = np.round(df['marketing_search'], 2)
    df['marketing_social'] = np.round(df['marketing_social'], 2)
    df['marketing_display'] = np.round(df['marketing_display'], 2)
    df['marketing_email'] = np.round(df['marketing_email'], 2)
    df['marketing_affiliate'] = np.round(df['marketing_affiliate'], 2)
        
    # Competitor price index (inverse correlation with bookings)
    base_price_index = 100
    price_volatility = np.random.normal(0, 5, n_days).cumsum()
    df['competitor_price_index'] = base_price_index + price_volatility
    df['competitor_price_index'] = np.round(df['competitor_price_index'], 2)
    
    # Weather disruption index (0-10 scale, rare events)
    weather_disruption = np.zeros(n_days)
    disruption_events = np.random.random(n_days) < 0.05  # 5% of days
    weather_disruption[disruption_events] = np.random.uniform(3, 10, disruption_events.sum())
    # Smooth out disruptions
    df['weather_disruption_index'] = pd.Series(weather_disruption).rolling(3, center=True, min_periods=1).mean().values
    df['weather_disruption_index'] = np.round(df['weather_disruption_index'], 2)
    
    # 11. INTRODUCE MISSING VALUES (1-2%)
    missing_rate = 0.015
    for col in ['marketing_spend', 'competitor_price_index', 'weather_disruption_index']:
        missing_mask = np.random.random(n_days) < missing_rate
        df.loc[missing_mask, col] = np.nan
    
    # 12. SELECT AND ORDER FINAL COLUMNS
    final_columns = [
    'date',
    'bookings',
    'day_of_week',
    'marketing_spend',
    'marketing_search',
    'marketing_social',
    'marketing_display',
    'marketing_email',
    'marketing_affiliate',
    'competitor_price_index',
    'is_holiday',
    'is_school_holiday',
    'weather_disruption_index'
    ]
    
    df = df[final_columns]
    
    return df


def get_data_summary(df):
    """
    Generate summary statistics for the dataset.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    
    Returns:
    --------
    dict
        Summary statistics
    """
    summary = {
        'total_observations': len(df),
        'date_range': (df['date'].min(), df['date'].max()),
        'total_days': (df['date'].max() - df['date'].min()).days + 1,
        'missing_values': df.isnull().sum().to_dict(),
        'missing_pct': (df.isnull().sum() / len(df) * 100).to_dict(),
        'numeric_stats': df.select_dtypes(include=[np.number]).describe().to_dict()
    }
    
    return summary


if __name__ == "__main__":
    # Test the generator
    df = generate_synthetic_data()
    print("Data generated successfully!")
    print(f"Shape: {df.shape}")
    print(f"\nFirst 5 rows:\n{df.head()}")
    print(f"\nData types:\n{df.dtypes}")
    print(f"\nMissing values:\n{df.isnull().sum()}")
    
    summary = get_data_summary(df)
    print(f"\nDate range: {summary['date_range'][0]} to {summary['date_range'][1]}")
    print(f"Total observations: {summary['total_observations']}")
