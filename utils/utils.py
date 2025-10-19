import streamlit as st
import pandas as pd
import numpy as np
from typing import Any, Dict, List, Optional, Tuple

# ============================================================================
# SESSION STATE MANAGEMENT
# ============================================================================

def save_to_session(key: str, value: Any) -> None:
    """
    Save a value to Streamlit session state.
    
    Parameters:
    -----------
    key : str
        Session state key
    value : Any
        Value to store
    """
    st.session_state[key] = value


def load_from_session(key: str, default: Any = None) -> Any:
    """
    Load a value from Streamlit session state.
    
    Parameters:
    -----------
    key : str
        Session state key
    default : Any
        Default value if key not found
    
    Returns:
    --------
    Any
        Stored value or default
    """
    return st.session_state.get(key, default)


def clear_session_key(key: str) -> None:
    """
    Remove a specific key from session state.
    
    Parameters:
    -----------
    key : str
        Session state key to remove
    """
    if key in st.session_state:
        del st.session_state[key]


def clear_all_session(exclude: List[str] = None) -> None:
    """
    Clear all session state except specified keys.
    
    Parameters:
    -----------
    exclude : List[str]
        Keys to preserve (e.g., ['current_page'])
    """
    if exclude is None:
        exclude = []
    
    keys_to_delete = [k for k in st.session_state.keys() if k not in exclude]
    for key in keys_to_delete:
        del st.session_state[key]


# ============================================================================
# DATA QUALITY VALIDATION
# ============================================================================

def validate_data_quality(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Comprehensive data quality validation.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    
    Returns:
    --------
    dict
        Validation results with checks and issues
    """
    results = {
        'is_valid': True,
        'checks': {},
        'issues': [],
        'warnings': [],
        'summary': {}
    }
    
    # Check 1: Required columns for time series
    required_cols = ['date', 'bookings']
    missing_cols = [col for col in required_cols if col not in df.columns]
    
    results['checks']['required_columns'] = len(missing_cols) == 0
    if missing_cols:
        results['is_valid'] = False
        results['issues'].append(f"Missing required columns: {missing_cols}")
    
    # Check 2: Duplicate dates
    if 'date' in df.columns:
        n_duplicates = df['date'].duplicated().sum()
        results['checks']['no_duplicate_dates'] = n_duplicates == 0
        results['summary']['duplicate_dates'] = n_duplicates
        
        if n_duplicates > 0:
            results['warnings'].append(f"{n_duplicates} duplicate dates detected")
    
    # Check 3: Continuous date range (no gaps)
    if 'date' in df.columns:
        date_range = pd.date_range(
            start=df['date'].min(),
            end=df['date'].max(),
            freq='D'
        )
        expected_days = len(date_range)
        actual_days = len(df)
        
        results['checks']['continuous_dates'] = expected_days == actual_days
        results['summary']['expected_days'] = expected_days
        results['summary']['actual_days'] = actual_days
        results['summary']['missing_days'] = expected_days - actual_days
        
        if expected_days != actual_days:
            results['warnings'].append(
                f"Date range has gaps: {expected_days - actual_days} days missing"
            )
    
    # Check 4: Missing values
    missing_counts = df.isnull().sum()
    missing_pct = (missing_counts / len(df) * 100).round(2)
    
    results['summary']['missing_values'] = missing_counts.to_dict()
    results['summary']['missing_pct'] = missing_pct.to_dict()
    
    total_missing = missing_counts.sum()
    results['checks']['has_missing_values'] = total_missing > 0
    
    if total_missing > 0:
        results['warnings'].append(f"Total missing values: {total_missing}")
    
    # Check 5: Outliers (for numeric columns)
    if 'bookings' in df.columns:
        outliers = detect_outliers(df, 'bookings', method='zscore', threshold=3)
        results['summary']['outliers_count'] = len(outliers)
        results['checks']['has_outliers'] = len(outliers) > 0
        
        if len(outliers) > 0:
            results['warnings'].append(
                f"{len(outliers)} outliers detected in bookings (|z| > 3)"
            )
    
    # Check 6: Data types
    if 'date' in df.columns:
        results['checks']['date_is_datetime'] = pd.api.types.is_datetime64_any_dtype(df['date'])
        if not results['checks']['date_is_datetime']:
            results['issues'].append("'date' column is not datetime type")
    
    if 'bookings' in df.columns:
        results['checks']['bookings_is_numeric'] = pd.api.types.is_numeric_dtype(df['bookings'])
        if not results['checks']['bookings_is_numeric']:
            results['issues'].append("'bookings' column is not numeric type")
    
    # Check 7: Negative values
    if 'bookings' in df.columns:
        n_negative = (df['bookings'] < 0).sum()
        results['checks']['no_negative_bookings'] = n_negative == 0
        results['summary']['negative_bookings'] = n_negative
        
        if n_negative > 0:
            results['issues'].append(f"{n_negative} negative booking values detected")
            results['is_valid'] = False
    
    # Check 8: Data range
    if 'date' in df.columns:
        results['summary']['date_range'] = {
            'start': df['date'].min(),
            'end': df['date'].max(),
            'days': len(df)
        }
    
    return results


def detect_outliers(
    df: pd.DataFrame,
    column: str,
    method: str = 'zscore',
    threshold: float = 3
) -> pd.DataFrame:
    """
    Detect outliers in a numeric column.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    column : str
        Column name to check for outliers
    method : str
        Detection method ('zscore' or 'iqr')
    threshold : float
        Threshold for outlier detection
    
    Returns:
    --------
    pd.DataFrame
        Rows containing outliers
    """
    if column not in df.columns:
        return pd.DataFrame()
    
    if method == 'zscore':
        # Z-score method
        mean = df[column].mean()
        std = df[column].std()
        z_scores = np.abs((df[column] - mean) / std)
        outlier_mask = z_scores > threshold
        
    elif method == 'iqr':
        # Interquartile range method
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - threshold * IQR
        upper_bound = Q3 + threshold * IQR
        outlier_mask = (df[column] < lower_bound) | (df[column] > upper_bound)
    
    else:
        raise ValueError(f"Unknown method: {method}. Use 'zscore' or 'iqr'")
    
    return df[outlier_mask].copy()


# ============================================================================
# DATA FORMATTING AND DISPLAY
# ============================================================================

def format_number(num: float, decimals: int = 0) -> str:
    """
    Format number with thousands separator.
    
    Parameters:
    -----------
    num : float
        Number to format
    decimals : int
        Number of decimal places
    
    Returns:
    --------
    str
        Formatted number
    """
    if decimals == 0:
        return f"{num:,.0f}"
    else:
        return f"{num:,.{decimals}f}"


def format_percentage(num: float, decimals: int = 1) -> str:
    """
    Format number as percentage.
    
    Parameters:
    -----------
    num : float
        Number to format (0.15 = 15%)
    decimals : int
        Number of decimal places
    
    Returns:
    --------
    str
        Formatted percentage
    """
    return f"{num * 100:.{decimals}f}%"


def create_metric_card(label: str, value: str, delta: str = None) -> str:
    """
    Create HTML for a metric card.
    
    Parameters:
    -----------
    label : str
        Metric label
    value : str
        Metric value
    delta : str
        Change indicator (optional)
    
    Returns:
    --------
    str
        HTML string
    """
    delta_html = ""
    if delta:
        delta_color = "#00c853" if delta.startswith("+") else "#ff5252"
        delta_html = f"<div style='color: {delta_color}; font-size: 0.9rem; margin-top: 0.25rem;'>{delta}</div>"
    
    return f"""
        <div style='background-color: #f0f2f6; padding: 1rem; border-radius: 0.5rem; border-left: 4px solid #1f77b4;'>
            <div style='font-size: 0.9rem; color: #666; margin-bottom: 0.25rem;'>{label}</div>
            <div style='font-size: 1.5rem; font-weight: bold; color: #1f77b4;'>{value}</div>
            {delta_html}
        </div>
    """


# ============================================================================
# DATA EXPORT
# ============================================================================

def convert_df_to_csv(df: pd.DataFrame) -> str:
    """
    Convert DataFrame to CSV string for download.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Dataframe to convert
    
    Returns:
    --------
    str
        CSV string
    """
    return df.to_csv(index=False).encode('utf-8')


# ============================================================================
# DATE UTILITIES
# ============================================================================

def get_date_features(date: pd.Timestamp) -> Dict[str, int]:
    """
    Extract features from a date.
    
    Parameters:
    -----------
    date : pd.Timestamp
        Input date
    
    Returns:
    --------
    dict
        Date features
    """
    return {
        'year': date.year,
        'month': date.month,
        'day': date.day,
        'day_of_week': date.dayofweek,
        'day_of_year': date.dayofyear,
        'week_of_year': date.isocalendar()[1],
        'quarter': date.quarter,
        'is_weekend': date.dayofweek >= 5,
        'is_month_start': date.is_month_start,
        'is_month_end': date.is_month_end
    }


# ============================================================================
# STYLING UTILITIES
# ============================================================================

def get_status_badge(status: str, message: str = None) -> str:
    """
    Create a colored status badge.
    
    Parameters:
    -----------
    status : str
        Status type ('success', 'warning', 'error', 'info')
    message : str
        Message to display
    
    Returns:
    --------
    str
        HTML badge
    """
    colors = {
        'success': {'bg': '#d4edda', 'border': '#28a745', 'text': '#155724', 'icon': '✓'},
        'warning': {'bg': '#fff3cd', 'border': '#ffc107', 'text': '#856404', 'icon': '⚠️'},
        'error': {'bg': '#f8d7da', 'border': '#dc3545', 'text': '#721c24', 'icon': '✗'},
        'info': {'bg': '#d1ecf1', 'border': '#17a2b8', 'text': '#0c5460', 'icon': 'ℹ️'}
    }
    
    style = colors.get(status, colors['info'])
    msg = message or status.capitalize()
    
    return f"""
        <div style='background-color: {style["bg"]}; 
                    border-left: 4px solid {style["border"]}; 
                    color: {style["text"]}; 
                    padding: 0.75rem; 
                    border-radius: 0.25rem; 
                    margin: 0.5rem 0;'>
            <strong>{style["icon"]} {msg}</strong>
        </div>
    """


# ============================================================================
# STATISTICAL UTILITIES
# ============================================================================

def calculate_basic_stats(series: pd.Series) -> Dict[str, float]:
    """
    Calculate basic statistics for a series.
    
    Parameters:
    -----------
    series : pd.Series
        Input series
    
    Returns:
    --------
    dict
        Basic statistics
    """
    return {
        'count': len(series),
        'mean': series.mean(),
        'median': series.median(),
        'std': series.std(),
        'min': series.min(),
        'max': series.max(),
        'q25': series.quantile(0.25),
        'q75': series.quantile(0.75),
        'missing': series.isnull().sum(),
        'missing_pct': series.isnull().sum() / len(series) * 100
    }


# ============================================================================
# VALIDATION HELPERS
# ============================================================================

def check_data_loaded(data_key: str = 'raw_data') -> bool:
    """
    Check if data is loaded in session state and show warning if not.
    
    Parameters:
    -----------
    data_key : str
        Session state key for data
    
    Returns:
    --------
    bool
        True if data is loaded
    """
    data = load_from_session(data_key)
    
    if data is None:
        st.warning("⚠️ Please load data in **Data Acquisition** page first.")
        return False
    
    return True