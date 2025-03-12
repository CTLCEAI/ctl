import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import datetime

# Function to clean column names
def clean_column_names(df):
    """Clean and standardize column names"""
    columns = []
    for i, col in enumerate(df.columns):
        if 'Unnamed' in str(col):
            columns.append(f'Column_{i+1}')
        else:
            columns.append(str(col).strip())
    df.columns = columns
    return df

@st.cache_data
def get_sheet_names():
    """Get and cache sheet names from Excel file"""
    try:
        excel_file = pd.ExcelFile('ctl_tracker.xlsx')
        return excel_file.sheet_names
    except Exception as e:
        st.error(f"Error loading Excel file: {str(e)}")
        return []

@st.cache_data
def load_sheet_data(sheet_name):
    """Load and cache data from specific sheet"""
    try:
        df = pd.read_excel('ctl_tracker.xlsx', sheet_name=sheet_name, engine="openpyxl")
        return clean_column_names(df)
    except Exception as e:
        st.error(f"Error loading sheet data: {str(e)}")
        return None

def get_year_from_df(df):
    """Extract available years from date columns"""
    date_columns = df.select_dtypes(include=['datetime64', 'object']).columns
    years = set()
    
    for col in date_columns:
        try:
            df[col] = pd.to_datetime(df[col], errors='coerce')
            years.update(df[col].dt.year.dropna().unique())
        except Exception:
            pass
    
    return sorted([int(year) for year in years if not pd.isna(year)])

def filter_df_by_year(df, selected_year):
    """Filter dataframe by selected year"""
    if selected_year == "All Years":
        return df
    
    date_columns = df.select_dtypes(include=['datetime64']).columns
    mask = pd.Series(False, index=df.index)
    
    for col in date_columns:
        mask |= (df[col].dt.year == int(selected_year))
    
    return df[mask]

def create_correlation_heatmap(df):
    """Create and display a correlation heatmap"""
    numeric_df = df.select_dtypes(include=['int64', 'float64'])
    if numeric_df.empty:
        st.warning("No numeric data available for correlation analysis.")
        return
    
    correlation_matrix = numeric_df.corr()
    fig = px.imshow(
        correlation_matrix,
        text_auto=True,
        aspect="auto",
        color_continuous_scale="emrld",
        title="Correlation Heatmap"
    )
    
    st.plotly_chart(fig, use_container_width=True)

def create_enhanced_table(df):
    """Create an interactive table view"""
    if df.empty:
        st.warning("No data available.")
        return
    st.dataframe(df)

def create_summary_metrics(df, selected_year):
    """Create enhanced summary metrics"""
    st.subheader("📊 Key Metrics")
    if selected_year != "All Years":
        st.info(f"Showing metrics for year: {selected_year}")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Records", f"{len(df):,}")

    with col2:
        st.metric("Total Features", f"{len(df.columns):,}")

    with col3:
        missing_values = df.isnull().sum().sum()
        missing_percentage = (missing_values / (len(df) * len(df.columns)) * 100).round(2)
        st.metric("Missing Values", f"{missing_values:,}", f"{missing_percentage}%")

    with col4:
        numeric_cols = len(df.select_dtypes(include=['int64', 'float64']).columns)
        st.metric("Numeric Features", f"{numeric_cols:,}")

def main():
    # Title and description
    st.title("🏢 CTL Tracker Analytics Dashboard")
    st.markdown("""
    This dashboard provides comprehensive analytics and visualization for CTL company tracking data.
    Use the sidebar to navigate between different sheets and filter by year.
    """)

    # Get sheet names
    sheet_names = get_sheet_names()
    
    if sheet_names:
        st.sidebar.title("📑 Navigation")
        
        selected_sheet = st.sidebar.selectbox("Select Sheet", sheet_names)
        
        df = load_sheet_data(selected_sheet)
        
        if df is not None:
            available_years = get_year_from_df(df)
            year_options = ["All Years"] + [str(year) for year in available_years]
            selected_year = st.sidebar.selectbox("Select Year", year_options)
            
            filtered_df = filter_df_by_year(df, selected_year) if selected_year != "All Years" else df
            
            tabs = st.tabs(["📊 Overview", "📈 Status Analysis", "📋 Detailed View"])
            
            with tabs[0]:
                create_summary_metrics(filtered_df, selected_year)
                st.markdown("---")
                create_correlation_heatmap(filtered_df)
            
            with tabs[1]:
                st.subheader("📊 Status Analysis")
                st.warning("Status analysis functionality not implemented yet.")
            
            with tabs[2]:
                create_enhanced_table(filtered_df)
            
            st.markdown("---")
            st.markdown(f"Dashboard last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    main()
