import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import plotly.express as px

from datetime import datetime

# [Previous CSS styles remain the same...]

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
        df = pd.read_excel('ctl_tracker.xlsx', sheet_name=sheet_name)
        return clean_column_names(df)
    except Exception as e:
        st.error(f"Error loading sheet data: {str(e)}")
        return None

def get_year_from_df(df):
    """Extract available years from date columns"""
    date_columns = df.select_dtypes(include=['datetime64']).columns
    years = set()
    
    for col in date_columns:
        years.update(df[col].dt.year.unique())
    
    return sorted(list(years))

def filter_df_by_year(df, selected_year):
    """Filter dataframe by selected year"""
    if selected_year == "All Years":
        return df
    
    date_columns = df.select_dtypes(include=['datetime64']).columns
    mask = pd.Series(False, index=df.index)
    
    for col in date_columns:
        mask |= (df[col].dt.year == selected_year)
    
    return df[mask]

def create_enhanced_pie_chart(df, column, title):
    """Create an enhanced pie chart with better styling and interactivity"""
    if df.empty:
        st.warning(f"No data available for the selected year in {column}")
        return None
        
    value_counts = df[column].value_counts()
    
    # Calculate percentages for labels
    total = value_counts.sum()
    percentages = (value_counts / total * 100).round(1)
    
    # Custom text labels
    labels = [f"{idx}<br>{val} ({pct}%)" for idx, val, pct 
             in zip(value_counts.index, value_counts.values, percentages)]

    fig = go.Figure(data=[go.Pie(
        labels=value_counts.index,
        values=value_counts.values,
        text=labels,
        textinfo='text',
        hovertemplate="<b>%{label}</b><br>" +
                      "Count: %{value}<br>" +
                      "Percentage: %{percent}<br>" +
                      "<extra></extra>",
        hole=0.4,
        marker=dict(
            colors=px.colors.qualitative.Set3,
            line=dict(color='white', width=2)
        )
    )])
    
    fig.update_layout(
        title=dict(
            text=title,
            x=0.5,
            y=0.95,
            xanchor='center',
            yanchor='top',
            font=dict(size=20)
        ),
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.2,
            xanchor="center",
            x=0.5
        ),
        annotations=[dict(
            text=f'Total: {total}',
            x=0.5,
            y=0.5,
            font_size=16,
            showarrow=False
        )],
        height=500,
    )
    
    return fig

def create_status_analysis(df, selected_year):
    """Create enhanced status analysis with interactive pie charts"""
    st.subheader("📊 Status Analysis")
    if selected_year != "All Years":
        st.info(f"Showing data for year: {selected_year}")
    
    # Find status-related columns
    status_columns = [col for col in df.columns if 'status' in col.lower()]
    if not status_columns:
        potential_status_cols = [col for col in df.columns if any(keyword in str(col).lower() 
                               for keyword in ['status', 'state', 'condition', 'phase', 'stage'])]
        status_columns = potential_status_cols

    if status_columns:
        col1, col2 = st.columns(2)
        
        for i, status_col in enumerate(status_columns[:2]):  # Show up to 2 status charts
            with col1 if i % 2 == 0 else col2:
                fig = create_enhanced_pie_chart(
                    df, 
                    status_col,
                    f"{status_col} Distribution"
                )
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
                
                    # Display metrics
                    status_counts = df[status_col].value_counts()
                    total = status_counts.sum()
                    
                    metrics_cols = st.columns(len(status_counts))
                    for j, (status, count) in enumerate(status_counts.items()):
                        with metrics_cols[j]:
                            percentage = (count / total * 100).round(1)
                            st.metric(
                                f"{status}",
                                f"{count:,}",
                                f"{percentage}%"
                            )

def create_time_series_analysis(df):
    """Create time series analysis if date columns are present"""
    date_columns = df.select_dtypes(include=['datetime64']).columns
    
    if len(date_columns) > 0:
        st.subheader("📅 Time Series Analysis")
        
        # Select date column for analysis
        selected_date_col = st.selectbox(
            "Select Date Column for Analysis",
            date_columns
        )
        
        # Group by date and count
        df['Month_Year'] = df[selected_date_col].dt.to_period('M')
        monthly_counts = df.groupby('Month_Year').size().reset_index()
        monthly_counts['Month_Year'] = monthly_counts['Month_Year'].astype(str)
        
        # Create line chart
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=monthly_counts['Month_Year'],
            y=monthly_counts[0],
            mode='lines+markers',
            name='Count',
            line=dict(width=3),
            marker=dict(size=8)
        ))
        
        fig.update_layout(
            title="Monthly Trend",
            xaxis_title="Month",
            yaxis_title="Count",
            height=400,
            showlegend=True
        )
        
        st.plotly_chart(fig, use_container_width=True)

def create_summary_metrics(df, selected_year):
    """Create enhanced summary metrics"""
    st.subheader("📊 Key Metrics")
    if selected_year != "All Years":
        st.info(f"Showing metrics for year: {selected_year}")
    
    # Basic metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Total Records",
            f"{len(df):,}",
            delta=None
        )
    
    with col2:
        st.metric(
            "Total Features",
            f"{len(df.columns):,}",
            delta=None
        )
    
    with col3:
        missing_values = df.isnull().sum().sum()
        missing_percentage = (missing_values / (len(df) * len(df.columns)) * 100).round(2)
        st.metric(
            "Missing Values",
            f"{missing_values:,}",
            f"{missing_percentage}%"
        )
    
    with col4:
        numeric_cols = len(df.select_dtypes(include=['int64', 'float64']).columns)
        st.metric(
            "Numeric Features",
            f"{numeric_cols:,}",
            delta=None
        )

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
        # Sidebar navigation
        st.sidebar.title("📑 Navigation")
        
        # Sheet selection
        selected_sheet = st.sidebar.selectbox(
            "Select Sheet",
            sheet_names
        )
        
        # Load selected sheet data
        df = load_sheet_data(selected_sheet)
        
        if df is not None:
            # Year filter in sidebar
            available_years = get_year_from_df(df)
            year_options = ["All Years"] + [str(year) for year in available_years]
            selected_year = st.sidebar.selectbox(
                "Select Year",
                year_options
            )
            
            # Filter data by year
            if selected_year != "All Years":
                filtered_df = filter_df_by_year(df, int(selected_year))
            else:
                filtered_df = df
            
            # Create tabs for different views
            tabs = st.tabs([
                "📊 Overview",
                "📈 Status Analysis",
                "📋 Detailed View"
            ])
            
            with tabs[0]:
                create_summary_metrics(filtered_df, selected_year)
                st.markdown("---")
                create_time_series_analysis(filtered_df)
                create_correlation_heatmap(filtered_df)
            
            with tabs[1]:
                create_status_analysis(filtered_df, selected_year)
            
            with tabs[2]:
                create_enhanced_table(filtered_df)
            
            # Footer
            st.markdown("---")
            st.markdown(
                f"Dashboard last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
            )

if __name__ == "__main__":
    main()
