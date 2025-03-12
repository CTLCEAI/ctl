import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import subprocess
from datetime import datetime

# Automatically install missing package
try:
    import openpyxl
except ModuleNotFoundError:
    subprocess.run(["pip", "install", "openpyxl"])
    import openpyxl  # Re-import after installation

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
            years.update(df
