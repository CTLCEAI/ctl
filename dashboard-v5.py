import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import openai


# Load data from the Excel file
@st.cache_data
def save_data(df):
    try:
        df.to_excel('AI Assets v2.xlsx', index=False)
        return True
    except Exception as e:
        st.error(f"Error saving data: {str(e)}")
        return False

def crud_operations(data):
    st.markdown("## IP Assets Management")
    st.markdown("---")

    # Initialize the data structure if needed columns don't exist
    required_columns = ['Asset Name', 'Asset Type', 'Stage', 'Description', 'Creation Date']
    for col in required_columns:
        if col not in data.columns:
            data[col] = ''

    # Create tabs for different CRUD operations
    crud_tab1, crud_tab2, crud_tab3, crud_tab4 = st.tabs([
        "➕ Create Asset",
        "👀 View Assets",
        "✏️ Update Asset",
        "🗑️ Delete Asset"
    ])

    with crud_tab1:
        st.markdown("### Add New IP Asset")
        
        # Form for creating new asset
        with st.form("create_asset_form"):
            col1, col2 = st.columns(2)
            
            with col1:
                asset_name = st.text_input("Asset Name")
                asset_type = st.selectbox("Asset Type", ["Type A", "Type B", "Type C"])
                stage = st.selectbox("Stage", ["A", "B", "C"])
            
            with col2:
                description = st.text_area("Description")
                creation_date = st.date_input("Creation Date")
            
            submitted = st.form_submit_button("Add Asset")
            
            if submitted and asset_name:
                try:
                    # Find the appropriate column indices based on the year
                    year = creation_date.year
                    year_columns = {
                        2019: (2, 8),   
                        2020: (13, 19),  
                        2021: (24, 30),
                        2022: (35, 41),
                        2023: (47, 53),
                        2024: (60, 66)
                    }
                    
                    if year in year_columns:
                        start_col, end_col = year_columns[year]
                        
                        # Create new row with the correct structure
                        new_row = pd.Series(index=data.columns, dtype='object')
                        new_row[data.columns[start_col]] = asset_name
                        new_row[data.columns[start_col + 1]] = asset_type
                        new_row[data.columns[start_col + 4]] = stage
                        new_row[data.columns[start_col + 2]] = description
                        
                        # Append to existing data
                        updated_data = pd.concat([data, pd.DataFrame([new_row])], ignore_index=True)
                        
                        if save_data(updated_data):
                            st.success("Asset added successfully!")
                            st.rerun()
                    else:
                        st.error(f"Year {year} is not configured in the system.")
                except Exception as e:
                    st.error(f"Error adding asset: {str(e)}")

    with crud_tab2:
        st.markdown("### View IP Assets")
        
        # Extract and display assets from all years
        all_assets = []
        year_columns = {
            2019: (2, 8),   
            2020: (13, 19),  
            2021: (24, 30),
            2022: (35, 41),
            2023: (47, 53),
            2024: (60, 66)
        }
        
        for year, (start_col, end_col) in year_columns.items():
            year_data = data.iloc[28:, start_col:end_col].copy()
            if not year_data.empty:
                year_data.columns = ['Asset Name', 'Asset Type', 'Description', 'Other Info', 'Stage', 'Status']
                year_data['Year'] = year
                all_assets.append(year_data)
        
        if all_assets:
            combined_assets = pd.concat(all_assets, ignore_index=True)
            
            # Filters
            col1, col2, col3 = st.columns(3)
            with col1:
                filter_year = st.multiselect("Filter by Year", sorted(combined_assets['Year'].unique().tolist()))
            with col2:
                filter_stage = st.multiselect("Filter by Stage", sorted(combined_assets['Stage'].dropna().unique().tolist()))
            with col3:
                search_term = st.text_input("Search Assets", "")
            
            # Apply filters
            filtered_data = combined_assets.copy()
            if filter_year:
                filtered_data = filtered_data[filtered_data['Year'].isin(filter_year)]
            if filter_stage:
                filtered_data = filtered_data[filtered_data['Stage'].isin(filter_stage)]
            if search_term:
                filtered_data = filtered_data[
                    filtered_data['Asset Name'].str.contains(search_term, case=False, na=False)
                ]
            
            # Display filtered data
            st.dataframe(
                filtered_data.dropna(subset=['Asset Name']).style.background_gradient(cmap='Blues'),
                use_container_width=True,
                height=400
            )
        else:
            st.info("No assets found in the system.")

    with crud_tab3:
        st.markdown("### Update IP Asset")
        
        # Get all assets for updating
        all_assets = []
        for year, (start_col, end_col) in year_columns.items():
            year_data = data.iloc[28:, start_col:end_col].copy()
            if not year_data.empty:
                year_data.columns = ['Asset Name', 'Asset Type', 'Description', 'Other Info', 'Stage', 'Status']
                year_data['Year'] = year
                year_data['start_col'] = start_col
                all_assets.append(year_data)
        
        if all_assets:
            combined_assets = pd.concat(all_assets, ignore_index=True)
            asset_names = combined_assets.dropna(subset=['Asset Name'])['Asset Name'].unique().tolist()
            
            asset_to_update = st.selectbox(
                "Select Asset to Update",
                asset_names,
                key="update_asset_select"
            )
            
            if asset_to_update:
                asset_data = combined_assets[combined_assets['Asset Name'] == asset_to_update].iloc[0]
                
                with st.form("update_asset_form"):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        updated_name = st.text_input("Asset Name", asset_data['Asset Name'])
                        updated_type = st.selectbox(
                            "Asset Type",
                            ["Type A", "Type B", "Type C"],
                            index=["Type A", "Type B", "Type C"].index(asset_data['Asset Type'])
                            if asset_data['Asset Type'] in ["Type A", "Type B", "Type C"] else 0
                        )
                        updated_stage = st.selectbox(
                            "Stage",
                            ["A", "B", "C"],
                            index=["A", "B", "C"].index(asset_data['Stage'])
                            if asset_data['Stage'] in ["A", "B", "C"] else 0
                        )
                    
                    with col2:
                        updated_description = st.text_area("Description", asset_data['Description'])
                        updated_year = st.selectbox("Year", list(year_columns.keys()), 
                                                  index=list(year_columns.keys()).index(asset_data['Year']))
                    
                    update_submitted = st.form_submit_button("Update Asset")
                    
                    if update_submitted:
                        try:
                            # Remove old entry
                            old_start_col = asset_data['start_col']
                            data.iloc[28:, int(old_start_col)] = data.iloc[28:, int(old_start_col)].replace(asset_data['Asset Name'], '')
                            
                            # Add new entry
                            new_start_col = year_columns[updated_year][0]
                            empty_row_idx = data.iloc[28:, new_start_col].isna().idxmax()
                            
                            data.iloc[empty_row_idx, new_start_col] = updated_name
                            data.iloc[empty_row_idx, new_start_col + 1] = updated_type
                            data.iloc[empty_row_idx, new_start_col + 2] = updated_description
                            data.iloc[empty_row_idx, new_start_col + 4] = updated_stage
                            
                            if save_data(data):
                                st.success("Asset updated successfully!")
                                st.rerun()
                        except Exception as e:
                            st.error(f"Error updating asset: {str(e)}")

    with crud_tab4:
        st.markdown("### Delete IP Asset")
        
        # Get all assets for deletion
        all_assets = []
        for year, (start_col, end_col) in year_columns.items():
            year_data = data.iloc[28:, start_col:end_col].copy()
            if not year_data.empty:
                year_data.columns = ['Asset Name', 'Asset Type', 'Description', 'Other Info', 'Stage', 'Status']
                year_data['Year'] = year
                year_data['start_col'] = start_col
                all_assets.append(year_data)
        
        if all_assets:
            combined_assets = pd.concat(all_assets, ignore_index=True)
            asset_names = combined_assets.dropna(subset=['Asset Name'])['Asset Name'].unique().tolist()
            
            asset_to_delete = st.selectbox(
                "Select Asset to Delete",
                asset_names,
                key="delete_asset_select"
            )
            
            if asset_to_delete:
                asset_data = combined_assets[combined_assets['Asset Name'] == asset_to_delete].iloc[0]
                st.warning(f"Are you sure you want to delete {asset_to_delete}?")
                
                if st.button("Delete Asset"):
                    try:
                        # Find and remove the asset
                        start_col = int(asset_data['start_col'])
                        data.iloc[28:, start_col] = data.iloc[28:, start_col].replace(asset_to_delete, '')
                        
                        if save_data(data):
                            st.success("Asset deleted successfully!")
                            st.rerun()
                    except Exception as e:
                        st.error(f"Error deleting asset: {str(e)}")
        else:
            st.info("No assets found in the system.")
def load_data():
    try:
        file_path = 'AI Assets v2.xlsx'
        return pd.read_excel(file_path)
    except FileNotFoundError:
        st.error("Excel file not found. Please check the file path.")
        return None
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None

def print_year_data(data, year):
    # Define column ranges for different years
    year_columns = {
        2019: (2, 8),   
        2020: (13, 19),  
        2021: (24, 30),
        2022: (35, 41),
        2023: (47, 53),
        2024: (60, 66)
    }
    
    # Define vertical mappings
    vertical_mappings = {
        'TAFGAI': 'Financial & Risk',
        'SOFINAA': 'Social Impact',
        'PAL': 'Education',
        'MEDGIVER': 'Healthcare & Medical',
        'NUURAI': 'Mental & Spiritual Wellness',
        'CONTRACKAI': 'Legal',
        'STORQUE':'Investment'
    }
    
    # Define TRL stage mappings
    stage_mappings = {
        'A': 'TRL 7-9',
        'B': 'TRL 4-6',
        'C': 'TRL 1-3'
    }
    
    if year not in year_columns:
        st.error(f"Data for {year} is not configured.")
        return
    
    start_col, end_col = year_columns[year]
    rows_to_display = []
    stages = []
    verticals = []
    
    # Get title from row 27
    title = data.iloc[27, start_col] if pd.notna(data.iloc[27, start_col]) else f"Data for {year}"
    st.markdown(f"## {title}")
    
    # Data collection with spinner
    with st.spinner(f'Loading {year} data...'):
        start_index = 28  # Title is on row 27, so we start from 28
        for i in range(start_index, len(data)):
            row_subset = data.iloc[i, start_col:end_col]
            if row_subset.isnull().all():
                break
            
            rows_to_display.append(data.iloc[i:i+1, start_col:end_col])
            
            vertical = data.iloc[i, start_col + 3]
            if pd.notna(vertical):
                mapped_vertical = vertical_mappings.get(vertical, vertical)
                verticals.append(mapped_vertical)
            
            stage = data.iloc[i, start_col + 4]
            if pd.notna(stage):
                mapped_stage = stage_mappings.get(stage, stage)
                stages.append(mapped_stage)

    # Generate visualizations and analysis if data exists
    if rows_to_display:
        subset_df = pd.concat(rows_to_display)
        stage_counts = pd.Series(stages).value_counts()
        vertical_counts = pd.Series(verticals).value_counts()

        # Create metrics cards for stages
        st.markdown("### TRL Distribution")
        metrics_container = st.container()
        m1, m2, m3, m4 = metrics_container.columns(4)
        
        with m1:
            st.metric("TRL 7-9", stage_counts.get('TRL 7-9', 0))
        with m2:
            st.metric("TRL 4-6", stage_counts.get('TRL 4-6', 0))
        with m3:
            st.metric("TRL 1-3", stage_counts.get('TRL 1-3', 0))
        with m4:
            st.metric("Total Projects", len(stages))

        # TRL Distribution Visualizations
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown(f"### TRL Distribution for {year}")
            fig1 = plt.figure(figsize=(8, 8))
            plt.pie(stage_counts.values,
                   labels=stage_counts.index,
                   autopct='%1.1f%%',
                #    colors=['#4e73df', '#1cc88a', '#36b9cc'],
                    colors = plt.cm.Set3(np.linspace(0, 1, len(vertical_counts))),
                   shadow=True)
            plt.title(f'TRL Distribution {year}')
            st.pyplot(fig1)
            plt.close(fig1)
        
        with col2:
            st.markdown("### TRL Count Analysis")
            fig2 = plt.figure(figsize=(8, 8))
            plt.bar(range(len(stage_counts)), 
                   stage_counts.values,
                #    color=['#4e73df', '#1cc88a', '#36b9cc']
                    color = plt.cm.Set3(np.linspace(0, 1, len(vertical_counts))))
            plt.xticks(range(len(stage_counts)), 
                      stage_counts.index,
                      rotation=45)
            plt.title('TRL Count Distribution')
            plt.ylabel('Number of Projects')
            
            for i, v in enumerate(stage_counts.values):
                plt.text(i, v, str(v), ha='center', va='bottom')
            
            st.pyplot(fig2)
            plt.close(fig2)

        # Vertical Distribution Analysis
        st.markdown("### Vertical Distribution")
        col3, col4 = st.columns([1, 1])
        
        with col3:
            st.markdown(f"### Vertical Distribution for {year}")
            fig3 = plt.figure(figsize=(8, 8))
            colors = plt.cm.Set3(np.linspace(0, 1, len(vertical_counts)))
            plt.pie(vertical_counts.values,
                   labels=vertical_counts.index,
                   autopct='%1.1f%%',
                   colors=colors,
                   shadow=True)
            plt.title(f'Vertical Distribution {year}')
            st.pyplot(fig3)
            plt.close(fig3)
        
        with col4:
            st.markdown("### Vertical Count Analysis")
            fig4 = plt.figure(figsize=(8, 8))
            bars = plt.bar(range(len(vertical_counts)), 
                    vertical_counts.values,
                    # color=colors
                    color = plt.cm.Set3(np.linspace(0, 1, len(vertical_counts)))
                    )
            plt.xticks(range(len(vertical_counts)), 
                      vertical_counts.index,
                      rotation=45,
                      ha='right')
            plt.title('Vertical Count Distribution')
            plt.ylabel('Number of Projects')
            
            for bar in bars:
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2., height,
                        f'{int(height)}',
                        ha='center', va='bottom')
            
            plt.tight_layout()
            st.pyplot(fig4)
            plt.close(fig4)

        # Add multiple analysis buttons side by side
        st.markdown("### Analysis Options")

        # Function to generate specific analysis
        def generate_analysis(analysis_type):
            prompts = {
                "summary": f"""
                Provide a concise summary of the IP portfolio for year {year}:
                - Total Projects: {len(stages)}
                - TRL Stage Distribution: {dict(stage_counts)}
                - Vertical Distribution: {dict(vertical_counts)}
                
                Focus on:
                1. Overall portfolio health
                2. Key strengths
                3. Portfolio composition
                4. Year-over-year changes (if apparent)
                
                Keep it business-focused and highlight the most important insights. Express the figures in % terms for the year and across the years comparatively. Analyze the % composition of each stage in the TRL (Technology Readiness Level, comparing them with the rest of the stages. Describe the concentration and distribution patterns, highlighting any significant variations or trends.
                """,
                
                "trl": f"""
                Analyze the TRL (Technology Readiness Level) distribution for year {year}:
                - TRL Distribution: {dict(stage_counts)}
                - Total Projects: {len(stages)}
                
                Provide insights on:
                1. Maturity profile of the portfolio
                2. Balance between early-stage and mature projects
                3. Risks and opportunities in the current TRL distribution
                4. Recommendations for TRL progression
                
                Focus on actionable insights for R&D and commercialization strategy. with supporting facts and figures in figures, % terms and years quoted accordingly.
                """,
                
                "recommendations": f"""
                Based on the portfolio data for year {year}:
                - Vertical Distribution: {dict(vertical_counts)}
                - TRL Distribution: {dict(stage_counts)}
                - Total Projects: {len(stages)}
                
                Provide strategic recommendations covering:
                1. Investment focus areas
                2. Risk mitigation strategies
                3. Opportunity areas for expansion
                4. Portfolio balancing suggestions
                5. Next steps for portfolio optimization
                
                Make recommendations specific and actionable with with supporting facts and figures in figures, % terms and years quoted accordingly.
                """
            }
            
            try:
                client = openai.OpenAI(api_key="sk-svcacct-KhL3atKnJCrYgWi4UVRnrLWhmQ9zMjEzBnlN91dkSVeusipZ-EFfwl942McnlT3BlbkFJyVR8sLL9GgAjhzz2zIeeAZRRLWLuLoXzECK6Jv8cFNOuVfcOPU2Z248uidCAA")
                
                with st.spinner(f"Generating {analysis_type.replace('_', ' ')}..."):
                    response = client.chat.completions.create(
                        model="gpt-4",
                        messages=[
                            {"role": "system", "content": "You are a senior IP portfolio analyst specializing in technology and innovation assessment."},
                            {"role": "user", "content": prompts[analysis_type]}
                        ],
                        max_tokens=500,
                        temperature=0.7
                    )
                    
                    # Updated styling with larger text and better spacing
                    st.markdown("""
                        <style>
                            .analysis-container {
                                padding: 2rem;
                                background-color: #f8f9fa;
                                border-radius: 0.5rem;
                                box-shadow: 0 0.15rem 1.75rem 0 rgba(58, 59, 69, 0.15);
                                margin: 1rem 0;
                                font-size: 1.1rem;
                                line-height: 1.6;
                            }
                            .analysis-container p {
                                margin-bottom: 1rem;
                                font-size: 18px;
                            }
                        </style>
                    """, unsafe_allow_html=True)
                    
                    st.markdown(f"""
                        <div class="analysis-container">
                            {response.choices[0].message.content}
                        </div>
                    """, unsafe_allow_html=True)
                    st.markdown("---")

            except Exception as e:
                st.error(f"Error generating analysis: {str(e)}")

        # Handle button clicks
        col1, col2, col3 = st.columns([0.33, 0.33, 0.33])

        # Define buttons with their respective types and icons in a more organized way
        button_config = {
            'summary': ('📊 What is happening', col1),
            'trl': ('🔬 Why Did this happen', col2),
            'recommendations': ('💡 What should I do', col3)
        }

        # Create buttons and handle clicks
        for analysis_type, (label, col) in button_config.items():
            with col:
                if st.button(label, key=f"{analysis_type}_{year}", use_container_width=True):
                    generate_analysis(analysis_type)

        # Display data table with improved styling
        st.markdown("### Detailed Data View")
        display_df = subset_df.copy()
        
        # Rename columns
        columns = ['Project Name', 'Country', 'Project Owner', 'Vertical', 'TRL Categories', 'Remarks']
        if len(display_df.columns) == len(columns):
            display_df.columns = columns
        
        # Apply mappings
        if 'TRL Categories' in display_df.columns:
            display_df['TRL Categories'] = display_df['TRL Categories'].map(stage_mappings)
        if 'Vertical' in display_df.columns:
            display_df['Vertical'] = display_df['Vertical'].map(vertical_mappings)
            
        st.dataframe(
            display_df.style.background_gradient(cmap='Blues'),
            use_container_width=True,
            height=400
        )
        
        # Download button
        csv = display_df.to_csv(index=False)
        st.download_button(
            label="Download Data as CSV",
            data=csv,
            file_name=f'ip_assets_{year}.csv',
            mime='text/csv',
        )

    else:
        st.warning(f"No data found for {year}.")

def get_all_years_data(data):
    # Set global figure parameters for better size control
    plt.rcParams['figure.dpi'] = 100
    plt.rcParams['savefig.dpi'] = 100
    
    # Define column ranges for different years
    year_columns = {
        2019: (2, 8),   
        2020: (13, 19),  
        2021: (24, 30),
        2022: (35, 41),
        2023: (47, 53),
        2024: (60, 66)
    }
    
    # Define vertical mappings
    vertical_mappings = {
        'TAFGAI (Fintech)': 'Financial & Risk',
        'TAFGAI': 'Financial & Risk',
        'SOFINAA (ESGtech)': 'Social Impact',
        'SOFINAA': 'Social Impact',
        'PAL/LIFT (Edutech)': 'Education',
        'PAL': 'Education',
        'MEDGIVER (Medtech)': 'Healthcare & Medical',
        'MEDGIVER': 'Healthcare & Medical',
        'NUURAI (Counsellingtech)': 'Mental & Spiritual Wellness',
        'NUURAI': 'Mental & Spiritual Wellness',
        'CONTRACKAI (Legaltech)': 'Legal',
        'CONTRACKAI': 'Legal',
        'STORQUE (Wealthtech)': 'Investment',
        'STORQUE': 'Investment'
    }
    
    stage_mappings = {
        'A': 'TRL 7-9',
        'B': 'TRL 4-6',
        'C': 'TRL 1-3'
    }

    st.markdown("## All Years Analysis")
    
    all_data = []
    all_stages = []
    all_verticals = []
    
    # Process data
    with st.spinner('Loading data from all years...'):
        for year, (start_col, end_col) in year_columns.items():
            year_data = data.iloc[28:, start_col:end_col].copy()
            year_data = year_data[year_data.iloc[:, 0].notna()]
            
            if not year_data.empty:
                year_data.columns = ['Project Name', 'Country', 'Project Owner', 'Vertical', 'Stage', 'Status']
                year_data['Year'] = year
                
                stages = year_data['Stage'].map(stage_mappings).dropna()
                all_stages.extend([{'Year': year, 'Stage': stage} for stage in stages])
                
                verticals = year_data['Vertical'].map(lambda x: next(
                    (v for k, v in vertical_mappings.items() if str(k) in str(x)), x if pd.notna(x) else None
                ))
                all_verticals.extend([{'Year': year, 'Vertical': v} for v in verticals if pd.notna(v)])
                
                all_data.append(year_data)
    
    if all_data:
        combined_df = pd.concat(all_data, ignore_index=True)
        stages_df = pd.DataFrame(all_stages)
        verticals_df = pd.DataFrame(all_verticals)
        
        # Metrics
        total_projects = len(stages_df)
        stage_counts = stages_df['Stage'].value_counts()
        vertical_counts = verticals_df['Vertical'].value_counts()
        
        # Display metrics
        metrics_cols = st.columns(4)
        metrics_cols[0].metric("TRL 7-9", stage_counts.get('TRL 7-9', 0))
        metrics_cols[1].metric("TRL 4-6", stage_counts.get('TRL 4-6', 0))
        metrics_cols[2].metric("TRL 1-3", stage_counts.get('TRL 1-3', 0))
        metrics_cols[3].metric("Total Projects", total_projects)
        
        # TRL Distribution
        trl_cols = st.columns(2)
        
        with trl_cols[0]:
            st.markdown("### Overall TRL Distribution")
            fig1, ax1 = plt.subplots(figsize=(6, 6))
            wedges, texts, autotexts = ax1.pie(
                stage_counts.values,
                labels=stage_counts.index,
                autopct='%1.1f%%',
                # colors=['#4e73df', '#1cc88a', '#36b9cc']
                colors = plt.cm.Set3(np.linspace(0, 1, len(vertical_counts)))
            )
            ax1.set_title('TRL Distribution (All Years)')
            st.pyplot(fig1)
            plt.close(fig1)
        
        with trl_cols[1]:
            st.markdown("### TRL Distribution by Year")
            stage_pivot = pd.crosstab(stages_df['Year'], stages_df['Stage'])
            fig2, ax2 = plt.subplots(figsize=(6, 4))
            stage_pivot.plot(kind='bar', stacked=True, ax=ax2)
            plt.title('TRL Distribution by Year')
            plt.xlabel('Year')
            plt.ylabel('Number of Projects')
            plt.legend(title='TRL Stage', bbox_to_anchor=(1.05, 1))
            plt.tight_layout()
            st.pyplot(fig2)
            plt.close(fig2)
        
        # Vertical Distribution
        st.markdown("### Vertical Distribution")
        vertical_cols = st.columns(2)
        
        vertical_colors = {
            'Financial & Risk': '#98FB98',
            'Social Impact': '#B0C4DE',
            'Education': '#FFE4B5',
            'Healthcare & Medical': '#FFA07A',
            'Mental & Spiritual Wellness': '#DDA0DD',
            'Legal': '#87CEEB',
            'Investment': '#F08080'
        }
        
        with vertical_cols[0]:
            st.markdown("### Overall Vertical Distribution")
            fig3, ax3 = plt.subplots(figsize=(6, 6))
            colors = [vertical_colors.get(v, '#808080') for v in vertical_counts.index]
            
            def make_autopct(values):
                def my_autopct(pct):
                    if pct < 4:
                        return f'{pct:.1f}%\n'
                    return f'{pct:.1f}%'
                return my_autopct
            
            wedges, texts, autotexts = ax3.pie(
                vertical_counts.values,
                labels=vertical_counts.index,
                autopct=make_autopct(vertical_counts.values),
                # colors=colors,
                colors = plt.cm.Set3(np.linspace(0, 1, len(vertical_counts))),
                startangle=90,
                textprops={'fontsize': 8},
                pctdistance=0.75
            )
            
            ax3.legend(
                wedges,
                vertical_counts.index,
                title="Verticals",
                loc="center left",
                bbox_to_anchor=(1, 0, 0.5, 1),
                fontsize=8
            )
            
            for text in texts:
                text.set_text('')
            
            plt.setp(autotexts, size=8, weight='bold')
            
            ax3.set_title('Vertical Distribution (All Years)', 
                         pad=10,
                         fontsize=10)
            
            plt.tight_layout()
            st.pyplot(fig3)
            plt.close(fig3)
            
        with vertical_cols[1]:
            st.markdown("### Vertical Distribution by Year")
            vertical_pivot = pd.crosstab(verticals_df['Year'], verticals_df['Vertical'])
            
            fig4, ax4 = plt.subplots(figsize=(6, 4))
            
            bottom = np.zeros(len(vertical_pivot))
            for column in vertical_pivot.columns:
                values = vertical_pivot[column].values
                ax4.bar(vertical_pivot.index, values, bottom=bottom,
                       label=column,
                    #    color=vertical_colors.get(column, '#808080')
                        color = plt.cm.Set3(np.linspace(0, 1, len(vertical_counts)))
                       )
                bottom += values
            
            ax4.set_title('Vertical Distribution by Year')
            ax4.set_xlabel('Year')
            ax4.set_ylabel('Number of Projects')
            
            for i in range(len(vertical_pivot.index)):
                total = vertical_pivot.iloc[i].sum()
                ax4.text(vertical_pivot.index[i], total, f'{int(total)}',
                        ha='center', va='bottom')
            
            ax4.legend(title='Vertical', bbox_to_anchor=(1.05, 1), loc='upper left')
            ax4.set_xticks(vertical_pivot.index)
            ax4.set_xticklabels(vertical_pivot.index, rotation=0)
            ax4.yaxis.grid(True, linestyle='--', alpha=0.7)
            
            plt.tight_layout()
            st.pyplot(fig4)
            plt.close(fig4)
        
        # Trend Analysis
        st.markdown("### Trend Analysis")
        trend_cols = st.columns(2)
        
        with trend_cols[0]:
            st.markdown("### TRL Stage Trends")
            fig5, ax5 = plt.subplots(figsize=(6, 4))
            
            for stage in stage_counts.index:
                stage_trend = stages_df[stages_df['Stage'] == stage].groupby('Year').size()
                ax5.plot(stage_trend.index, stage_trend.values, marker='o', label=stage)
            
            plt.title('TRL Stage Trends Over Time')
            plt.xlabel('Year')
            plt.ylabel('Number of Projects')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            st.pyplot(fig5)
            plt.close(fig5)
        
        with trend_cols[1]:
            st.markdown("### Projects by Year")
            yearly_totals = stages_df['Year'].value_counts().sort_index()
            fig6, ax6 = plt.subplots(figsize=(6, 4))
            bars = ax6.bar(yearly_totals.index, yearly_totals.values, color='#4e73df')
            
            for bar in bars:
                height = bar.get_height()
                ax6.text(bar.get_x() + bar.get_width()/2., height,
                        f'{int(height)}', ha='center', va='bottom')
            
            plt.title('Total Projects by Year')
            plt.xlabel('Year')
            plt.ylabel('Number of Projects')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            st.pyplot(fig6)
            plt.close(fig6)

        # Add multiple analysis buttons side by side
        st.markdown("### Analysis Options")
        button_cols = st.columns(3)
        
        with button_cols[0]:
            summary_button = st.button("📊 Portfolio Overview", key="all_years_summary")
        with button_cols[1]:
            trl_button = st.button("🔬 TRL Trend Analysis", key="all_years_trl")
        with button_cols[2]:
            recommendations_button = st.button("💡 Strategic Insights", key="all_years_recommendations")

        # Function to generate specific analysis
        def generate_analysis(analysis_type):
            prompts = {
                "summary": f"""
                Provide a comprehensive overview of the IP portfolio across all years:
                - Total Projects: {total_projects}
                - Current TRL Distribution: {dict(stage_counts)}
                - Vertical Distribution: {dict(vertical_counts)}
                - Yearly Project Counts: {dict(stages_df['Year'].value_counts().sort_index())}
                
                Focus on:
                1. Overall portfolio evolution
                2. Key growth patterns
                3. Major shifts in portfolio composition
                4. Notable achievements and milestones
                
                Highlight the most significant trends and changes over time.  Express the figures in % terms for the year and across the years comparatively. Analyze the % composition of each stage in the TRL (Technology Readiness Level, comparing them with the rest of the stages. Describe the concentration and distribution patterns, highlighting any significant variations or trends.
                """,
                
                "trl": f"""
                Analyze the TRL (Technology Readiness Level) trends across all years:
                - TRL Distribution: {dict(stage_counts)}
                - Yearly TRL Breakdown: {dict(pd.crosstab(stages_df['Year'], stages_df['Stage']).to_dict())}
                - Total Projects: {total_projects}
                
                Provide insights on:
                1. Evolution of portfolio maturity
                2. Year-over-year progression patterns
                3. Success in project advancement
                4. Areas needing acceleration
                5. Risk assessment across TRL stages
                
                Focus on long-term patterns and strategic implications.Express the figures in % terms for the year and across the years comparatively. Analyze the % composition of each stage in the TRL (Technology Readiness Level, comparing them with the rest of the stages. Describe the concentration and distribution patterns, highlighting any significant variations or trends.
                """,
                
                "recommendations": f"""
                Based on the complete portfolio data:
                - Vertical Distribution: {dict(vertical_counts)}
                - TRL Stage Progress: {dict(stage_counts)}
                - Yearly Growth: {dict(stages_df['Year'].value_counts().sort_index())}
                
                Provide strategic recommendations covering:
                1. Long-term investment strategy
                2. Portfolio optimization opportunities
                3. Risk management across verticals
                4. Growth acceleration opportunities
                5. Strategic focus areas for future development
                
                Make recommendations specific, actionable, and based on observed trends with supporting facts and figures in figures, % terms and years quoted accordingly.
                """
            }
            
            try:
                client = openai.OpenAI(api_key="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")
                
                with st.spinner(f"Generating {analysis_type.replace('_', ' ')}..."):
                    response = client.chat.completions.create(
                        model="gpt-4",
                        messages=[
                            {"role": "system", "content": "You are a senior IP portfolio analyst specializing in technology and innovation assessment with expertise in long-term trend analysis."},
                            {"role": "user", "content": prompts[analysis_type]}
                        ],
                        max_tokens=800,
                        temperature=0.7
                    )
                    
                    # Updated styling with larger text and better spacing
                    st.markdown("""
                        <style>
                            .analysis-container {
                                padding: 2rem;
                                background-color: #f8f9fa;
                                border-radius: 0.5rem;
                                box-shadow: 0 0.15rem 1.75rem 0 rgba(58, 59, 69, 0.15);
                                margin: 1rem 0;
                                font-size: 1.1rem;
                                line-height: 1.6;
                            }
                            .analysis-container p {
                                margin-bottom: 1rem;
                                font-size: 18px;
                            }
                            .analysis-container ul, .analysis-container ol {
                                margin-bottom: 1rem;
                                font-size: 18px;
                                padding-left: 2rem;
                            }
                            .analysis-container li {
                                margin-bottom: 0.5rem
                            }
                            .analysis-container strong, .analysis-container b {
                                font-weight: 600;
                            }
                            .analysis-container h3, .analysis-container h4 {
                                margin-top: 1.5rem;
                                margin-bottom: 1rem;
                                font-size: 20px;
                                font-weight: 600;
                            }
                        </style>
                    """, unsafe_allow_html=True)
                    
                    st.markdown(f"""
                        <div class="analysis-container">
                            {response.choices[0].message.content}
                        </div>
                    """, unsafe_allow_html=True)
                    st.markdown("---")

            except Exception as e:
                st.error(f"Error generating analysis: {str(e)}")

        # Handle button clicks
        if summary_button:
            generate_analysis("summary")
        if trl_button:
            generate_analysis("trl")
        if recommendations_button:
            generate_analysis("recommendations")

        # Display data table with improved styling
        st.markdown("### Detailed Data View")
        display_df = combined_df.copy()
        
        st.dataframe(
            display_df.style.background_gradient(cmap='Blues'),
            use_container_width=True,
            height=400
        )
        
        # Download button
        csv = display_df.to_csv(index=False)
        st.download_button(
            label="Download Complete Data as CSV",
            data=csv,
            file_name='all_years_ip_assets.csv',
            mime='text/csv',
        )
    else:
        st.warning("No data found for analysis.")
    
    return combined_df

def main():
    # Set page configuration
    st.set_page_config(
        page_title="Analystics Dashboard",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Custom CSS for better styling
    st.markdown("""
        <style>
        .main {
            padding: 0rem 1rem;
        }
        .stMetric {
            background-color: #f8f9fa;
            padding: 1rem;
            border-radius: 0.5rem;
            box-shadow: 0 0.15rem 1.75rem 0 rgba(58, 59, 69, 0.15);
        }
        .st-emotion-cache-1v0mbdj img {
            margin-bottom: 1rem;
            border-radius: 0.5rem;
            box-shadow: 0 0.15rem 1.75rem 0 rgba(58, 59, 69, 0.15);
        }
        </style>
        """, unsafe_allow_html=True)

    # Sidebar with logo and navigation
    with st.sidebar:
        st.title("📊 IP Assets Analytics")
        st.markdown("---")
        st.subheader("Navigation")
        st.markdown("Use the tabs below to explore different views of the IP Assets data.")
        st.markdown("---")
        st.markdown("### About")
        st.info("This dashboard provides comprehensive analytics of IP Assets from 2019 to 2024.")

    # Load data
    data = load_data()

    # Main tabs with icons
    tab1, tab2, tab3, tab4 = st.tabs([
        "📈 Overview", 
        "📅 Yearly Analysis", 
        "📊 Cumulative Analysis",
        "⚙️ Manage Assets"
    ])

    with tab1:
        st.markdown("## IP Assets Overview Dashboard")
        st.markdown("---")
        
        filtered_data = data.loc[data['Unnamed: 1'].isin([2019, 2020, 2021, 2022, 2023, 2024]), 
                               ['Unnamed: 1', 'No. of IP Assets']]
        filtered_data.columns = ['Year', 'No. of IP Assets']
        filtered_data['No. of IP Assets'] = pd.to_numeric(filtered_data['No. of IP Assets'], errors='coerce')
        filtered_data = filtered_data.groupby('Year')['No. of IP Assets'].sum().reset_index()
        filtered_data = filtered_data.dropna()
        total_assets = filtered_data['No. of IP Assets'].sum()

        # KPI metrics in a row
        kpi1, kpi2, kpi3 = st.columns(3)
        with kpi1:
            st.metric(
                "Total IP Assets",
                f"{total_assets:,}",
                "All Years Combined"
            )
        with kpi2:
            current_year = filtered_data.iloc[-1]['Year']
            current_year_assets = filtered_data.iloc[-1]['No. of IP Assets']
            st.metric(
                f"Current Year ({int(current_year)})",
                f"{current_year_assets:,}",
                "Latest Count"
            )
        with kpi3:
            yoy_growth = ((current_year_assets - filtered_data.iloc[-2]['No. of IP Assets']) / 
                         filtered_data.iloc[-2]['No. of IP Assets'] * 100)
            st.metric(
                "Year-over-Year Growth",
                f"{yoy_growth:.1f}%",
                "vs Last Year"
            )

        st.markdown("---")

        # Visualizations
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("### Annual IP Assets Trend")
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.bar(filtered_data['Year'], filtered_data['No. of IP Assets'], 
                  color='#4e73df', alpha=0.8)
            ax.set_title('IP Assets Distribution Over Years', pad=20)
            ax.set_xlabel('Year')
            ax.set_ylabel('Number of IP Assets')
            ax.grid(True, alpha=0.3)
            
            for i, v in enumerate(filtered_data['No. of IP Assets']):
                ax.text(filtered_data['Year'].iloc[i], v, f'{v:,}', 
                       ha='center', va='bottom')

            plt.tight_layout()
            st.pyplot(fig)

        with col2:
            st.markdown("### Data Summary")
            st.dataframe(
                filtered_data.style.format({'No. of IP Assets': '{:,}'}),
                hide_index=True,
                use_container_width=True
            )

    with tab2:
        st.markdown("## Yearly Detailed Analysis")
        st.markdown("---")
        
        col1, col2 = st.columns([1, 2])
        with col1:
            selected_year = st.selectbox(
                "Select Year for Analysis",
                [2019, 2020, 2021, 2022, 2023, 2024],
                format_func=lambda x: f"Year {x}"
            )
        
        print_year_data(data, selected_year)

    with tab3:
            st.markdown("## Cumulative Analysis")
            st.markdown("---")
            cumulative_df = get_all_years_data(data)
    with tab4:
        crud_operations(data)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")

# Add this in the sidebar
with st.sidebar:
    dark_mode = st.toggle("🌙 Dark Mode")
    if dark_mode:
        st.markdown("""
            <style>
            .main {
                background-color: #1a1a1a;
                color: white;
            }
            .stMetric {
                background-color: #2d2d2d;
                color: white;
            }
            </style>
            """, unsafe_allow_html=True)
