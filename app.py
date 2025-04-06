import streamlit as st
import pandas as pd
import json
import os
from utils import process_file_path
from summary import create_summary_df

def get_confidence_indicator(confidence_score: float) -> str:
    """Return an emoji indicator based on confidence score"""
    if confidence_score >= 0.9:
        return "🟢"  # Green circle for high confidence
    elif confidence_score >= 0.8:
        return "🟡"  # Yellow circle for medium confidence
    else:
        return "🔴"  # Red circle for low confidence

def main():
    st.set_page_config(
        page_title="CSV Analysis Tool",
        page_icon="📊",
        layout="wide"
    )

    st.title("📈 AI-Driven Data Insights Powered by LLMs")

    # Initialize session state variables
    if 'last_prompt' not in st.session_state:
        st.session_state.last_prompt = None
    if 'df' not in st.session_state:
        st.session_state.df = None
    if 'stats' not in st.session_state:
        st.session_state.stats = None
    if 'analysis' not in st.session_state:
        st.session_state.analysis = None
    if 'prompt' not in st.session_state:
        st.session_state.prompt = None
    if 'table_name' not in st.session_state:
        st.session_state.table_name = "dataset"
    if 'summary_df' not in st.session_state:
        st.session_state.summary_df = None
    if 'file_loaded' not in st.session_state:
        st.session_state.file_loaded = False
    if 'selected_table' not in st.session_state:
        st.session_state.selected_table = "sample_200_columns_100_rows.csv"
    if 'app_mode' not in st.session_state:
        st.session_state.app_mode = "Home"

    # Set up the sidebar
    st.sidebar.title("Navigation")
    
    # App mode selection
    app_mode = st.sidebar.selectbox(
        "Select Mode:",
        options=["Home", "Data Curation App"],
        key="app_mode_selector"
    )
    
    # Update the session state with the selected mode
    st.session_state.app_mode = app_mode
    
    # Horizontal line separator
    st.sidebar.markdown("---")
    
    # App description - only show in Home mode
    if st.session_state.app_mode == "Home":
        st.sidebar.markdown("""
        ### About This App
        
        This tool analyzes CSV datasets using AI to provide intelligent insights:
        
        - **Statistical Analysis**: Automatically calculates data statistics
        - **Column Analysis**: Identifies column purpose and data meaning
        - **Missing Data Detection**: Shows percentage of missing values
        - **Confidence Indicators**: 🟢 >90%, 🟡 80-90%, 🔴 <80%
        - **AI-Powered Insights**: Suggests analytics approaches
        
        The app can handle large datasets (200+ columns) by using batched processing
        and optimized token usage.
        """)
    
    # Horizontal line separator
    st.sidebar.markdown("---")
    
    # Table selection options
    table_files = {
        "Sample Dataset (200 columns)": {
            "path": "attached_assets/sample_200_columns_100_rows.csv",
            "table_name": "sample_dataset"
        },
        "Table 1": {
            "path": "attached_assets/table1.csv",
            "table_name": "table1"
        },
        "Table 2": {
            "path": "attached_assets/table2.csv",
            "table_name": "table2"
        }
    }
    
    # Only show dataset selection in Data Curation mode
    file_path = None
    table_name = None
    
    if st.session_state.app_mode == "Data Curation App":
        st.sidebar.subheader("Dataset Configuration")
        
        # Select the table file
        selected_table_option = st.sidebar.selectbox(
            "Select Dataset:",
            options=list(table_files.keys())
        )
        
        # Get the file path and table name for the selected table
        file_path = table_files[selected_table_option]["path"]
        table_name = table_files[selected_table_option]["table_name"]
    
    # Only process data in Data Curation mode and if a file path is selected
    if st.session_state.app_mode == "Data Curation App" and file_path is not None:
        # Auto-load data when table selection changes
        if file_path != st.session_state.selected_table:
            st.session_state.selected_table = file_path
            try:
                # Validate file content
                df = pd.read_csv(file_path)
                if df.empty:
                    st.error("The CSV file appears to be empty.")
                elif len(df.columns) == 0:
                    st.error("No columns found in the CSV file.")
                else:
                    # Save to session state
                    st.session_state.df = df
                    st.session_state.stats = get_column_stats(df)
                    st.session_state.file_loaded = True
                    
                    # Automatically create summary with the table name
                    if table_name is not None:
                        try:
                            st.session_state.summary_df = create_summary_df(df, table_name)
                            st.session_state.table_name = table_name
                        except Exception as e:
                            st.error(f"Error creating summary: {str(e)}")
                    else:
                        st.error("Table name is missing. Cannot create summary.")
                    
                    # Reset analysis
                    st.session_state.analysis = None
            except Exception as e:
                st.error(f"Error loading data: {str(e)}")
        
        # Load the initial table data if not already loaded
        if not st.session_state.file_loaded and file_path is not None:
            try:
                # Validate file content
                df = pd.read_csv(file_path)
                if df.empty:
                    st.error("The CSV file appears to be empty.")
                elif len(df.columns) == 0:
                    st.error("No columns found in the CSV file.")
                else:
                    # Save to session state
                    st.session_state.df = df
                    st.session_state.stats = get_column_stats(df)
                    st.session_state.file_loaded = True
                    
                    # Automatically create summary with the table name
                    if table_name is not None:
                        try:
                            st.session_state.summary_df = create_summary_df(df, table_name)
                            st.session_state.table_name = table_name
                        except Exception as e:
                            st.error(f"Error creating summary: {str(e)}")
                    else:
                        st.error("Table name is missing. Cannot create summary.")
            except Exception as e:
                st.error(f"Error loading data: {str(e)}")
    elif st.session_state.app_mode == "Home":
        # In home mode, we don't need to load any data
        pass
    else:
        # No valid mode or missing file path
        st.session_state.file_loaded = False
    
    # Analysis section
    st.sidebar.markdown("---")
    
    # Get API Key from environment or let user input it
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        api_key = st.sidebar.text_input("OpenAI API Key", type="password")
        st.sidebar.caption("Your API key is not stored and only used for this analysis.")
    
    # We don't need the endpoint for OpenAI API
    endpoint = ""
    
    # Show Populate Data button only in Data Curation mode
    if st.session_state.app_mode == "Data Curation App" and file_path is not None:
        # Populate Data button
        if st.sidebar.button("Populate Data with AI Analysis", type="primary"):
            if not api_key:
                st.error("OpenAI API key not found. Please provide a valid API key.")
            elif file_path is None:
                st.error("No dataset selected. Please select a dataset first.")
            else:
                try:
                    with st.spinner("Analyzing data with AI..."):
                        # Show a progress message during processing
                        progress_placeholder = st.empty()
                        progress_placeholder.info("Processing data - this may take a moment for large datasets...")
                        
                        # Process the file using the loaded dataframe
                        # Use a valid file path (file_path is not None at this point)
                        file_path_str = str(file_path)
                        _, _, analysis, prompt = process_file_path(
                            file_path_str,
                            api_key=api_key,
                            endpoint=endpoint
                        )
                        
                        # Store results in session state
                        st.session_state.analysis = analysis
                        st.session_state.prompt = prompt
                        st.session_state.last_prompt = prompt
                        
                        # Replace progress message when done
                        progress_placeholder.empty()
                        st.success("AI analysis completed successfully!")
                except Exception as e:
                    st.error(f"An error occurred during analysis: {str(e)}")
                    st.write("Please check the console logs for more details.")
    
    # Main Content Area
    if st.session_state.app_mode == "Home":
        # Home view - display application overview and instructions
        st.header("📊 AI-Powered Data Analysis and Curation")
        
        st.markdown("""
        ## Welcome to the AI-Driven Data Insights Tool
        
        This application helps you understand and analyze CSV datasets using artificial 
        intelligence to generate comprehensive insights about your data.
        
        ### Key Features
        
        - **Automated Data Analysis**: Get instant statistics and insights about your data
        - **AI Column Interpretation**: The system identifies the purpose and meaning of each column
        - **Missing Data Detection**: Automatically calculates the percentage of missing values
        - **Batch Processing**: Handles large datasets with hundreds of columns efficiently
        - **Confidence Scoring**: Each interpretation includes a confidence indicator
        - **Exportable Results**: Download analysis results as CSV and JSON
        
        ### How to Use
        
        1. Select "Data Curation App" from the navigation dropdown above
        2. Choose one of the available datasets
        3. Click "Populate Data with AI Analysis" to generate insights
        4. Explore the generated metadata and column interpretations
        
        ### Confidence Indicators
        
        The system uses color-coded confidence indicators:
        - 🟢 High confidence (>90%)
        - 🟡 Medium confidence (80-90%)
        - 🔴 Low confidence (<80%)
        """)
        
        # Add some visuals
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Supported Columns", "200+", "Batch Processing")
        with col2:
            st.metric("Processing Time", "~30 sec", "for 100 columns")
        with col3:
            st.metric("Analysis Types", "3", "Description, Observations, Suggestions")
            
    elif st.session_state.app_mode == "Data Curation App" and file_path is not None:
        if st.session_state.file_loaded and hasattr(st.session_state, 'df') and st.session_state.df is not None:
            # Show data preview
            st.header("Data Preview")
            st.dataframe(st.session_state.df.head(), use_container_width=True)
            st.info(f"File contains {len(st.session_state.df)} rows and {len(st.session_state.df.columns)} columns.")
            
            # Display summary if available (should always be available now)
            if hasattr(st.session_state, 'summary_df') and st.session_state.summary_df is not None:
                display_summary(st.session_state.summary_df)
            
            # Display analysis if available
            if hasattr(st.session_state, 'analysis') and st.session_state.analysis is not None:
                display_analysis()
            else:
                # Display instructions if no analysis has been run
                st.info("Click 'Populate Data with AI Analysis' to get AI-generated insights about this dataset.")
                st.markdown("""
                ## Analysis Process:
                1. The dataset summary is automatically generated when you select a dataset
                2. Run the AI analysis to get intelligent insights and column interpretations
                """)
        else:
            # Instructions when no data is loaded
            st.warning("There was an issue loading the dataset. Please check the error messages above.")
    else:
        # Fallback message if no valid mode is selected
        st.info("Please select a mode from the navigation dropdown in the sidebar.")

def display_summary(summary_df):
    """Display the summary DataFrame with dataset information"""
    st.header("Dataset Summary")
    
    # Display key metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Table Name", summary_df["Table Name"].iloc[0])
    with col2:
        st.metric("Total Rows", summary_df["Total Rows"].iloc[0])
    with col3:
        st.metric("Total Columns", summary_df["Total Columns"].iloc[0])
    
    # More metrics about column types
    col4, col5, col6 = st.columns(3)
    with col4:
        st.metric("Numeric Columns", summary_df["Numeric Columns"].iloc[0])
    with col5:
        st.metric("Categorical Columns", summary_df["Categorical Columns"].iloc[0])
    with col6:
        st.metric("Datetime Columns", summary_df["Datetime Columns"].iloc[0])
    
    # Sample data in expander
    with st.expander("View Sample Data"):
        try:
            sample_data = json.loads(summary_df["Sample Data"].iloc[0])
            sample_df = pd.DataFrame(sample_data)
            st.dataframe(sample_df, use_container_width=True)
        except Exception as e:
            st.error(f"Could not display sample data: {str(e)}")
    
    # Export functionality
    st.download_button(
        "Export Summary (JSON)",
        summary_df.to_json(orient='records'),
        f"{summary_df['Table Name'].iloc[0]}_summary.json",
        "application/json"
    )

def display_analysis():
    """Display the GPT analysis results"""
    st.header("AI Analysis Results")
    
    # Show the GPT prompts in an expander
    with st.expander("Show GPT Prompts", expanded=False):
        st.markdown("#### GPT Prompts Information")
        st.markdown("For large datasets, the analysis is done in multiple stages:")
        
        st.markdown("##### Overview Analysis Prompt")
        st.markdown("This prompt is used to analyze the overall dataset characteristics:")
        if hasattr(st.session_state, 'overview_prompt'):
            st.code(st.session_state.overview_prompt, language="text")
        else:
            st.code(st.session_state.prompt, language="text")

    st.subheader("GPT-4 Generated Insights")

    # Dataset Description
    st.markdown("#### Description of the Dataset")
    st.write(st.session_state.analysis["dataset_description"])

    # Suggested Analysis
    st.markdown("#### Suggested Analysis")
    for suggestion in st.session_state.analysis["suggested_analysis"]:
        st.markdown(f"- {suggestion}")

    # Column Details Table with Export
    st.markdown("#### Column Details")
    if st.session_state.analysis["columns"]:
        column_data = {
            'Column Name': [],
            'Column Title': [],
            'Data Type': [],
            'Missing %': [],
            'Confidence Score': [],
            'Column Description': []
        }

        # Get column metadata from summary_df if available
        column_metadata = {}
        if hasattr(st.session_state, 'summary_df') and 'Column Metadata' in st.session_state.summary_df.columns:
            try:
                # Parse the column metadata from JSON string
                metadata_json = st.session_state.summary_df['Column Metadata'].iloc[0]
                column_metadata = json.loads(metadata_json)
            except Exception as e:
                st.warning(f"Could not parse column metadata: {str(e)}")
                column_metadata = {"dtypes": {}, "missing_percentages": {}}

        for col in st.session_state.analysis['columns']:
            column_name = col['name']
            column_data['Column Name'].append(column_name)
            column_data['Column Title'].append(col['title'])
            
            # Get data type from stored metadata if available
            if column_metadata and "dtypes" in column_metadata and column_name in column_metadata["dtypes"]:
                dtype = column_metadata["dtypes"].get(column_name, "Unknown")
            else:
                # Fallback to directly accessing DataFrame
                try:
                    dtype = str(st.session_state.df[column_name].dtype) if column_name in st.session_state.df.columns else 'N/A'
                except:
                    dtype = 'Unknown'
            column_data['Data Type'].append(dtype)
            
            # Get missing percentage from stored metadata if available
            if column_metadata and "missing_percentages" in column_metadata and column_name in column_metadata["missing_percentages"]:
                missing_pct = column_metadata["missing_percentages"].get(column_name, 0)
                column_data['Missing %'].append(f"{missing_pct:.2f}%")
            else:
                # Fallback to directly calculating from DataFrame
                try:
                    missing_pct = st.session_state.df[column_name].isna().mean() * 100 if column_name in st.session_state.df.columns else 0
                    column_data['Missing %'].append(f"{missing_pct:.2f}%")
                except:
                    column_data['Missing %'].append("N/A")
                
            confidence = col.get('confidence_score', 0)
            column_data['Confidence Score'].append(f"{get_confidence_indicator(confidence)} {confidence:.2%}")
            column_data['Column Description'].append(col['description'])

        # Display the dataframe
        column_df = pd.DataFrame(column_data)
        st.dataframe(column_df, use_container_width=True)

        # Export Column Details
        st.download_button(
            "Export Column Details (CSV)",
            column_df.to_csv(index=False),
            "column_details.csv",
            "text/csv"
        )

    # Key Observations
    st.markdown("#### Key Observations")
    for observation in st.session_state.analysis["key_observations"]:
        st.markdown(f"- {observation}")

def get_column_stats(df):
    """Import and call the get_column_stats function from utils.py"""
    from utils import get_column_stats as gcs
    return gcs(df)

if __name__ == "__main__":
    main()
