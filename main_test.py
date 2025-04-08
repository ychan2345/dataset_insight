import streamlit as st
import pandas as pd
import json
from utils import prepare_gpt_prompt, prepare_overview_prompt, get_gpt_analysis, prepare_columns_batch_prompt
from typing import Dict, Any, Tuple
import os
from dotenv import load_dotenv
import streamlit.components.v1 as components
import time
load_dotenv()
# Set up run counter for debugging
if 'run_counter' not in st.session_state:
    st.session_state.run_counter = 0
st.session_state.run_counter += 1
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
def get_confidence_indicator(confidence_score: float) -> str:
    """Return an emoji indicator based on confidence score"""
    if confidence_score >= 0.9:
        return "🟢"  # Green circle for high confidence
    elif confidence_score >= 0.8:
        return "🟡"  # Yellow circle for medium confidence
    else:
        return "🔴"  # Red circle for low confidence
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
    
    # Add confidence filter above the table
    if st.session_state.analysis["columns"]:
        # First, convert the AI response to a DataFrame for easier filtering
        
        # Only create the columns list and DataFrame if it doesn't exist in session state
        if 'analysis_columns_df' not in st.session_state:
            # Create a list to hold all columns data
            columns_list = []
            
            # Process all columns from the AI response only
            for col in st.session_state.analysis['columns']:
                column_name = col['name']
                column_title = col['title']
                confidence = col.get('confidence_score', 0)
                description = col['description']
                
                # Use any type information in the AI response, or mark as "Unknown"
                dtype = col.get('data_type', 'Unknown')
                
                # Use any missing percentage in the AI response, or default to "N/A"
                missing_str = col.get('missing_percentage', 'N/A')
                if isinstance(missing_str, (int, float)):
                    missing_str = f"{missing_str:.2f}%"
                missing_pct = 0  # Default value for sorting
                
                # Store all column data in a dictionary
                columns_list.append({
                    'Column Name': column_name,
                    'Column Title': column_title,
                    'Data Type': dtype,
                    'Missing %': missing_str,
                    'Missing Pct': missing_pct,  # For sorting
                    'Confidence Score': f"{get_confidence_indicator(confidence)} {confidence:.2%}",
                    'Confidence Value': confidence,  # For filtering
                    'Column Description': description
                })
            
            # Convert to DataFrame and store in session state
            if columns_list:  # Only create DataFrame if we have columns
                st.session_state.analysis_columns_df = pd.DataFrame(columns_list)
        
        # Debug information - show min/max confidence values in the dataframe
        try:
            st.write("Debug - Confidence value range:", 
                    st.session_state.analysis_columns_df['Confidence Value'].min(), 
                    "to", 
                    st.session_state.analysis_columns_df['Confidence Value'].max())
                    
            # Add histogram for debugging
            confidence_bins = {
                "≥0.9": (st.session_state.analysis_columns_df['Confidence Value'] >= 0.9).sum(),
                "0.8-0.9": ((st.session_state.analysis_columns_df['Confidence Value'] >= 0.8) & 
                          (st.session_state.analysis_columns_df['Confidence Value'] < 0.9)).sum(),
                "<0.8": (st.session_state.analysis_columns_df['Confidence Value'] < 0.8).sum()
            }
            st.write("Debug - Confidence distribution:", confidence_bins)
        except Exception as e:
            st.warning(f"Debug info error: {str(e)}")
        
        # Store the selected filter in session state
        if 'selected_confidence_filter' not in st.session_state:
            st.session_state.selected_confidence_filter = "All Columns"
        
        # Filter options for confidence levels using radio buttons
        confidence_filter = st.radio(
            "Filter by Confidence Level:",
            options=[
                "All Columns",
                "🟢 High Confidence (>90%)",
                "🟡 Medium Confidence (80-90%)",
                "🔴 Low Confidence (<80%)"
            ],
            index=["All Columns", "🟢 High Confidence (>90%)", "🟡 Medium Confidence (80-90%)", "🔴 Low Confidence (<80%)"].index(st.session_state.selected_confidence_filter),
            horizontal=True,
            key="confidence_filter_radio"
        )
        
        # Update the session state value
        st.session_state.selected_confidence_filter = confidence_filter
        
        try:
            # Apply DataFrame filtering based on selection - always use session state DataFrame
            if confidence_filter == "🟢 High Confidence (>90%)":
                filtered_df = st.session_state.analysis_columns_df[st.session_state.analysis_columns_df['Confidence Value'] >= 0.9]
            elif confidence_filter == "🟡 Medium Confidence (80-90%)":
                filtered_df = st.session_state.analysis_columns_df[
                    (st.session_state.analysis_columns_df['Confidence Value'] >= 0.8) & 
                    (st.session_state.analysis_columns_df['Confidence Value'] < 0.9)
                ]
            elif confidence_filter == "🔴 Low Confidence (<80%)":
                filtered_df = st.session_state.analysis_columns_df[st.session_state.analysis_columns_df['Confidence Value'] < 0.8]
            else:
                # "All Columns" option
                filtered_df = st.session_state.analysis_columns_df.copy()
            
            # Remove the numeric columns used for filtering/sorting before display
            display_df = filtered_df.drop(columns=['Confidence Value', 'Missing Pct'])
                
            # Display the dataframe with filter stats
            if confidence_filter == "All Columns":
                st.info(f"Showing all {len(display_df)} columns")
            else:
                total_columns = len(st.session_state.analysis_columns_df)
                st.info(f"Showing {len(display_df)} of {total_columns} columns matching filter: {confidence_filter}")
            
            # Always show the DataFrame, even if empty
            st.dataframe(display_df, use_container_width=True)
            
            # Add a message if the filtered DataFrame is empty
            if len(display_df) == 0:
                st.warning("No columns match the selected filter criteria.")
            # Export Column Details - only export the filtered view
            if confidence_filter == "All Columns":
                export_filename = "column_details.csv"
            else:
                # Create a filename based on the filter
                filter_key = confidence_filter.split(" ")[0]  # Get the emoji
                export_filename = f"column_details_{filter_key}.csv"
                
            st.download_button(
                f"Export Filtered Columns ({len(display_df)} columns)",
                display_df.to_csv(index=False),
                export_filename,
                "text/csv"
            )
        except Exception as e:
            st.error(f"Error during filtering: {str(e)}")
            st.write("Session state keys:", list(st.session_state.keys()))
            if 'analysis_columns_df' in st.session_state:
                st.write("DataFrame columns:", list(st.session_state.analysis_columns_df.columns))
            else:
                st.warning("analysis_columns_df not found in session state")
    # Key Observations
    st.markdown("#### Key Observations")
    for observation in st.session_state.analysis["key_observations"]:
        st.markdown(f"- {observation}")
def get_column_stats(df):
    """Import and call the get_column_stats function from utils.py"""
    from utils import get_column_stats as gcs
    return gcs(df)
def main():
    st.set_page_config(
        page_title="AI-Driven Data Insights Powered by LLMs",
        page_icon="🤖",
        layout="wide"
    )
    st.title("🤖 AI-Powered Data Analysis and Curation")
    # Initialize session state variables
    if 'overview_prompt' not in st.session_state:
        st.session_state.overview_prompt = None
    if 'stats' not in st.session_state:
        st.session_state.stats = None
    if 'analysis' not in st.session_state:
        st.session_state.analysis = None
    if 'prompt' not in st.session_state:
        st.session_state.prompt = None
    if 'summary_df' not in st.session_state:
        st.session_state.summary_df = None
    if 'app_mode' not in st.session_state:
        st.session_state.app_mode = "Home"
    if 'has_analysis_results' not in st.session_state:
        st.session_state.has_analysis_results = False
    # Debug information in sidebar
    st.sidebar.write(f"App has run {st.session_state.run_counter} times")
    
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
    # App description - only show in Home mode
    if st.session_state.app_mode == "Home":
        st.sidebar.markdown("""
        ### About This App
        
        This tool analyzes system tables using AI to provide intelligent insights:
        
        - **Statistical Analysis**: Automatically calculates data statistics
        - **Column Analysis**: Identifies column purpose and data meaning
        - **Missing Data Detection**: Shows percentage of missing values
        - **Confidence Indicators**: 🟢 >90%, 🟡 80-90%, 🔴 <80%
        - **AI-Powered Insights**: Suggests analytics approaches
        
        The app can handle large datasets (200+ columns) by using batched processing
        and optimized token usage.
        """)
    if st.session_state.app_mode == "Home":
        # Home view - display application overview and instructions
        
        st.markdown("""
        ## Welcome to the AI-Driven Data Insights Tool
        
        This application helps you understand and analyze system tables using artificial 
        intelligence to generate comprehensive insights about your data.
        
        ### Key Features
        
        - **Automated Data Analysis**: Get instant statistics and insights about your data
        - **AI Column Interpretation**: The system identifies the purpose and meaning of each column
        - **Missing Data Detection**: Automatically calculates the percentage of missing values
        - **Batch Processing**: Handles large datasets with hundreds of columns efficiently
        - **Confidence Scoring**: Each interpretation includes a confidence indicator
        - **Exportable Results**: Download analysis results as CSV
        
        ### How to Use
        
        1. Select "Data Curation App" from the navigation dropdown menu
        2. Choose one of the available datasets
        3. Click "Populate Data with AI Analysis" to generate insights
        """)
    
    elif st.session_state.app_mode == "Data Curation App":
        # Data Curation view
        st.markdown("## Data Curation App")
        
        st.markdown("### Select a Dataset to Analyze")
        
        # NOTE: Here we separate the form submission from the filtering logic
        # FORM - Just for dataset selection and analysis generation
        with st.form(key="dataset_selection_form"):
            # Let user select dataset
            dataset_option = st.selectbox(
                "Choose a dataset:",
                options=["table1", "table2", "sample_data_ref", "sample_180_columns_100_rows", "sample_200_columns_100_rows"]
            )
            
            # Form submit button for running the analysis
            submit_button = st.form_submit_button("Populate Data with AI Analysis")
            
            if submit_button:
                st.session_state.last_submission_time = pd.Timestamp.now()
                st.sidebar.write(f"Last form submission: {st.session_state.last_submission_time}")
                
                # Show progress
                with st.spinner(f"Analyzing dataset {dataset_option}... This may take a minute for large datasets."):
                    # Process the selected dataset
                    start_time = time.time()
                    
                    # File path based on selection
                    file_path = f"attached_assets/{dataset_option}.csv"
                    
                    # Get API key from environment
                    api_key = os.environ.get("AZURE_OPENAI_API_KEY")
                    endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT")
                    
                    # Process file
                    try:
                        from utils import process_file_path
                        df, stats, analysis, prompt = process_file_path(file_path, api_key, endpoint)
                        
                        # Store results in session state
                        st.session_state.df = df
                        st.session_state.stats = stats
                        st.session_state.analysis = analysis
                        st.session_state.prompt = prompt
                        
                        # Create summary DataFrame
                        from summary import create_summary_df
                        st.session_state.summary_df = create_summary_df(df, dataset_option)
                        
                        # Mark that we have results
                        st.session_state.has_analysis_results = True
                        
                        end_time = time.time()
                        st.success(f"Analysis completed in {end_time - start_time:.2f} seconds!")
                    except Exception as e:
                        st.error(f"Error processing file: {str(e)}")
        
        # OUTSIDE THE FORM - Display analysis results if available
        # This ensures filtering doesn't cause resubmission
        if st.session_state.get('has_analysis_results', False):
            # Display the summary information
            if hasattr(st.session_state, 'summary_df'):
                display_summary(st.session_state.summary_df)
            
            # Display the GPT analysis
            if hasattr(st.session_state, 'analysis'):
                display_analysis()
