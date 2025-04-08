import streamlit as st
import pandas as pd
import json
from utils import prepare_gpt_prompt, prepare_overview_prompt, get_gpt_analysis, prepare_columns_batch_prompt
from typing import Dict, Any, Tuple
import os
from dotenv import load_dotenv
import streamlit.components.v1 as components
import dataiku
from dataiku import pandasutils as pdu
from langchain_openai import AzureChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema import HumanMessage, SystemMessage

load_dotenv()

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
                
        # Only create the columns list and DataFrame if it doesn't exist in session state
        if 'analysis_columns_df' not in st.session_state:
            # Create a list to hold all columns data
            columns_list = []
            
            # Process all columns first
            for col in st.session_state.analysis['columns']:
                column_name = col['name']
                column_title = col['title']
                confidence = col.get('confidence_score', 0)
                description = col['description']
                
                # Get data type from stored metadata if available
                if column_metadata and "dtypes" in column_metadata and column_name in column_metadata["dtypes"]:
                    dtype = column_metadata["dtypes"].get(column_name, "Unknown")
                else:
                    # Fallback to directly accessing DataFrame
                    try:
                        dtype = str(st.session_state.df[column_name].dtype) if column_name in st.session_state.df.columns else 'N/A'
                    except:
                        dtype = 'Unknown'
                
                # Get missing percentage from stored metadata if available
                if column_metadata and "missing_percentages" in column_metadata and column_name in column_metadata["missing_percentages"]:
                    missing_pct = column_metadata["missing_percentages"].get(column_name, 0)
                    missing_str = f"{missing_pct:.2f}%"
                else:
                    # Fallback to directly calculating from DataFrame
                    try:
                        missing_pct = st.session_state.df[column_name].isna().mean() * 100 if column_name in st.session_state.df.columns else 0
                        missing_str = f"{missing_pct:.2f}%"
                    except:
                        missing_str = "N/A"
                        missing_pct = 0
                
                # Store all column data in a dictionary
                columns_list.append({
                    'Column Name': column_name,
                    'Column Title': column_title,
                    'Data Type': dtype,
                    'Missing %': missing_str,
                    'Missing Pct': missing_pct,  # Numeric version for sorting
                    'Confidence Score': f"{get_confidence_indicator(confidence)} {confidence:.2%}",
                    'Confidence Value': confidence,  # Numeric version for filtering
                    'Column Description': description
                })
            
            # Convert to DataFrame and store in session state
            st.session_state.analysis_columns_df = pd.DataFrame(columns_list)
        
        # Debug information - show min/max confidence values in the dataframe
        # Comment this out in production
        st.write("Debug - Confidence value range:", 
                st.session_state.analysis_columns_df['Confidence Value'].min(), 
                "to", 
                st.session_state.analysis_columns_df['Confidence Value'].max())
        
        # Filter options for confidence levels using radio buttons
        confidence_filter = st.radio(
            "Filter by Confidence Level:",
            options=[
                "All Columns",
                "🟢 High Confidence (>90%)",
                "🟡 Medium Confidence (80-90%)",
                "🔴 Low Confidence (<80%)"
            ],
            horizontal=True,  # Display radio buttons horizontally
            key="confidence_filter_radio"  # Add a unique key
        )
        
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

    # Key Observations
    st.markdown("#### Key Observations")
    for observation in st.session_state.analysis["key_observations"]:
        st.markdown(f"- {observation}")

def get_confidence_indicator(confidence_score: float) -> str:
    """Return an emoji indicator based on confidence score"""
    if confidence_score >= 0.9:
        return "🟢"  # Green circle for high confidence
    elif confidence_score >= 0.8:
        return "🟡"  # Yellow circle for medium confidence
    else:
        return "🔴"  # Red circle for low confidence

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


elif st.session_state.app_mode == "Data Curation App":

    tables = ['dim_gbl_sap_customer', 'v3_titanic_df']
    tables_choice = st.sidebar.selectbox("Select a Table", tables)

    # Read recipe inputs
    df_input = dataiku.Dataset("Data_Curation_Table")
    df_preview = df_input.get_dataframe()

    summary_df = df_preview[df_preview['Table Name'] == tables_choice].reset_index(drop=True)
    stats = json.loads(summary_df['Summary Info'][0])

    st.session_state.summary_df = summary_df
    st.session_state.stats = stats

    five_obs_json = summary_df.loc[0, "Sample Data"]
    five_obs_list = json.loads(five_obs_json)
    five_obs_df = pd.DataFrame(five_obs_list)
    
    st.header("Data Preview")
    st.dataframe(five_obs_df, use_container_width=True)
    st.info(f"File contains {summary_df['Total Rows'][0]} rows and {summary_df['Total Columns'][0]} columns.")

    # Display summary if available (should always be available now)
    if hasattr(st.session_state, 'summary_df') and st.session_state.summary_df is not None:
        display_summary(summary_df)

    # Display instructions if no analysis has been run
    st.info("Click 'Populate Data with AI Analysis' to get AI-generated insights about this dataset.")
    st.markdown("""
    ## Analysis Process:
    1. The dataset summary is automatically generated when you select a dataset
    2. Run the AI analysis to get intelligent insights and column interpretations
    """)
            
    if st.sidebar.button('Populate Data with AI Analysis', type="primary"):

        if df_preview is not None:

            try:
                with st.spinner("Analyzing data with AI..."):

                    # Get credentials from environment variables
                    api_key = os.getenv("AZURE_OPENAI_API_KEY")
                    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")

                    # Show a progress message during processing
                    progress_placeholder = st.empty()
                    progress_placeholder.info("Processing data - this may take a moment for large datasets...")
                    
                    #summary_df = df_preview[df_preview['Table Name'] == tables_choice].reset_index(drop=True)
                    #stats = json.loads(summary_df['Summary Info'][0])

                    #st.session_state.summary_df = summary_df
                    #st.session_state.stats = stats


                    # For large datasets, prepare an overview prompt
                    # We'll store this for UI display, but use the batched approach for actual analysis
                    if summary_df['Total Columns'][0] > 30:  # For larger datasets
                        # Create the regular prompt for display purposes, using summary_df if available
                        prompt = prepare_gpt_prompt(stats, summary_df)
                        
                        # Also create the overview prompt and store it 
                        overview_prompt = prepare_overview_prompt(stats, summary_df)
                        
                        # We'll add this to session state in the UI
                        if 'overview_prompt' not in st.session_state:
                            st.session_state.overview_prompt = overview_prompt
                    else:
                        # For smaller datasets, use the standard approach with summary_df if available
                        prompt = prepare_gpt_prompt(stats, summary_df)

                    # Also pass the summary_df to get_gpt_analysis
                    analysis = get_gpt_analysis(stats, api_key, endpoint, summary_df)

                    # Store results in session state
                    st.session_state.analysis = analysis
                    st.session_state.prompt = prompt
                    st.session_state.overview_prompt = prompt
                    
                    # Replace progress message when done
                    progress_placeholder.empty()
                    st.success("AI analysis completed successfully!")

            except Exception as e:
                st.error(f"An error occurred during analysis: {str(e)}")
                st.write("Please check the console logs for more details.")


            # Main Content Area
            if st.session_state.summary_df is not None:

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
