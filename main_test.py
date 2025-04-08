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
    if st.session_state.analysis["columns"]:
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
        
        # Create DataFrame directly from AI response
        columns_data = []
        for col in st.session_state.analysis['columns']:
            column_name = col['name']
            confidence = col.get('confidence_score', 0)
            confidence_numeric = confidence  # Store the raw numeric value for filtering
            
            # Get data type from stored metadata
            if column_metadata and "dtypes" in column_metadata and column_name in column_metadata["dtypes"]:
                dtype = column_metadata["dtypes"].get(column_name, "Unknown")
            else:
                dtype = "Unknown"
            
            # Get missing percentage from stored metadata
            if column_metadata and "missing_percentages" in column_metadata and column_name in column_metadata["missing_percentages"]:
                missing_pct = column_metadata["missing_percentages"].get(column_name, 0)
                missing_pct_str = f"{missing_pct:.2f}%"
            else:
                missing_pct_str = "N/A"
            
            # Add to the list of dictionaries
            columns_data.append({
                'Column Name': column_name,
                'Column Title': col['title'],
                'Data Type': dtype,
                'Missing %': missing_pct_str,
                'Confidence Score': f"{get_confidence_indicator(confidence)} {confidence:.2%}",
                'Column Description': col['description'],
                '_confidence_numeric': confidence_numeric  # Hidden column for filtering
            })
        
        # Create DataFrame from the list of dictionaries
        column_df = pd.DataFrame(columns_data)
        
        # Add confidence filter using radio buttons
        confidence_filter = st.radio(
            "Filter by confidence level:",
            options=["All", "High confidence (>90%)", "Medium confidence (80-90%)", "Low confidence (<80%)"],
            horizontal=True
        )

        st.write(column_df)
        
        # Apply confidence filter based on selection
        filtered_df = column_df.copy()
        if confidence_filter != "All":
            if confidence_filter == "High confidence (>90%)":
                filtered_df = column_df[column_df['_confidence_numeric'] > 0.9]
            elif confidence_filter == "Medium confidence (80-90%)":
                filtered_df = column_df[(column_df['_confidence_numeric'] >= 0.8) & (column_df['_confidence_numeric'] <= 0.9)]
            elif confidence_filter == "Low confidence (<80%)":
                filtered_df = column_df[column_df['_confidence_numeric'] < 0.8]
        
        # Remove the hidden column used for filtering
        filtered_df = filtered_df.drop('_confidence_numeric', axis=1)
        
        # Display filter results info
        if confidence_filter != "All":
            st.info(f"Showing {len(filtered_df)} columns with {confidence_filter}")
        
        # Display the filtered dataframe
        st.dataframe(filtered_df, use_container_width=True)

        # Export Column Details (full dataset, without the hidden confidence column)
        #export_df = column_df.drop('_confidence_numeric', axis=1)
        #st.download_button(
        #    "Export Column Details (CSV)",
        #    export_df.to_csv(index=False),
        #    "column_details.csv",
        #    "text/csv"
        #)

    # Key Observations
    st.markdown("#### Key Observations")
    for observation in st.session_state.analysis["key_observations"]:
        st.markdown(f"- {observation}")

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

    # Let user select dataset
    dataset_option = st.sidebar.selectbox(
        "Choose a dataset:",
        options=["v3_titanic_df", "dim_gbl_sap_customer"]
    )

    # Read recipe inputs
    df_input = dataiku.Dataset("Data_Curation_Table")
    df_preview = df_input.get_dataframe()

    summary_df = df_preview[df_preview['Table Name'] == dataset_option].reset_index(drop=True)
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
    display_summary(summary_df)

    # Display instructions if no analysis has been run
    st.info("Click 'Populate Data with AI Analysis' to get AI-generated insights about this dataset.")
    st.markdown("""
    ## Analysis Process:
    1. The dataset summary is automatically generated when you select a dataset
    2. Run the AI analysis to get intelligent insights and column interpretations
    """)

    #tables = ['dim_gbl_sap_customer', 'v3_titanic_df']
    #tables_choice = st.sidebar.selectbox("Select a Table", tables)

    # Read recipe inputs
    #df_input = dataiku.Dataset("Data_Curation_Table")
    #df_preview = df_input.get_dataframe()

    #summary_df = df_preview[df_preview['Table Name'] == tables_choice].reset_index(drop=True)
    #stats = json.loads(summary_df['Summary Info'][0])

    #st.session_state.summary_df = summary_df
    #st.session_state.stats = stats

    #five_obs_json = summary_df.loc[0, "Sample Data"]
    #five_obs_list = json.loads(five_obs_json)
    #five_obs_df = pd.DataFrame(five_obs_list)
    
    #st.header("Data Preview")
    #st.dataframe(five_obs_df, use_container_width=True)
    #st.info(f"File contains {summary_df['Total Rows'][0]} rows and {summary_df['Total Columns'][0]} columns.")

    # Here we separate the form submission from the filtering logic
    # FORM - Just for dataset selection and analysis generation
     
        
    # Form submit button for running the analysis
    submit_button = st.sidebar.button("Populate Data with AI Analysis")
        
    if submit_button:

        st.session_state.last_submission_time = pd.Timestamp.now()
        st.sidebar.write(f"Last form submission: {st.session_state.last_submission_time}")

        # Show progress
        with st.spinner(f"Analyzing dataset {dataset_option}... This may take a minute for large datasets."):

            # Process the selected dataset
            start_time = time.time()

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

            # Mark that we have results
            st.session_state.has_analysis_results = True
            
            # Replace progress message when done
            progress_placeholder.empty()
            st.success("AI analysis completed successfully!")

        # This ensures filtering doesn't cause resubmission
        if st.session_state.get('has_analysis_results', False):

            st.write(st.session_state.has_analysis_results)
            # Display the summary information
            if hasattr(st.session_state, 'summary_df'):
                display_summary(st.session_state.summary_df)
            
            # Display the GPT analysis
            if hasattr(st.session_state, 'analysis'):
                display_analysis()
    
