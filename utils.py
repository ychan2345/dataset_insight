import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple
import json
from langchain.chat_models import AzureChatOpenAI
from langchain.schema import HumanMessage, SystemMessage
from langchain.prompts import ChatPromptTemplate

def prepare_gpt_prompt(stats: Dict[str, Any], summary_df=None) -> str:
    """Prepare the prompt for GPT-4 analysis.
    
    Args:
        stats: Dictionary of column statistics with metadata
        summary_df: Optional summary DataFrame created by create_summary_df function
    """
    prompt = """Analyze the following dataset and provide detailed insights strictly in valid JSON format with no extra text.

Dataset Information:
---------------------
Table Name: {table_name}
Total Rows: {rows}
Total Columns: {columns}

Summary Information:
---------------------
{column_info}

Instructions:
1. Provide a brief description of the dataset including any notable patterns or anomalies.
2. Suggest up to 3 potential analyses or next steps that could be performed on this dataset.
3. For each column in the dataset, provide a concise title, a brief description of its contents or significance, and a confidence score (between 0 and 1) indicating your confidence in the interpretation.
4. List up to 3 key observations from the data.

Return your response strictly as a valid JSON object using the following format:
{{
  "dataset_description": "A brief overview of the dataset.",
  "suggested_analysis": ["First analysis suggestion", "Second analysis suggestion", "Third analysis suggestion"],
  "columns": [
      {{
          "name": "ColumnName",
          "title": "A concise title",
          "description": "A brief description of the column",
          "confidence_score": 0.95
      }}
  ],
  "key_observations": ["Observation one", "Observation two", "Observation three"]
}}

Make sure your response contains only the JSON object with the keys specified above."""

    column_details = []
    
    # Initialize variables to prevent unbound variable errors
    table_name = "dataset"
    rows = 0
    columns = 0
    
    # If summary_df is provided, extract the table name and other metadata from it
    if summary_df is not None:
        try:
            table_name = summary_df["Table Name"].iloc[0]
            rows = summary_df["Total Rows"].iloc[0]
            columns = summary_df["Total Columns"].iloc[0]
        except (KeyError, IndexError, AttributeError) as e:
            # Keep the defaults if there's an error
            print(f"Error extracting basic metadata from summary_df: {str(e)}")
            pass
        
        # Try to load the column stats from the summary DataFrame
        try:
            summary_stats = json.loads(summary_df["Summary Info"].iloc[0])
            
            # Build column details from the summary stats
            for col, info in summary_stats.items():
                # Add missing percentage for all columns
                missing_pct = info.get('missing_pct', 0)
                missing_info = f", missing={missing_pct}%" if missing_pct > 0 else ""
                
                # Check for datetime type first
                if info.get('type') == 'datetime':
                    min_date = info['stats'].get('min_date', 'None')
                    max_date = info['stats'].get('max_date', 'None')
                    details = f"- {col} (Datetime): range={min_date} to {max_date}{missing_info}"
                elif 'stats' in info and 'mean' in info['stats']:
                    # Handle NA values in numerical statistics
                    mean_val = info['stats'].get('mean')
                    mean_str = str(mean_val) if mean_val is not None else "None"
                    
                    median_val = info['stats'].get('median')
                    median_str = str(median_val) if median_val is not None else "None"
                    
                    details = f"- {col} (Numerical): mean={mean_str}, median={median_str}{missing_info}"
                elif 'message' in info:
                    # For columns with too many unique values
                    details = f"- {col} (Categorical): {info['message']} (total unique: {info['total_unique']}){missing_info}"
                else:
                    # Process categorical columns
                    try:
                        distribution = list(info['distribution'].keys())[:3]  # Show only top 3 values in prompt
                        details = f"- {col} (Categorical): top values={distribution}"
                        if info.get('truncated', False):
                            details += f" (total unique: {info.get('total_unique', 0)})"
                        details += missing_info
                    except (KeyError, TypeError):
                        # Fallback for any column that doesn't fit standard patterns
                        details = f"- {col} (Other): type={info.get('type', 'unknown')}{missing_info}"
                column_details.append(details)
                
        except (json.JSONDecodeError, KeyError, IndexError) as e:
            # If there's an error parsing the summary DataFrame, fall back to using stats directly
            print(f"Error parsing summary DataFrame: {str(e)}. Falling back to direct stats.")
            # Continue with normal stats processing below
            column_details = []  # Reset column_details to ensure we don't have duplicates
    
    # If summary_df was not provided or failed to parse, use the stats dict directly
    if summary_df is None or not column_details:
        for col, info in stats.items():
            # Add missing percentage for all columns
            missing_pct = info.get('missing_pct', 0)
            missing_info = f", missing={missing_pct}%" if missing_pct > 0 else ""
            
            # Check for datetime type first
            if info.get('type') == 'datetime':
                min_date = info['stats'].get('min_date', 'None')
                max_date = info['stats'].get('max_date', 'None')
                details = f"- {col} (Datetime): range={min_date} to {max_date}{missing_info}"
            elif 'stats' in info and 'mean' in info['stats']:
                # Handle NA values in numerical statistics
                mean_val = info['stats'].get('mean')
                mean_str = str(mean_val) if mean_val is not None else "None"
                
                median_val = info['stats'].get('median')
                median_str = str(median_val) if median_val is not None else "None"
                
                details = f"- {col} (Numerical): mean={mean_str}, median={median_str}{missing_info}"
            elif 'message' in info:
                # For columns with too many unique values
                details = f"- {col} (Categorical): {info['message']} (total unique: {info['total_unique']}){missing_info}"
            else:
                # Process categorical columns
                try:
                    distribution = list(info['distribution'].keys())[:3]  # Show only top 3 values in prompt
                    details = f"- {col} (Categorical): top values={distribution}"
                    if info.get('truncated', False):
                        details += f" (total unique: {info.get('total_unique', 0)})"
                    details += missing_info
                except (KeyError, TypeError):
                    # Fallback for any column that doesn't fit standard patterns
                    details = f"- {col} (Other): type={info.get('type', 'unknown')}{missing_info}"
            column_details.append(details)

        # Generate a table name from the data if we don't have a summary_df
        table_name = "dataset"
        
        # Try to get the metadata from stats
        if '_metadata' in stats:
            rows = stats['_metadata'].get('total_rows', 0)
            columns = stats['_metadata'].get('total_columns', 0)
        # Otherwise use default values (should have been extracted from summary_df if available)
    
    # Format the prompt with the collected values
    formatted_prompt = prompt.format(
        table_name=table_name,
        rows=rows,
        columns=columns,
        column_info="\n".join(column_details)
    )

    # Debug log for prompt size
    print(f"Prompt size (characters): {len(formatted_prompt)}")
    return formatted_prompt

def prepare_overview_prompt(stats: Dict[str, Any], summary_df=None) -> str:
    """Prepare a prompt for general dataset overview without detailed column analysis.
    Uses the metadata from stats to avoid requiring direct DataFrame access.
    
    Args:
        stats: Dictionary with column statistics including _metadata
        summary_df: Optional summary DataFrame created by create_summary_df function
    """
    # Note: This function no longer requires direct DataFrame access
    prompt = """Analyze the following dataset and provide general insights strictly in valid JSON format with no extra text.
Focus only on overall characteristics - no detailed column analysis.

Dataset Information:
---------------------
Table Name: {table_name}
Total Rows: {rows}
Total Columns: {columns}

Column pattern: {column_pattern}

Instructions:
1. Provide a brief description of the dataset including any notable patterns or anomalies.
2. Suggest up to 3 potential analyses or next steps that could be performed on this dataset.
3. List up to 3 key observations from the data.

Return your response strictly as a valid JSON object using the following format:
{{
  "dataset_description": "A brief overview of the dataset (1-2 sentences).",
  "suggested_analysis": ["First analysis suggestion", "Second analysis suggestion", "Third analysis suggestion"],
  "key_observations": ["Observation one", "Observation two", "Observation three"]
}}

Make sure your response contains only the JSON object with the keys specified above."""

    # Initialize variables
    table_name = "dataset"
    rows = 0
    columns = 0
    column_pattern = ""
    column_names = []
    
    # First try to get column names from stats dictionary
    if stats:
        # Get all column names excluding _metadata
        column_names = [col for col in stats.keys() if col != '_metadata']
        
        # Get metadata from stats
        if '_metadata' in stats:
            # Get row and column counts from stats metadata
            rows = stats['_metadata'].get('total_rows', 0)
            columns = stats['_metadata'].get('total_columns', 0)
    
    # If column_names is still empty, try to get from summary_df if available
    if not column_names and summary_df is not None:
        # Try to extract column names from summary_df
        if 'structure' in summary_df.columns and not summary_df['structure'].empty:
            structure = summary_df.iloc[0]['structure']
            if isinstance(structure, dict) and 'columns' in structure:
                column_names = structure['columns']
    
    # If still no column names, use an empty list
    if not column_names:
        column_names = []
    
    # Generate column pattern if we have column names
    if column_names:
        if not columns:  # If columns count wasn't set from metadata
            columns = len(column_names)
            
        # Check if columns follow a pattern (like col_1, col_2, etc.)
        if column_names[0].startswith('col_'):
            column_pattern = f"col_1 through col_{columns}"
        else:
            # Show just a few column names as example
            sample_columns = column_names[:5]
            column_pattern = ', '.join(sample_columns)
            if len(column_names) > 5:
                column_pattern += f", ... and {len(column_names) - 5} more columns"
        
        # Try to extract a table name from the first column if it makes sense
        if any(keyword in column_names[0].lower() for keyword in ["id", "key", "name"]):
            prefix = column_names[0].split("_")[0] if "_" in column_names[0] else column_names[0]
            table_name = f"{prefix}_data"
            
    # Check if we can get table name from summary_df
    if summary_df is not None and 'table_name' in summary_df.columns and not summary_df['table_name'].empty:
        table_name = summary_df.iloc[0]['table_name']
    
    # Use the collected information to format the prompt
    formatted_prompt = prompt.format(
        table_name=table_name,
        rows=rows,
        columns=columns,
        column_pattern=column_pattern
    )

    print(f"Overview prompt size (characters): {len(formatted_prompt)}")
    return formatted_prompt

def prepare_columns_batch_prompt(stats: Dict[str, Any], columns_batch: list, summary_df=None) -> str:
    """Prepare a prompt for analyzing a batch of columns.
    Uses metadata from stats to eliminate dependency on the DataFrame.
    
    Args:
        stats: Dictionary with column statistics (including _metadata with dataset size info)
        columns_batch: List of column names to analyze in this batch
        summary_df: Optional summary DataFrame created by create_summary_df function
    """
    # Note: This function no longer requires direct DataFrame access
    
    # Initialize variables
    table_name = "dataset"
    rows = 0
    columns = 0
    total_columns = 0
    
    # Get metadata from stats
    if '_metadata' in stats:
        rows = stats['_metadata'].get('total_rows', 0)
        total_columns = stats['_metadata'].get('total_columns', 0)
    
    # Use total_columns as columns count
    columns = total_columns
    
    # Try to determine table name from available sources
    if summary_df is not None and 'table_name' in summary_df.columns and not summary_df['table_name'].empty:
        table_name = summary_df.iloc[0]['table_name']
    elif len(stats) > 1:  # Ensure there's at least one column besides _metadata
        # Get first column name (excluding _metadata)
        all_column_names = [col for col in stats.keys() if col != '_metadata']
        if all_column_names and all_column_names[0].startswith('col_'):
            table_name = "dataset"
        elif all_column_names and any(keyword in all_column_names[0].lower() for keyword in ["id", "key", "name"]):
            prefix = all_column_names[0].split("_")[0] if "_" in all_column_names[0] else all_column_names[0]
            table_name = f"{prefix}_data"
    
    # For extremely large datasets (200+ columns), use an ultra concise prompt
    if columns >= 200 or total_columns >= 200:
        prompt = """Analyze the following columns from the dataset and provide detailed insights strictly in valid JSON format.

Dataset Information:
---------------------
Table Name: {table_name}
Total Rows: {rows}
Total Columns: {total_columns} (analyzing {batch_size} now)

Columns to analyze in this batch: {column_names}

Instructions:
For each column, provide:
1. A concise title (2-3 words)
2. A very brief description (5-10 words)
3. A confidence score (between 0 and 1)

Return your response strictly as a valid JSON object using the following format:
{{
    "columns": [
        {{
            "name": "column_name",
            "title": "Brief title",
            "description": "Very concise description",
            "confidence_score": 0.8
        }}
    ]
}}

Make sure your response contains only the JSON object with the columns key."""

        # Just list column names without stats to save space
        column_names = ", ".join(columns_batch)
        
        # Generate a table name from available data sources
        # First try to get from summary_df if available
        if summary_df is not None and 'table_name' in summary_df.columns and not summary_df['table_name'].empty:
            table_name = summary_df.iloc[0]['table_name']
        # If no summary_df, try to infer from column names in stats
        elif len(stats) > 1:  # Ensure there's at least one column besides _metadata
            # Get first column name (excluding _metadata)
            all_column_names = [col for col in stats.keys() if col != '_metadata']
            if all_column_names and all_column_names[0].startswith('col_'):
                table_name = "dataset"
            elif all_column_names and any(keyword in all_column_names[0].lower() for keyword in ["id", "key", "name"]):
                prefix = all_column_names[0].split("_")[0] if "_" in all_column_names[0] else all_column_names[0]
                table_name = f"{prefix}_data"
        
        formatted_prompt = prompt.format(
            table_name=table_name,
            rows=rows,
            total_columns=total_columns,
            batch_size=len(columns_batch),
            column_names=column_names
        )
    else:
        # Standard prompt for smaller datasets
        prompt = """Analyze the following columns from the dataset and provide detailed insights strictly in valid JSON format.

Dataset Information:
---------------------
Table Name: {table_name}
Total Rows: {rows}
Total Columns: {total_columns} (analyzing {batch_size} now)

Columns to analyze in this batch:
{column_info}

Instructions:
For each column, provide:
1. A concise title (2-3 words)
2. A brief description of its contents or significance (under 15 words)
3. A confidence score (between 0 and 1) indicating your confidence in the interpretation

Return your response strictly as a valid JSON object using the following format:
{{
    "columns": [
        {{
            "name": "column_name",
            "title": "Brief title",
            "description": "Brief description",
            "confidence_score": 0.8
        }}
    ]
}}

Make sure your response contains only the JSON object with the columns key."""

        # Create simplified column details to reduce prompt size
        column_details = []
        for col in columns_batch:
            info = stats.get(col, {})
            if info.get('type') == 'datetime':
                # Handle datetime columns
                min_date = info['stats'].get('min_date', 'None')
                max_date = info['stats'].get('max_date', 'None')
                details = f"- {col}: datetime range={min_date} to {max_date}, missing={info.get('missing_pct', 0)}%"
            elif 'stats' in info and 'mean' in info['stats']:
                # More concise numeric stats with null handling
                mean_val = info['stats'].get('mean')
                mean_str = str(mean_val) if mean_val is not None else "None"
                details = f"- {col}: mean={mean_str}, missing={info.get('missing_pct', 0)}%"
            else:
                # Simplified categorical data
                details = f"- {col}: categorical, missing={info.get('missing_pct', 0)}%"
            column_details.append(details)

        # Generate a table name from available data sources
        # First try to get from summary_df if available
        if summary_df is not None and 'table_name' in summary_df.columns and not summary_df['table_name'].empty:
            table_name = summary_df.iloc[0]['table_name']
        # If no summary_df, try to infer from column names in stats
        elif len(stats) > 1:  # Ensure there's at least one column besides _metadata
            # Get first column name (excluding _metadata)
            all_column_names = [col for col in stats.keys() if col != '_metadata']
            if all_column_names and all_column_names[0].startswith('col_'):
                table_name = "dataset"
            elif all_column_names and any(keyword in all_column_names[0].lower() for keyword in ["id", "key", "name"]):
                prefix = all_column_names[0].split("_")[0] if "_" in all_column_names[0] else all_column_names[0]
                table_name = f"{prefix}_data"
        
        formatted_prompt = prompt.format(
            table_name=table_name,
            rows=rows,
            total_columns=total_columns,
            batch_size=len(columns_batch),
            column_info="\n".join(column_details)
        )

    print(f"Batch prompt size (characters): {len(formatted_prompt)}")
    return formatted_prompt

def get_gpt_analysis(stats: Dict[str, Any], api_key: str, 
                   endpoint: str = "", summary_df=None) -> Dict[str, Any]:
    """Get GPT-4 analysis of the dataset using OpenAI via Langchain.
    For large datasets, breaks column analysis into batches.
    
    Args:
        stats: Dictionary with column statistics (including _metadata with dataset size info)
        api_key: OpenAI API key
        endpoint: Optional API endpoint (for Azure OpenAI)
        summary_df: Optional summary DataFrame created by create_summary_df function
    """
    # Note: This function no longer requires direct DataFrame access
    if not api_key:
        raise ValueError("OpenAI API key is missing")

    try:
        # Initialize OpenAI chat model (using gpt-4o, the newest model)
        # the newest OpenAI model is "gpt-4o" which was released May 13, 2024.
        # do not change this unless explicitly requested by the user
        chat = AzureChatOpenAI(
                openai_api_key=api_key,
                azure_endpoint=endpoint,
                #deployment_name=deployment_name,
                openai_api_version="2024-10-21",
                temperature=0.5,
                max_tokens=1500,
                response_format={"type": "json_object"} # Ensure we get JSON responses
            )

        # Step 1: Get general dataset overview (no detailed column analysis)
        overview_prompt = prepare_overview_prompt(stats, summary_df)
        overview_messages = [
            SystemMessage(content="You are a data analysis expert. Provide concise dataset overview in JSON format."),
            HumanMessage(content=overview_prompt)
        ]
        
        overview_response = chat.invoke(overview_messages)
        
        try:
            overview_analysis = json.loads(overview_response.content)
            # Extract general information
            dataset_description = overview_analysis.get("dataset_description", "No description available")
            suggested_analysis = overview_analysis.get("suggested_analysis", [])[:3]
            key_observations = overview_analysis.get("key_observations", [])[:3]
        except json.JSONDecodeError:
            # Handle error in overview analysis
            dataset_description = "Error: Could not generate dataset description"
            suggested_analysis = ["Error: Could not generate analysis suggestions"]
            key_observations = ["Error: Could not generate observations"]
        
        # Step 2: Process columns in batches with dynamic sizing
        # Get columns from stats dictionary (excluding _metadata)
        all_columns = [col for col in stats.keys() if col != '_metadata']
        all_column_details = []
        
        # Get total columns count from metadata if available
        total_columns = stats.get('_metadata', {}).get('total_columns', len(all_columns))
        
        # Use dynamic batch sizing - smaller batches for larger datasets
        # Start with batches of 15, gradually reduce batch sizes for larger datasets
        # to avoid token limit issues as accumulated context grows
        if total_columns >= 200:  # Extremely large datasets (200+ columns)
            # First third of columns use batch size 15
            first_third = all_columns[:len(all_columns)//3]
            # Second third of columns use batch size 10
            second_third = all_columns[len(all_columns)//3:2*len(all_columns)//3]
            # Last third of columns use batch size 8
            last_third = all_columns[2*len(all_columns)//3:]
            
            # Create batches with different sizes
            first_batches = [first_third[i:i + 15] for i in range(0, len(first_third), 15)]
            second_batches = [second_third[i:i + 10] for i in range(0, len(second_third), 10)]
            last_batches = [last_third[i:i + 8] for i in range(0, len(last_third), 8)]
            
            # Combine all batches
            column_batches = first_batches + second_batches + last_batches
        elif total_columns > 120:  # Very large datasets (like 180 columns)
            # First half of columns use batch size 15
            first_half = all_columns[:len(all_columns)//2]
            # Second half of columns use batch size 10
            second_half = all_columns[len(all_columns)//2:]
            
            # Create batches with different sizes
            first_batches = [first_half[i:i + 15] for i in range(0, len(first_half), 15)]
            second_batches = [second_half[i:i + 10] for i in range(0, len(second_half), 10)]
            
            # Combine all batches
            column_batches = first_batches + second_batches
        else:
            # For smaller datasets, use consistent batch size
            batch_size = 15
            column_batches = [all_columns[i:i + batch_size] for i in range(0, len(all_columns), batch_size)]
        
        # Process each batch
        for batch_idx, columns_batch in enumerate(column_batches):
            print(f"Processing column batch {batch_idx+1}/{len(column_batches)}")
            
            # Create batch-specific prompt
            batch_prompt = prepare_columns_batch_prompt(stats, columns_batch, summary_df)
            
            # Create messages for the batch
            batch_messages = [
                SystemMessage(content="You are a data analysis expert. Provide concise column analysis in JSON format."),
                HumanMessage(content=batch_prompt)
            ]
            
            # Get response for this batch
            batch_response = chat.invoke(batch_messages)
            
            try:
                # Print response content for debugging
                print(f"Batch {batch_idx+1} response content preview (first 100 chars): {batch_response.content[:100]}")
                
                # Try to parse the JSON response
                batch_analysis = json.loads(batch_response.content)
                batch_columns = batch_analysis.get("columns", [])
                
                # Check if we got columns back
                if not batch_columns:
                    print(f"Warning: No columns returned in batch {batch_idx+1}")
                    print(f"Full response: {batch_response.content}")
                    
                    # Create a fallback analysis for the columns in this batch
                    fallback_columns = []
                    for col in columns_batch:
                        fallback_columns.append({
                            "name": col,
                            "title": f"Data column {col}",
                            "description": "Numeric data column (analysis failed)",
                            "confidence_score": 0.5
                        })
                    all_column_details.extend(fallback_columns)
                else:
                    # Add valid columns to our results
                    all_column_details.extend(batch_columns)
            except json.JSONDecodeError as e:
                print(f"Error parsing batch {batch_idx+1} response: {str(e)}")
                print(f"Response content that failed to parse: {batch_response.content}")
                
                # Create a fallback analysis for the columns in this batch
                fallback_columns = []
                for col in columns_batch:
                    fallback_columns.append({
                        "name": col,
                        "title": f"Data column {col}",
                        "description": "Numeric data column (analysis failed)",
                        "confidence_score": 0.5
                    })
                all_column_details.extend(fallback_columns)
        
        # Combine all results
        return {
            "dataset_description": dataset_description,
            "suggested_analysis": suggested_analysis,
            "columns": all_column_details,
            "key_observations": key_observations
        }

    except Exception as e:
        print(f"Error in GPT analysis: {str(e)}")
        # Include more details for debugging
        error_message = f"Error: {str(e)}"
        if "Unauthorized" in str(e) or "authentication" in str(e).lower():
            error_message = "Error: OpenAI API key is invalid or has expired. Please provide a valid API key."
        elif "model" in str(e).lower() and "does not exist" in str(e).lower():
            error_message = "Error: The requested model (gpt-4o) is not available with the provided API key. Please provide an API key with access to the latest models."
        elif "quota" in str(e).lower() or "rate limit" in str(e).lower():
            error_message = "Error: OpenAI API rate limit exceeded. Please try again later or use an API key with higher quotas."
        
        return {
            "dataset_description": error_message,
            "suggested_analysis": ["Error: Could not generate analysis suggestions"],
            "columns": [],
            "key_observations": ["Error: Could not generate observations"]
        }

def process_file_path(file_path: str, api_key: str, endpoint: str) -> Tuple[pd.DataFrame, Dict[str, Any], Dict[str, Any], str]:
    """Process a CSV file from a file path and return dataframe, stats, analysis, and the prompt."""
    try:
        # Load and validate the dataframe
        df = pd.read_csv(file_path)
        if df.empty:
            raise ValueError("The CSV file is empty")
        if len(df.columns) == 0:
            raise ValueError("No columns to parse from file")

        # Generate statistics for the data
        stats = get_column_stats(df)
        
        # Use the summary_df if it's available in session state
        import streamlit as st
        summary_df = st.session_state.get('summary_df')
        
        # For large datasets, prepare an overview prompt
        # We'll store this for UI display, but use the batched approach for actual analysis
        if len(df.columns) > 30:  # For larger datasets
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
        
        # Get the analysis (this will use batching internally if needed)
        # Also pass the summary_df to get_gpt_analysis
        analysis = get_gpt_analysis(stats, api_key, endpoint, summary_df)
        
        # Return everything needed by the UI
        return df, stats, analysis, prompt 
    except Exception as e:
        raise Exception(f"Error processing file: {str(e)}")
