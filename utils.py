import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple
import json
from openai import OpenAI
import os

def get_column_stats(df: pd.DataFrame) -> Dict[str, Any]:
    """Generate statistics for each column in the dataframe."""
    stats = {}

    for column in df.columns:
        col_series = df[column]
        column_type = str(col_series.dtype)
        is_date = False

        # Check if the column is already recognized as a datetime type
        if pd.api.types.is_datetime64_any_dtype(col_series):
            is_date = True
        # For object columns, attempt to convert to datetime
        elif col_series.dtype == object:
            converted = pd.to_datetime(col_series, errors='coerce', infer_datetime_format=True)
            # If more than 80% of the values are successfully converted, treat as date
            if converted.notna().mean() > 0.8:
                is_date = True

        if is_date:
            stats[column] = {
                'type': column_type,
                'note': 'Date column - no summary statistics computed',
                'missing_count': int(col_series.isnull().sum())
            }
        elif np.issubdtype(col_series.dtype, np.number):
            # Numerical column: compute only mean, median, and std (ignoring missing values)
            stats[column] = {
                'type': column_type,
                'stats': {
                    'mean': float(col_series.mean(skipna=True)),
                    'median': float(col_series.median(skipna=True)),
                    'std': float(col_series.std(skipna=True))
                },
                'missing_count': int(col_series.isnull().sum())
            }
        else:
            # Categorical column: compute distribution (value_counts ignores NaN by default)
            value_counts = col_series.value_counts(dropna=True).to_dict()
            stats[column] = {
                'type': column_type,
                'distribution': {
                    str(k): int(v)
                    for k, v in value_counts.items()
                },
                'missing_count': int(col_series.isnull().sum())
            }

    return stats


def prepare_gpt_prompt(df: pd.DataFrame, stats: Dict[str, Any]) -> str:
    """Prepare the prompt for GPT-4 analysis."""
    prompt = """Please analyze the following dataset and provide insights in a valid JSON format.

Dataset Overview:
- Total Rows: {rows}
- Total Columns: {columns}

Column Information:
{column_info}

Required JSON Response Format:
{{
    "dataset_description": "A comprehensive overview of the entire dataset",
    "suggested_analysis": [
        "Analysis suggestion 1",
        "Analysis suggestion 2",
        "Analysis suggestion 3"
    ],
    "columns": [
        {{
            "name": "column_name",
            "title": "Human readable title",
            "description": "Detailed description of the column and its significance",
            "confidence_score": 0.95
        }}
    ],
    "key_observations": [
        "Key insight 1",
        "Key insight 2",
        "Key insight 3"
    ]
}}

Note: For each column, provide a confidence score between 0 and 1 indicating how confident you are about the column title and description."""

    column_details = []
    for col, info in stats.items():
        if 'stats' in info:
            details = f"- {col} (Numerical):\n  Stats: {info['stats']}\n  Missing Values: {info['missing_count']}"
        elif 'distribution' in info:
            details = f"- {col} (Categorical):\n  Distribution: {info['distribution']}\n  Missing Values: {info['missing_count']}"
        else:
            details = f"- {col} (Date):\n  Note: {info.get('note', '')}\n  Missing Values: {info['missing_count']}"
        column_details.append(details)

    return prompt.format(rows=len(df),
                         columns=len(df.columns),
                         column_info="\n".join(column_details))


def get_gpt_analysis(df: pd.DataFrame, stats: Dict[str, Any], api_key: str) -> Dict[str, Any]:
    """Get GPT-4 analysis of the dataset."""
    prompt = prepare_gpt_prompt(df, stats)

    try:
        client = OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{
                "role": "system",
                "content": "You are a data analysis expert. Always respond with valid JSON."
            }, {
                "role": "user",
                "content": prompt
            }],
            response_format={"type": "json_object"}
        )

        # Parse the response and ensure all required fields exist
        analysis = json.loads(response.choices[0].message.content)
        return {
            "dataset_description": analysis.get("dataset_description", "No description available"),
            "suggested_analysis": analysis.get("suggested_analysis", []),
            "columns": analysis.get("columns", []),
            "key_observations": analysis.get("key_observations", [])
        }
    except json.JSONDecodeError as e:
        return {
            "dataset_description": "Error: Invalid JSON response from analysis",
            "suggested_analysis": ["Error: Could not generate analysis suggestions"],
            "columns": [],
            "key_observations": ["Error: Could not generate observations"]
        }
    except Exception as e:
        return {
            "dataset_description": f"Error: {str(e)}",
            "suggested_analysis": [],
            "columns": [],
            "key_observations": []
        }


def process_uploaded_file(uploaded_file, api_key: str) -> Tuple[pd.DataFrame, Dict[str, Any], Dict[str, Any]]:
    """Process the uploaded CSV file and return dataframe, stats, and analysis."""
    df = pd.read_csv(uploaded_file)
    stats = get_column_stats(df)
    analysis = get_gpt_analysis(df, stats, api_key)
    return df, stats, analysis
