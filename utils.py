import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple
import json
from openai import OpenAI
import os

def get_column_stats(df: pd.DataFrame) -> Dict[str, Any]:
    """Generate statistics for each column in the dataframe.

    For object columns:
      - Clean values by stripping quotes and whitespace.
      - Attempt to convert all non-missing values to datetime. 
        If every non-missing value converts, treat as date.
      - Otherwise, attempt to convert all non-missing values to numeric.
        If every non-missing value converts, compute numeric summary statistics.
      - Otherwise, treat as categorical.
        If the number of distinct values exceeds 50, skip computing the full distribution.
    """
    stats = {}

    for column in df.columns:
        col_series = df[column]
        column_type = str(col_series.dtype)
        missing_count = int(col_series.isnull().sum())
        is_date = False

        # If already a datetime type, flag as date.
        if pd.api.types.is_datetime64_any_dtype(col_series):
            is_date = True

        # Process object columns
        elif col_series.dtype == object:
            # Clean the column: remove leading/trailing quotes and whitespace.
            cleaned_series = col_series.apply(lambda x: x.strip(' "\'') if isinstance(x, str) else x)
            # Work only on non-missing values.
            non_missing = cleaned_series.dropna()

            # Attempt to convert all non-missing values to datetime.
            dt_converted = pd.to_datetime(non_missing, errors='coerce', infer_datetime_format=True)
            if len(non_missing) > 0 and dt_converted.isna().sum() == 0:
                is_date = True
                # Optionally update missing_count from cleaned_series
                missing_count = int(cleaned_series.isnull().sum())
            else:
                # Attempt to convert all non-missing values to numeric.
                num_converted = pd.to_numeric(non_missing, errors='coerce')
                if len(non_missing) > 0 and num_converted.isna().sum() == 0:
                    stats[column] = {
                        'type': column_type,
                        'stats': {
                            'mean': float(num_converted.mean()),
                            'median': float(num_converted.median()),
                            'std': float(num_converted.std())
                        },
                        'missing_count': int(cleaned_series.isnull().sum())
                    }
                    continue  # Skip to next column since stats are computed.
                else:
                    # Ambiguous values: treat as categorical.
                    value_counts = cleaned_series.value_counts(dropna=True)
                    if len(value_counts) > 50:
                        stats[column] = {
                            'type': column_type,
                            'note': 'Too many distinct values; distribution skipped.',
                            'missing_count': int(cleaned_series.isnull().sum())
                        }
                    else:
                        stats[column] = {
                            'type': column_type,
                            'distribution': {str(k): int(v) for k, v in value_counts.to_dict().items()},
                            'missing_count': int(cleaned_series.isnull().sum())
                        }
                    continue

        # If flagged as date (either originally or via conversion), record note.
        if is_date:
            stats[column] = {
                'type': column_type,
                'note': 'Date column - no summary statistics computed',
                'missing_count': missing_count
            }
        # For numeric columns already in numeric dtype.
        elif np.issubdtype(col_series.dtype, np.number):
            stats[column] = {
                'type': column_type,
                'stats': {
                    'mean': float(col_series.mean(skipna=True)),
                    'median': float(col_series.median(skipna=True)),
                    'std': float(col_series.std(skipna=True))
                },
                'missing_count': int(col_series.isnull().sum())
            }
        # For any other cases, treat as categorical.
        else:
            value_counts = col_series.value_counts(dropna=True)
            if len(value_counts) > 20:
                stats[column] = {
                    'type': column_type,
                    'note': 'Too many distinct values; distribution skipped.',
                    'missing_count': int(col_series.isnull().sum())
                }
            else:
                stats[column] = {
                    'type': column_type,
                    'distribution': {str(k): int(v) for k, v in value_counts.to_dict().items()},
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
            details = f"- {col} (Date or Other):\n  Note: {info.get('note', '')}\n  Missing Values: {info['missing_count']}"
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
