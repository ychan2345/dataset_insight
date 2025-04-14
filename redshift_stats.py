import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
import json
import os
import psycopg2
from psycopg2 import sql
from datetime import datetime
import warnings

# Custom JSON encoder for handling datetime/timestamp objects and other non-serializable types
class DateTimeEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, (datetime, pd.Timestamp)):
            return o.isoformat()
        elif isinstance(o, np.integer):
            return int(o)
        elif isinstance(o, np.floating):
            return float(o)
        elif isinstance(o, np.ndarray):
            return o.tolist()
        elif isinstance(o, (np.bool_)):
            return bool(o)
        elif pd.isna(o):
            return None
        try:
            return super().default(o)
        except TypeError:
            return str(o)  # Last resort: convert to string

def get_redshift_column_stats(
    conn_params: Dict[str, Any],
    table_name: str,
    schema_name: str = "public",
    sample_size: int = 10000
) -> Dict[str, Any]:
    """
    Generate statistics for each column in a Redshift table using native SQL aggregation.
    
    This function calculates statistics directly in Redshift using SQL aggregation functions,
    avoiding the need to extract the entire table. It also fetches a small sample for preview.
    
    Args:
        conn_params: Dictionary with psycopg2 connection parameters (host, port, dbname, user, password)
        table_name: Name of the table to analyze
        schema_name: Schema containing the table (default: public)
        sample_size: Number of rows to sample for preview display
        
    Returns:
        Dictionary containing column statistics and metadata in format compatible with utils.py
    """
    # Suppress the pandas UserWarning about DBAPI connections
    warnings.filterwarnings("ignore", message=".*other DBAPI2 objects are not tested.*")
    try:
        # Create psycopg2 connection
        conn = psycopg2.connect(**conn_params)
        cursor = conn.cursor()
        
        # Step 1: Get basic table information and column types
        # Get column information from information_schema
        column_query = f"""
            SELECT column_name, data_type
            FROM information_schema.columns
            WHERE table_schema = '{schema_name}' AND table_name = '{table_name}'
            ORDER BY ordinal_position
        """
        columns_df = pd.read_sql(column_query, conn)
        
        # Count total rows
        count_query = f'SELECT COUNT(*) AS row_count FROM "{schema_name}"."{table_name}"'
        row_count_df = pd.read_sql(count_query, conn)
        row_count = row_count_df.iloc[0]['row_count']
        
        # Get a sample for display purposes
        sample_query = f'SELECT * FROM "{schema_name}"."{table_name}" ORDER BY RANDOM() LIMIT {sample_size}'
        sample_df = pd.read_sql(sample_query, conn)
    
    except Exception as e:
        raise Exception(f"Error connecting to Redshift or querying metadata: {str(e)}")
    
    # Initialize stats dictionary with metadata
    stats = {
        "_metadata": {
            "total_rows": int(row_count),
            "total_columns": len(columns_df),
            "processed_rows": int(row_count),  # We're processing all rows via SQL
            "sample_rows": len(sample_df),
            "created_at": datetime.now().isoformat()
        }
    }
    
    # Process each column to get statistics
    for _, column_info in columns_df.iterrows():
        column_name = column_info['column_name']
        data_type = column_info['data_type']
        
        # Skip processing if column is in sample_df but contains only NULL values
        if column_name in sample_df.columns and sample_df[column_name].isna().all():
            stats[column_name] = {
                "type": "unknown",
                "missing_count": int(row_count),
                "missing_pct": 100.0
            }
            continue
            
        try:
            # Different query based on data type
            if data_type.startswith(('int', 'float', 'numeric', 'decimal', 'double', 'real')):
                # Numeric column statistics
                numeric_query = f"""
                    SELECT 
                        COUNT(*) as count,
                        COUNT(*) - COUNT("{column_name}") as missing_count,
                        AVG("{column_name}") as mean,
                        PERCENTILE_CONT(0.5) WITHIN GROUP(ORDER BY "{column_name}") as median,
                        MIN("{column_name}") as min,
                        MAX("{column_name}") as max,
                        STDDEV_POP("{column_name}") as std
                    FROM "{schema_name}"."{table_name}"
                """
                
                numeric_stats_df = pd.read_sql(numeric_query, conn)
                numeric_stats = numeric_stats_df.iloc[0].to_dict()
                
                # Calculate missing percentage
                missing_count = numeric_stats['missing_count'] 
                missing_pct = (missing_count / row_count) * 100 if row_count > 0 else 0
                
                # Format to match utils.py expected format
                stats[column_name] = {
                    "type": "numeric",
                    "missing_count": int(missing_count),
                    "missing_pct": float(missing_pct),
                    "stats": {
                        "count": int(numeric_stats['count']),
                        "mean": float(numeric_stats['mean']) if pd.notna(numeric_stats['mean']) else None,
                        "median": float(numeric_stats['median']) if pd.notna(numeric_stats['median']) else None,
                        "min": float(numeric_stats['min']) if pd.notna(numeric_stats['min']) else None,
                        "max": float(numeric_stats['max']) if pd.notna(numeric_stats['max']) else None,
                        "std": float(numeric_stats['std']) if pd.notna(numeric_stats['std']) else None
                    }
                }
            
            elif data_type.startswith(('date', 'time', 'timestamp')):
                # Date/time column statistics
                datetime_query = f"""
                    SELECT 
                        COUNT(*) as count,
                        COUNT(*) - COUNT("{column_name}") as missing_count,
                        MIN("{column_name}") as min_date,
                        MAX("{column_name}") as max_date
                    FROM "{schema_name}"."{table_name}"
                """
                
                datetime_stats_df = pd.read_sql(datetime_query, conn)
                datetime_stats = datetime_stats_df.iloc[0].to_dict()
                
                # Calculate missing percentage
                missing_count = datetime_stats['missing_count']
                missing_pct = (missing_count / row_count) * 100 if row_count > 0 else 0
                
                # Convert date objects to strings for JSON compatibility
                min_date = datetime_stats['min_date']
                max_date = datetime_stats['max_date']
                # Safely convert to string, handling various datetime types
                try:
                    min_date_str = min_date.isoformat() if pd.notna(min_date) else None
                except (AttributeError, TypeError):
                    min_date_str = str(min_date) if pd.notna(min_date) else None
                    
                try:
                    max_date_str = max_date.isoformat() if pd.notna(max_date) else None
                except (AttributeError, TypeError):
                    max_date_str = str(max_date) if pd.notna(max_date) else None
                
                # Format to match utils.py expected format
                stats[column_name] = {
                    "type": "datetime",
                    "missing_count": int(missing_count),
                    "missing_pct": float(missing_pct),
                    "stats": {
                        "count": int(datetime_stats['count']),
                        "min_date": min_date_str,
                        "max_date": max_date_str
                    }
                }
            
            else:
                # Categorical/text column statistics
                # Get unique count
                unique_query = f"""
                    SELECT 
                        COUNT(*) as count,
                        COUNT(*) - COUNT("{column_name}") as missing_count,
                        COUNT(DISTINCT "{column_name}") as unique_count
                    FROM "{schema_name}"."{table_name}"
                """
                
                cat_stats_df = pd.read_sql(unique_query, conn)
                cat_stats = cat_stats_df.iloc[0].to_dict()
                
                # Calculate missing percentage
                missing_count = cat_stats['missing_count']
                missing_pct = (missing_count / row_count) * 100 if row_count > 0 else 0
                unique_count = cat_stats['unique_count']
                
                # Get top values (limited to 5 to avoid performance issues)
                top_values_query = f"""
                    SELECT "{column_name}" as value, COUNT(*) as count
                    FROM "{schema_name}"."{table_name}"
                    WHERE "{column_name}" IS NOT NULL
                    GROUP BY "{column_name}"
                    ORDER BY COUNT(*) DESC
                    LIMIT 5
                """
                
                top_values_df = pd.read_sql(top_values_query, conn)
                
                # Create value counts dictionary
                distribution = {}
                for i, row in top_values_df.iterrows():
                    # Convert to string for consistent JSON serialization
                    value = str(row['value']) if pd.notna(row['value']) else "null"
                    distribution[value] = int(row['count'])
                
                # Format to match utils.py expected format
                stats[column_name] = {
                    "type": "categorical",
                    "missing_count": int(missing_count),
                    "missing_pct": float(missing_pct),
                    "total_unique": int(unique_count),
                    "distribution": distribution
                }
                
                # Check if there are too many unique values 
                if unique_count > 50:
                    stats[column_name]["truncated"] = True
                    # Include message for columns with too many unique values (helps with prompt formatting)
                    if unique_count > 1000:
                        stats[column_name]["message"] = f"High cardinality ({unique_count} unique values)"
                    else:
                        stats[column_name]["message"] = f"Multiple values ({unique_count} unique)"
        
        except Exception as e:
            # If we fail to get statistics for a column, log the error and include basic info
            print(f"Error processing column {column_name}: {str(e)}")
            stats[column_name] = {
                "type": "error",
                "error_message": str(e),
                "missing_count": None,
                "missing_pct": None
            }
    
    # Add sample data for preview - convert properly to handle timestamps and other special types
    # First convert to JSON with our custom encoder, then back to Python objects
    sample_json = json.dumps(sample_df.to_dict(orient="records"), cls=DateTimeEncoder)
    stats["_sample_data"] = json.loads(sample_json)
    
    # Close the connection
    conn.close()
    
    return stats

def create_redshift_summary_df(
    conn_params: Dict[str, Any],
    table_name: str,
    schema_name: str = "public",
    sample_size: int = 10000
) -> pd.DataFrame:
    """
    Create a summary DataFrame for a Redshift table with metadata and statistics.
    Format matches the expected format for prepare_gpt_prompt in utils.py.
    
    Args:
        conn_params: Dictionary with psycopg2 connection parameters (host, port, dbname, user, password)
        table_name: Name of the table to analyze
        schema_name: Schema containing the table (default: public)
        sample_size: Number of rows to sample for preview display
        
    Returns:
        DataFrame with table metadata, summary statistics, and sample data
    """
    # Suppress the pandas UserWarning about DBAPI connections
    warnings.filterwarnings("ignore", message=".*other DBAPI2 objects are not tested.*")
    try:
        # Get statistics using the Redshift-optimized function
        stats = get_redshift_column_stats(
            conn_params=conn_params,
            table_name=table_name,
            schema_name=schema_name,
            sample_size=sample_size
        )
        
        # Format structure in the way utils.py expects it
        structure = {
            "columns": [col for col in stats.keys() if col != "_metadata" and not col.startswith("_")],
            "shape": (stats["_metadata"]["total_rows"], stats["_metadata"]["total_columns"])
        }
        
        # Create a new dataframe for summary information (column names match what utils.py expects)
        summary_data = {
            "Table Name": [table_name],
            "Schema": [schema_name],
            "Total Rows": [stats["_metadata"]["total_rows"]],
            "Total Columns": [stats["_metadata"]["total_columns"]],
            "Size (MB)": [None],  # Could use SVV_TABLE_INFO but requires special permissions
            "Created At": [stats["_metadata"]["created_at"]],
            "Summary Info": [json.dumps(stats, cls=DateTimeEncoder)],
            "structure": [structure]  # Structure info in the format expected by prepare_gpt_prompt
        }
        
        # Create summary dataframe
        summary_df = pd.DataFrame(summary_data)
        
        # Extract a sample for display
        conn = psycopg2.connect(**conn_params)
        sample_query = f'SELECT * FROM "{schema_name}"."{table_name}" ORDER BY RANDOM() LIMIT {sample_size}'
        sample_df = pd.read_sql(sample_query, conn)
        conn.close()
            
        # Add sample to the summary dataframe
        summary_df["Sample Data"] = [sample_df]
        
        return summary_df
        
    except Exception as e:
        raise Exception(f"Error creating summary DataFrame: {str(e)}")

def process_redshift_table(
    host: str,
    port: str,
    dbname: str,
    user: str,
    password: str,
    table_name: str,
    schema_name: str = "public",
    api_key: str = None,
    endpoint: str = "",
    sample_size: int = 10000
) -> Tuple[pd.DataFrame, Dict[str, Any], Dict[str, Any], str]:
    """
    Process a Redshift table and return sample dataframe, stats, analysis, and prompt.
    This is an alternative to process_file_path/process_uploaded_file for direct Redshift integration.
    
    Args:
        host: Redshift host address
        port: Redshift port
        dbname: Redshift database name
        user: Redshift username
        password: Redshift password
        table_name: Name of the table to analyze
        schema_name: Schema containing the table
        api_key: OpenAI API key for analysis
        endpoint: API endpoint (for Azure OpenAI)
        sample_size: Number of rows to sample for preview
        
    Returns:
        Tuple of (sample_dataframe, stats, analysis, prompt)
    """
    try:
        # Create psycopg2 connection parameters
        conn_params = {
            'host': host,
            'port': port,
            'dbname': dbname,
            'user': user,
            'password': password
        }
        
        # Get statistics directly from Redshift
        stats = get_redshift_column_stats(
            conn_params=conn_params,
            table_name=table_name,
            schema_name=schema_name,
            sample_size=sample_size
        )
        
        # Create a sample dataframe from the sample data in stats
        sample_df = pd.DataFrame(stats["_sample_data"])
        
        # Create summary dataframe
        summary_df = create_redshift_summary_df(
            conn_params=conn_params,
            table_name=table_name,
            schema_name=schema_name,
            sample_size=sample_size
        )
        
        # Import functions from utils.py to prepare prompt and get analysis
        from utils import prepare_gpt_prompt, get_gpt_analysis
        
        # Prepare prompt for GPT analysis
        prompt = prepare_gpt_prompt(stats, summary_df)
        
        # Get GPT analysis if API key is provided
        analysis = {}
        if api_key:
            analysis = get_gpt_analysis(stats, api_key, endpoint, summary_df)
        
        return sample_df, stats, analysis, prompt
        
    except Exception as e:
        raise Exception(f"Error processing Redshift table: {str(e)}")
