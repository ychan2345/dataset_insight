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
