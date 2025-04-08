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
