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
            confidence_numeric = confidence  # Raw numeric value for filtering
            
            # Get data type from stored metadata if available
            dtype = column_metadata.get("dtypes", {}).get(column_name, "Unknown")
            
            # Get missing percentage from stored metadata if available
            if "missing_percentages" in column_metadata and column_name in column_metadata["missing_percentages"]:
                missing_pct = column_metadata["missing_percentages"][column_name]
                missing_pct_str = f"{missing_pct:.2f}%"
            else:
                missing_pct_str = "N/A"
            
            # Append column details to the list
            columns_data.append({
                'Column Name': column_name,
                'Column Title': col['title'],
                'Data Type': dtype,
                'Missing %': missing_pct_str,
                'Confidence Score': f"{get_confidence_indicator(confidence)} {confidence:.2%}",
                'Column Description': col['description'],
                '_confidence_numeric': confidence_numeric  # Hidden column for filtering
            })
        
        # Create a DataFrame from the list of dictionaries
        column_df = pd.DataFrame(columns_data)
        
        # Add a confidence filter using radio buttons in the main area
        confidence_filter = st.radio(
            "Filter by confidence level:",
            options=["All", "High confidence (>90%)", "Medium confidence (80-90%)", "Low confidence (<80%)"],
            horizontal=True
        )
        
        # Apply confidence filter based on the selected option
        filtered_df = column_df.copy()
        if confidence_filter != "All":
            if confidence_filter == "High confidence (>90%)":
                filtered_df = column_df[column_df['_confidence_numeric'] > 0.9]
            elif confidence_filter == "Medium confidence (80-90%)":
                filtered_df = column_df[(column_df['_confidence_numeric'] >= 0.8) & (column_df['_confidence_numeric'] <= 0.9)]
            elif confidence_filter == "Low confidence (<80%)":
                filtered_df = column_df[column_df['_confidence_numeric'] < 0.8]
        
        # Remove the hidden column before display
        filtered_df = filtered_df.drop('_confidence_numeric', axis=1)
        
        # Optionally display an info message if filtering has been applied
        if confidence_filter != "All":
            st.info(f"Showing {len(filtered_df)} columns with {confidence_filter}")
        
        # Display the filtered DataFrame
        st.dataframe(filtered_df, use_container_width=True)
        
        # Export the filtered DataFrame as CSV
        csv_data = filtered_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Export Column Details (CSV)",
            data=csv_data,
            file_name="column_details.csv",
            mime="text/csv"
        )

    # Key Observations
    st.markdown("#### Key Observations")
    for observation in st.session_state.analysis["key_observations"]:
        st.markdown(f"- {observation}")
