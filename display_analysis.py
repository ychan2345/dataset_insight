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
        column_data = {
            'Column Name': [],
            'Column Title': [],
            'Data Type': [],
            'Missing %': [],
            'Confidence Score': [],
            'Column Description': []
        }

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

        for col in st.session_state.analysis['columns']:
            column_name = col['name']
            column_data['Column Name'].append(column_name)
            column_data['Column Title'].append(col['title'])
            
            # Get data type from stored metadata
            if column_metadata and "dtypes" in column_metadata and column_name in column_metadata["dtypes"]:
                dtype = column_metadata["dtypes"].get(column_name, "Unknown")
            else:
                # No fallbacks - we only use the stored metadata now
                dtype = "Unknown"
            column_data['Data Type'].append(dtype)
            
            # Get missing percentage from stored metadata
            if column_metadata and "missing_percentages" in column_metadata and column_name in column_metadata["missing_percentages"]:
                missing_pct = column_metadata["missing_percentages"].get(column_name, 0)
                column_data['Missing %'].append(f"{missing_pct:.2f}%")
            else:
                # No fallbacks - we only use the stored metadata now
                column_data['Missing %'].append("N/A")
                
            confidence = col.get('confidence_score', 0)
            column_data['Confidence Score'].append(f"{get_confidence_indicator(confidence)} {confidence:.2%}")
            column_data['Column Description'].append(col['description'])

        # Display the dataframe
        column_df = pd.DataFrame(column_data)
        st.dataframe(column_df, use_container_width=True)

        # Export Column Details
        st.download_button(
            "Export Column Details (CSV)",
            column_df.to_csv(index=False),
            "column_details.csv",
            "text/csv"
        )

    # Key Observations
    st.markdown("#### Key Observations")
    for observation in st.session_state.analysis["key_observations"]:
        st.markdown(f"- {observation}")
