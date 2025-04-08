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
        # Add confidence filter using radio buttons
        confidence_filter = st.radio(
            "Filter by confidence level:",
            options=["All", "High confidence (>90%)", "Medium confidence (80-90%)", "Low confidence (<80%)"],
            horizontal=True
        )
        
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

        # Create the dataframe
        column_df = pd.DataFrame(column_data)
        
        # Apply confidence filter based on selection
        filtered_df = column_df.copy()
        if confidence_filter != "All":
            # Extract numeric confidence values (removing the emoji and % sign)
            confidences = []
            for conf_str in column_df['Confidence Score']:
                # Extract numeric percentage value (format is "🟢 95.45%" or similar)
                confidences.append(float(conf_str.split()[1].strip('%')) / 100)
            
            # Filter based on confidence thresholds
            if confidence_filter == "High confidence (>90%)":
                mask = [c > 0.9 for c in confidences]
                filtered_df = column_df[mask]
            elif confidence_filter == "Medium confidence (80-90%)":
                mask = [(c >= 0.8) and (c <= 0.9) for c in confidences]
                filtered_df = column_df[mask]
            elif confidence_filter == "Low confidence (<80%)":
                mask = [c < 0.8 for c in confidences]
                filtered_df = column_df[mask]
        
        # Display filter results info
        if confidence_filter != "All":
            st.info(f"Showing {len(filtered_df)} columns with {confidence_filter}")
        
        # Display the filtered dataframe
        st.dataframe(filtered_df, use_container_width=True)

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
