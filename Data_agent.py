import streamlit as st
import pandas as pd
import numpy as np
import google.generativeai as genai
import io
import re

# Page Configuration
st.set_page_config(page_title="AI Data Cleaning Agent", layout="wide")

# --- 1. Configuration & API Setup ---
api_key = st.secrets.get("GEMINI_API_KEY")

# --- 2. Sidebar: Data Source & Control ---
with st.sidebar:
    st.title("⚙️ Control Center")
    if api_key:
        # Configuring Gemini with REST transport for better compatibility
        genai.configure(api_key=api_key, transport='rest')
        st.success("✅ API Connected")
    else:
        st.error("❌ API Key Missing in Secrets")

    st.divider()
    uploaded_file = st.file_uploader("Upload CSV or Excel file", type=["csv", "xlsx"])
    
    if uploaded_file:
        # Load file into session state to maintain data across interactions
        if "df" not in st.session_state or st.sidebar.button("Reload Original File"):
            try:
                if uploaded_file.name.endswith('.csv'):
                    st.session_state.df = pd.read_csv(uploaded_file)
                else:
                    st.session_state.df = pd.read_excel(uploaded_file)
                st.toast(f"Loaded: {uploaded_file.name}")
            except Exception as e:
                st.error(f"Error loading file: {e}")

    if st.button("🗑️ Clear Cache & Reset"):
        st.session_state.df = None
        st.session_state.messages = []
        st.rerun()

# --- 3. Main Interface ---
st.title("🤖 AI Data Cleaning Agent")
st.caption("Powered by Gemini 2.5 Flash - Intelligent Data Engineering for Research")

if "df" in st.session_state and st.session_state.df is not None:
    # Automatic Data Audit Report
    with st.expander("📊 Real-time Data Quality Audit", expanded=True):
        c1, c2, c3 = st.columns(3)
        c1.metric("Total Rows", len(st.session_state.df))
        c2.metric("Missing Values", st.session_state.df.isnull().sum().sum())
        c3.metric("Duplicate Rows", st.session_state.df.duplicated().sum())
        st.dataframe(st.session_state.df.head(5))

    # Chat Interaction Logic
    if user_input := st.chat_input("Enter cleaning instructions (e.g., 'Standardize dates' or 'Identify outliers')..."):
        st.chat_message("user").write(user_input)

        # Preparing detailed data context for the LLM
        data_info = {
            "columns": list(st.session_state.df.columns),
            "sample_dict": st.session_state.df.head(3).to_dict(),
            "dtypes": st.session_state.df.dtypes.astype(str).to_dict()
        }
        
        # Comprehensive Engineering Prompt
prompt = f"""
        You are a Senior Data Engineering Agent. 
        Dataset Context: {data_info}
        User Instruction: '''{user_input}'''

        --- MANDATORY POLICIES (Apply to ALL tasks) ---
        1. **DATA INTEGRITY**: Do NOT delete any rows (no `dropna()`) unless the user explicitly says "delete rows". 
        2. **STRICT REPAIR**: If data conversion fails (e.g., date formatting, number parsing), use `errors='coerce'` to create NaT/NaN, then fill those specific cells with a logical default or leave as null. DO NOT drop the entire row.
        3. **PRESERVATION**: Keep all columns unless asked to drop them.

        --- TASK-SPECIFIC GUIDANCE (Execute based on User Instruction) ---
        - IF handling DATES: Support mixed formats ('.', '/', '-') and unify to YYYY-MM-DD.
        - IF handling OUTLIERS: Use Z-score or IQR to identify/mark values without deleting rows.
        - IF handling TEXT: Clean HTML, remove special characters, or unify casing as requested.
        - IF handling MISSING VALUES: Use fillna() with mean, median, mode, or a specific value.

        --- EXECUTION RULES ---
        - IF the request is unrelated to data processing/analysis, output ONLY: [REJECT].
        - ALWAYS assign the result back to `df`.
        - OUTPUT ONLY PURE PYTHON CODE. NO MARKDOWN, NO EXPLANATIONS.
        """
        with st.chat_message("assistant"):
            try:
                # Using the latest 2026 model logic
                target_model = "models/gemini-2.5-flash"
                model = genai.GenerativeModel(target_model)
                response = model.generate_content(prompt)
                # Cleaning potential markdown tags from response
                clean_code = re.sub(r'```python|```', '', response.text).strip()

                if "[REJECT]" in clean_code:
                    st.warning("⚠️ This instruction is outside the scope of data engineering.")
                else:
                    # Execution Environment
                    exec_env = {
                        'pd': pd, 
                        'np': np, 
                        're': re, 
                        'io': io,
                        'df': st.session_state.df.copy()
                    }
                    
                    # Execute AI-generated script
                    exec(clean_code, {}, exec_env)
                    
                    # Update state with processed data
                    st.session_state.df = exec_env['df']
                    st.success("✅ Task Executed Successfully")
                    st.dataframe(st.session_state.df.head(10))
                    
                    with st.expander("View Execution Logic"):
                        st.code(clean_code)
            except Exception as e:
                st.error(f"Execution Error: {e}")

    # --- 4. Export Section ---
    st.divider()
    export_col1, export_col2 = st.columns([3, 1])
    with export_col1:
        export_format = st.selectbox("Select Export Format:", ["CSV", "Excel", "JSON"])

    # Prepare download data
    if export_format == "CSV":
        export_data = st.session_state.df.to_csv(index=False).encode('utf-8')
        file_ext = "csv"
        mime_type = "text/csv"
    elif export_format == "Excel":
        buffer = io.BytesIO()
        with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
            st.session_state.df.to_excel(writer, index=False)
        export_data = buffer.getvalue()
        file_ext = "xlsx"
        mime_type = "application/vnd.ms-excel"
    else:
        export_data = st.session_state.df.to_json(orient='records').encode('utf-8')
        file_ext = "json"
        mime_type = "application/json"

    with export_col2:
        st.download_button(
            label=f"📥 Download .{file_ext}",
            data=export_data,
            file_name=f"cleaned_data.{file_ext}",
            mime=mime_type
        )
else:
    st.info("👋 Please upload a dataset in the sidebar to begin.")
