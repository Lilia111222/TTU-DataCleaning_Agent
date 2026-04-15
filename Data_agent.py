import streamlit as st
import pandas as pd
import google.generativeai as genai
import io
import re

st.set_page_config(page_title="Data Cleaning AI Agent", layout="wide")

# --- 1. Initialization ---
if "df" not in st.session_state:
    st.session_state.df = None
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- 2. Side panel: Settings and uploads ---
with st.sidebar:
    st.title("⚙️ Data Source Settings")

    # Core: Read Key from secrets
    api_key = st.secrets.get("GEMINI_API_KEY")

    if api_key:
        # Using REST transport protocol for compatibility
        genai.configure(api_key=api_key, transport='rest')
        has_api = True
        st.success(f"✅ API Connected (Key: {api_key[:4]}...)")
    else:
        st.error("❌ GEMINI_API_KEY not found in secrets.toml")
        has_api = False

    st.divider()
    uploaded_file = st.file_uploader("Upload CSV or Excel file", type=["csv", "xlsx"])
    
    if uploaded_file:
        if st.session_state.df is None or (
                hasattr(uploaded_file, 'name') and uploaded_file.name != st.session_state.get('last_file')):
            try:
                if uploaded_file.name.endswith('.csv'):
                    st.session_state.df = pd.read_csv(uploaded_file)
                else:
                    st.session_state.df = pd.read_excel(uploaded_file)
                st.session_state.last_file = uploaded_file.name
                st.toast(f"Successfully loaded: {uploaded_file.name}")
            except Exception as e:
                st.error(f"Read error: {e}")

    if st.button("🗑️ Clear All Cache"):
        st.session_state.df = None
        st.session_state.messages = []
        st.rerun()

# --- 3. Main Interface ---
st.title("🤖 Data Cleaning Agent (v2.5)")
st.caption("Powered by Gemini 2.5 Flash - Supporting deep logic and structured cleaning")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
        if "code" in msg:
            with st.expander("View AI-executed Python logic"):
                st.code(msg["code"])

# --- 4. Core Interaction Logic ---
if user_input := st.chat_input("Enter cleaning instructions (e.g., 'drop rows where email is null' or 'unify date format')..."):
    if not has_api:
        st.error("Please configure your API Key first.")
    else:
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.write(user_input)

        with st.chat_message("assistant"):
            with st.spinner("Gemini 2.5 is planning cleaning logic..."):
                try:
                    # Target model ID
                    target_model = "models/gemini-2.5-flash"
                    model = genai.GenerativeModel(target_model)

                    if st.session_state.df is not None:
                        buffer = io.StringIO()
                        st.session_state.df.info(buf=buffer)
                        data_info = f"Columns: {list(st.session_state.df.columns)}\nData Sample:\n{st.session_state.df.head(3).to_string()}"
                    else:
                        data_info = "No file loaded. Parse data from user input text if applicable."

                    prompt = f"""
                    You are a Python Data Scientist. 
                    Data Background: {data_info}
                    Instruction: '''{user_input}'''

                    Rules:
                    1. ONLY return pure Python code.
                    2. NO markdown formatting (No ```python).
                    3. Final result MUST be updated in the variable 'df'.
                    4. Use libraries: pd, io, re.
                    """

                    # Generate content using Gemini 2.5
                    response = model.generate_content(prompt)
                    clean_code = re.sub(r'```python|```', '', response.text).strip()

                    if "[REJECT]" in clean_code:
                        st.warning("Instruction is outside the scope of data processing.")
                    else:
                        # Prepare execution environment
                        env = {'pd': pd, 'io': io, 're': re}
                        if st.session_state.df is not None:
                            env['df'] = st.session_state.df.copy()

                        # Execute AI-generated script
                        exec(clean_code, env)

                        if 'df' in env:
                            st.session_state.df = env['df']
                            st.write(f"✅ Execution Successful! (Model: {target_model})")
                            st.dataframe(st.session_state.df.head(10))

                            st.session_state.messages.append({
                                "role": "assistant",
                                "content": "Cleaning task completed.",
                                "code": clean_code
                            })
                except Exception as e:
                    st.error(f"Runtime Error: {e}")

# --- 5. Export ---
if st.session_state.df is not None:
    st.divider()
    cols = st.columns([2, 1])
    with cols[0]:
        fmt = st.selectbox("Select export format:", ["CSV", "Excel", "JSON"])

    buf = io.BytesIO()
    if fmt == "CSV":
        data = st.session_state.df.to_csv(index=False).encode('utf-8')
        mime, ext = "text/csv", "csv"
    elif fmt == "Excel":
        with pd.ExcelWriter(buf, engine='xlsxwriter') as writer:
            st.session_state.df.to_excel(writer, index=False)
        data, mime, ext = buf.getvalue(), "application/vnd.ms-excel", "xlsx"
    else:
        data = st.session_state.df.to_json(orient='records').encode('utf-8')
        mime, ext = "application/json", "json"

    with cols[1]:
        st.write("")
        st.write("")
        st.download_button(f"📥 Download Cleaned File", data=data, file_name=f"cleaned_data.{ext}", mime=mime)
