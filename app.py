import os
import re
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_community.agent_toolkits.pandas.base import create_pandas_dataframe_agent




# ============================================================
# ✅ Environment setup
# ============================================================

load_dotenv()  # Load API key from .env
pd.options.display.float_format = '{:,.2f}'.format  # Disable scientific notation

# Streamlit configuration
st.set_page_config(page_title="🧠 AI Data Analyst Agent", layout="wide")
st.title("🧠 Ask Your Data — AI Data Analyst Agent")
st.caption("Built using official LangChain + OpenAI APIs.")

# ============================================================
# 📂 File upload section
# ============================================================

uploaded_file = st.file_uploader("📂 Upload a CSV or Excel file", type=["csv", "xlsx"])

if uploaded_file is not None:
    # Read file
    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)

    # 🔧 Fix: Convert object-type date columns to datetime
    for col in df.columns:
        if df[col].dtype == 'object' and 'date' in col.lower():
            try:
                df[col] = pd.to_datetime(df[col], format="%Y-%m-%d", errors="coerce")
            except Exception:
                pass

    # Display uploaded data
    st.success(f"✅ File uploaded successfully! ({df.shape[0]} rows, {df.shape[1]} columns)")
    st.dataframe(df.convert_dtypes())

    # ============================================================
    # 🤖 Initialize LangChain Agent
    # ============================================================

    llm = ChatOpenAI(
    temperature=0,
    api_key=os.getenv("OPENAI_API_KEY"),
    model="gpt-4-turbo"
    )

    CUSTOM_INSTRUCTIONS = """
    You are an expert data analyst.
    When performing analysis or aggregation (like totals, averages, or groupings),
    always return results in tabular format (columns and rows) for better readability.
    If applicable, also summarize key insights in plain English.
    """

    agent = create_pandas_dataframe_agent(
        llm,
        df,
        verbose=True,
        allow_dangerous_code=True,
        prefix=CUSTOM_INSTRUCTIONS
    )

    # ============================================================
    # 💬 Query Input
    # ============================================================

    query = st.text_area(
        "💬 Ask something about your data",
        placeholder="Example: 'Show total sales by category' or 'Plot profit trend over years'"
    )

    if st.button("Analyze"):
        with st.spinner("🔍 AI is analyzing your data..."):
            try:
                # ✅ Use the modern .invoke() method
                response = agent.invoke(query, return_intermediate_steps=True)

                # Display raw text output
                if "output" in response:
                    st.write("### 🧠 AI Result")
                    st.write(response["output"])

                # ====================================================
                # 📊 Detect and display tabular numeric results
                # ====================================================
                matches = re.findall(r'([\d\.e\+\-]+)\s+for\s+([\w\s]+)', response["output"])
                data = []
                for val, cat in matches:
                    try:
                        num = float(val.replace(',', '').replace('e+', 'E'))
                        data.append([cat.strip(), round(num, 2)])
                    except Exception:
                        pass

                if data:
                    df_result = pd.DataFrame(data, columns=["Category", "Value"])
                    st.write("### 📊 Structured Result")
                    st.dataframe(df_result)

                    # Optional: Display bar chart
                    st.bar_chart(df_result.set_index("Category"))

                # ====================================================
                # 📈 Try to detect DataFrame output from intermediate steps
                # ====================================================
                for step in response.get("intermediate_steps", []):
                    observation = step[1]
                    if isinstance(observation, pd.DataFrame):
                        st.write("### 📋 Filtered DataFrame Output")
                        st.dataframe(observation.head(20))

            except Exception as e:
                st.error(f"⚠️ Error: {e}")

else:
    st.info("👆 Please upload a dataset to start analysis.")

st.markdown("---")
st.caption("Built with ❤️ using LangChain + OpenAI + Streamlit")
