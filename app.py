import os
import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI   # ⭐ Gemini LLM
from langgraph.prebuilt import create_react_agent
from langchain.tools import tool

# ============================================================
# 🌍 Environment Setup
# ============================================================
load_dotenv()
st.set_page_config(page_title="🧠 AI Data Analyst (Gemini)", layout="wide")
st.title("🧠 AI Data Analyst (Gemini)")
st.caption("Built with LangChain + LangGraph + Gemini API")

# ============================================================
# 📂 File Upload
# ============================================================
uploaded_file = st.file_uploader("📂 Upload your CSV or Excel file", type=["csv", "xlsx"])

if uploaded_file is not None:
    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)

    st.success(f"✅ File uploaded successfully! ({df.shape[0]} rows, {df.shape[1]} columns)")
    st.dataframe(df.head(10))

    # ============================================================
    # ⚙️ Tools for Data Analysis
    # ============================================================
    @tool
    def preview_data(n: int = 5) -> str:
        """Show first n rows of the dataset."""
        return df.head(n).to_markdown(index=False)

    @tool
    def summarize_columns() -> str:
        """Summarize dataset columns and data types."""
        return df.dtypes.to_frame("dtype").to_markdown()

    @tool
    def describe_numeric() -> str:
        """Return basic statistics for numeric columns."""
        return df.describe().to_markdown()

    @tool
    def total_by_column(group_col: str, value_col: str) -> str:
        """Group by group_col and sum value_col."""
        if group_col not in df.columns or value_col not in df.columns:
            return f"❌ Column not found: {group_col} or {value_col}"
        grouped = df.groupby(group_col)[value_col].sum().reset_index()
        return grouped.to_markdown(index=False)

    tools = [preview_data, summarize_columns, describe_numeric, total_by_column]

    # ============================================================
    # 🤖 Gemini LLM Initialization
    # ============================================================
    llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0,
    google_api_key=os.getenv("GOOGLE_API_KEY")
)


    agent = create_react_agent(llm, tools)

    # ============================================================
    # 💬 Analysis Query
    # ============================================================
    query = st.text_area(
        "💬 Ask a question about your data",
        placeholder="Example: Show total Sales by Category"
    )

    if st.button("Analyze"):
        with st.spinner("🔍 AI analyzing your data..."):
            try:
                result = agent.invoke({"messages": [{"role": "user", "content": query}]})
                st.write("### 🧠 AI Response")
                st.markdown(result["messages"][-1].content)
            except Exception as e:
                st.error(f"⚠️ Error: {e}")

else:
    st.info("👆 Upload a dataset to start analysis.")
