import os
import re
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

# ============================================================
# 🧩 Safe fallback for create_pandas_dataframe_agent
# ============================================================
try:
    # Try official helper (works on langchain-community <=0.3.6)
    from langchain_community.agent_toolkits import create_pandas_dataframe_agent
except Exception:
    from langchain.agents import AgentType, initialize_agent
    from langchain_community.tools.python.tool import PythonAstREPLTool

    def create_pandas_dataframe_agent(
        llm, df, verbose=False, allow_dangerous_code=False, prefix=""
    ):
        """Minimal fallback implementation of Pandas DataFrame Agent."""
        import pandas as pd
        namespace = {"df": df, "pd": pd}
        tool = PythonAstREPLTool(locals=namespace)
        return initialize_agent(
            [tool],
            llm,
            agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            verbose=verbose,
            prefix=prefix,
        )

# ============================================================
# ⚙️ Environment setup
# ============================================================
load_dotenv()
pd.options.display.float_format = "{:,.2f}".format

st.set_page_config(page_title="🧠 AI Data Analyst Agent", layout="wide")
st.title("🧠 Ask Your Data — AI Data Analyst Agent")
st.caption("Built using LangChain + OpenAI + Streamlit")

# ============================================================
# 📂 File Upload
# ============================================================
uploaded_file = st.file_uploader("📂 Upload a CSV or Excel file", type=["csv", "xlsx"])

if uploaded_file is not None:
    # Read CSV / Excel
    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)

    # Fix date parsing warnings
    for col in df.columns:
        if df[col].dtype == "object" and "date" in col.lower():
            try:
                df[col] = pd.to_datetime(df[col], format="%Y-%m-%d", errors="coerce")
            except Exception:
                pass

    st.success(f"✅ File uploaded: {df.shape[0]} rows × {df.shape[1]} columns")
    st.dataframe(df.convert_dtypes())

    # ========================================================
    # 🤖 Initialize Model + Agent
    # ========================================================
    llm = ChatOpenAI(
        temperature=0,
        api_key=os.getenv("OPENAI_API_KEY"),
        model="gpt-4-turbo",
    )

    CUSTOM_INSTRUCTIONS = """
    You are an expert data analyst.
    When performing analysis or aggregation (like totals, averages, or groupings),
    always return results in clear tabular format (columns and rows).
    If applicable, summarize key insights in plain English.
    """

    agent = create_pandas_dataframe_agent(
        llm,
        df,
        verbose=True,
        allow_dangerous_code=True,
        prefix=CUSTOM_INSTRUCTIONS,
    )

    # ========================================================
    # 💬 Query
    # ========================================================
    query = st.text_area(
        "💬 Ask something about your data",
        placeholder="Example: Show total sales by category or Plot monthly profit trend",
    )

    if st.button("Analyze"):
        with st.spinner("🔍 AI analyzing your data..."):
            try:
                response = agent.invoke(query, return_intermediate_steps=True)

                # -----------------------------
                # 🧠 Display raw model output
                # -----------------------------
                if "output" in response:
                    st.write("### 🧠 AI Response")
                    st.write(response["output"])

                # -----------------------------
                # 📊 Extract numeric results
                # -----------------------------
                matches = re.findall(
                    r"([\d\.e\+\-]+)\s+for\s+([\w\s]+)", response.get("output", "")
                )
                data = []
                for val, cat in matches:
                    try:
                        num = float(val.replace(",", "").replace("e+", "E"))
                        data.append([cat.strip(), round(num, 2)])
                    except Exception:
                        pass

                if data:
                    df_result = pd.DataFrame(data, columns=["Category", "Value"])
                    st.write("### 📊 Structured Result")
                    st.dataframe(df_result)
                    st.bar_chart(df_result.set_index("Category"))

                # -----------------------------
                # 📈 Show intermediate DataFrames
                # -----------------------------
                for step in response.get("intermediate_steps", []):
                    observation = step[1]
                    if isinstance(observation, pd.DataFrame):
                        st.write("### 📋 Filtered Data Output")
                        st.dataframe(observation.head(20))

            except Exception as e:
                st.error(f"⚠️ Error: {e}")

else:
    st.info("👆 Upload a dataset to start your analysis.")

st.markdown("---")
st.caption("Built with ❤️ using LangChain, OpenAI, and Streamlit")
