import streamlit as st
import requests
import pandas as pd
import time

# Configuration
API_BASE_URL = "http://localhost:8000"

st.set_page_config(page_title="SignalForge Terminal", layout="wide")

st.title("⚡ SignalForge Research Terminal")
st.markdown("### Professional-grade Quantitative Research Platform")

# Sidebar: Controls
with st.sidebar:
    st.header("Research Parameters")
    symbol = st.text_input("Trading Symbol", value="BTCUSDT")
    timeframe = st.selectbox("Timeframe", ["1m", "5m", "15m", "1h", "1d"], index=3)

    if st.button("🚀 Launch Experiment", use_container_width=True):
        try:
            payload = {"symbol": symbol, "timeframe": timeframe}
            response = requests.post(f"{API_BASE_URL}/research/start", json=payload)
            if response.status_code == 200:
                res_data = response.json()
                st.session_state.current_exp_id = res_data["experiment_id"]
                st.success(f"Experiment Started: {res_data['experiment_id']}")
            else:
                st.error("Failed to start experiment")
        except Exception as e:
            st.error(f"Connection Error: {e}")

# Main Dashboard
if "current_exp_id" in st.session_state:
    exp_id = st.session_state.current_exp_id

    st.info(f"Monitoring Experiment: `{exp_id}`")

    # Poll for results
    with st.spinner("Waiting for Agents (Architect $\rightarrow$ Analyst $\rightarrow$ Modeler $\rightarrow$ Auditor)..."):
        found = False
        for _ in range(20): # Poll for 20 seconds
            try:
                resp = requests.get(f"{API_BASE_URL}/research/status/{exp_id}")
                if resp.status_code == 200:
                    data = resp.json()

                    # Results Layout
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Audit Verdict", data["verdict"])
                    with col2:
                        st.metric("Sharpe Ratio", f"{data['sharpe']:.2f}" if data['sharpe'] else "N/A")
                    with col3:
                        st.metric("Max Drawdown", f"{data['drawdown']:.2%}" if data['drawdown'] else "N/A")

                    st.subheader("Risk Auditor's Reasoning")
                    st.write(data["reasoning"])

                    st.subheader("Deterministic Configuration")
                    st.json(data["config"])

                    found = True
                    break
            except:
                pass
            time.sleep(1)

        if not found:
            st.warning("Experiment is still running or failed. Please check the logs.")
else:
    st.write("Welcome to SignalForge. Enter a symbol in the sidebar and click 'Launch Experiment' to begin the agentic research pipeline.")
