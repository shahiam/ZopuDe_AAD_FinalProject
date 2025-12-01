import streamlit as st
import pandas as pd
from data_loader import load_local_data
from simulation import run_monte_carlo_simulation


# -----------------------------
# Streamlit Configuration
# -----------------------------
st.set_page_config(page_title="Football Season Simulator", layout="wide")
st.title("Football Season Simulator — 2024-25")

st.sidebar.header("Simulation Settings")
num_simulations = st.sidebar.number_input("Monte Carlo runs:", 100, 10000, 1000, step=100)

# -----------------------------
# Load Data
# -----------------------------
with st.spinner("Loading match data..."):
    df = load_local_data()

if df.empty:
    st.error("No data found. Please add CSV files under the data/ directory.")
    st.stop()

# League mapping
league_map = {
    "ENG": "Premier League",
    "ES": "La Liga",
    "FR": "Ligue 1",
    "IT": "Serie A",
    "DE": "Bundesliga",
}

# Identify available leagues & most recent season
latest_season = df["Season"].max()
available_leagues = sorted(df["League"].unique())

selected_league = st.sidebar.selectbox(
    "Select League:",
    available_leagues,
    format_func=lambda x: league_map.get(x.split(".")[0], x),
)

# Split into historical and target season
historical_df = df[(df["League"] == selected_league) & (df["Season"] < latest_season)]
fixtures_df = df[(df["League"] == selected_league) & (df["Season"] == latest_season)]

if fixtures_df.empty:
    st.warning("No fixtures available for the latest season in this league.")
    st.stop()

# -----------------------------
# Simulation Trigger
# -----------------------------
st.divider()
st.header(f"Predicting {league_map.get(selected_league, selected_league)} {latest_season} Season")

if st.button("Run Monte Carlo Simulation"):
    with st.spinner(f"Running {num_simulations:,} simulations..."):
        predicted_table = run_monte_carlo_simulation(historical_df, fixtures_df, num_simulations)

    st.toast("✅ Simulation complete!")
    st.dataframe(predicted_table, use_container_width=True, height=600)
    st.caption("Modify `simulation.py` to adjust the model or add features like promotion/relegation.")
else:
    st.info("Click **Run Monte Carlo Simulation** to start.")
