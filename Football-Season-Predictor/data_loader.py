import pandas as pd
import streamlit as st
import glob
import os

@st.cache_data
def load_local_data():
    """
    Loads all league match data from the local 'data/' directory structure.
    Expected CSV format: Date,Home Team,Away Team,Home Goals,Away Goals
    Keeps fixtures with empty goal columns as NaN for simulation.
    """
    base_path = "data"
    all_files = glob.glob(os.path.join(base_path, "*", "*.csv"))

    if not all_files:
        st.error(f"No CSV files found under '{base_path}/'.")
        return pd.DataFrame()

    all_matches = []

    for file_path in all_files:
        season = os.path.basename(os.path.dirname(file_path))
        league_code = os.path.splitext(os.path.basename(file_path))[0].upper()

        try:
            df = pd.read_csv(file_path)
            df["Season"] = season
            df["League"] = league_code
            df.rename(columns={
                "Home Team": "HomeTeam",
                "Away Team": "AwayTeam",
                "Home Goals": "HomeGoals",
                "Away Goals": "AwayGoals",
            }, inplace=True)

            # Ensure columns exist
            if not {"HomeTeam", "AwayTeam", "HomeGoals", "AwayGoals"}.issubset(df.columns):
                st.warning(f"File {file_path} missing required columns. Skipped.")
                continue

            # Convert numeric columns but allow NaN
            df["HomeGoals"] = pd.to_numeric(df["HomeGoals"], errors="coerce")
            df["AwayGoals"] = pd.to_numeric(df["AwayGoals"], errors="coerce")

            # Parse date if present
            if "Date" in df.columns:
                df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

            all_matches.append(df)

        except Exception as e:
            st.warning(f"Failed to read {file_path}: {e}")

    if not all_matches:
        st.error("No valid match files were loaded.")
        return pd.DataFrame()

    master_df = pd.concat(all_matches, ignore_index=True)
    master_df = master_df[
        ["Season", "League", "Date", "HomeTeam", "AwayTeam", "HomeGoals", "AwayGoals"]
    ].sort_values(by=["Season", "League", "Date"], ignore_index=True)

    return master_df
