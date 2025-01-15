import streamlit as st
import pandas as pd
import numpy as np
import sqlite3
import joblib
from datetime import datetime

# Model ve scaler'ı yükle
model = joblib.load("best_model.pkl")
scaler = joblib.load("scaler.pkl")

# Recent form, H2H ve venue stats hesaplayıcılar (kod aynı)
def calculate_recent_form(team, match_date, conn, n=5):
    query = f"""
        SELECT home_team, away_team, home_goals, away_goals
        FROM Matches
        WHERE (home_team = '{team}' OR away_team = '{team}')
        AND date < '{match_date}'
        ORDER BY date DESC
        LIMIT {n}
    """
    matches = pd.read_sql_query(query, conn)
    matches['home_goals'] = pd.to_numeric(matches['home_goals'], errors='coerce')
    matches['away_goals'] = pd.to_numeric(matches['away_goals'], errors='coerce')
    if matches.empty:
        return 0, 0, 0
    goals_scored = matches.apply(lambda row: row['home_goals'] if row['home_team'] == team else row['away_goals'], axis=1).mean()
    goals_conceded = matches.apply(lambda row: row['away_goals'] if row['home_team'] == team else row['home_goals'], axis=1).mean()
    win_percentage = matches.apply(lambda row: 1 if (row['home_team'] == team and row['home_goals'] > row['away_goals']) or 
                                             (row['away_team'] == team and row['away_goals'] > row['home_goals']) else 0, axis=1).mean()
    return goals_scored, goals_conceded, win_percentage

def calculate_h2h(team1, team2, conn):
    query = f"""
        SELECT home_team, away_team, home_goals, away_goals
        FROM Matches
        WHERE (home_team = '{team1}' AND away_team = '{team2}')
           OR (home_team = '{team2}' AND away_team = '{team1}')
        ORDER BY date DESC
        LIMIT 5
    """
    matches = pd.read_sql_query(query, conn)
    matches['home_goals'] = pd.to_numeric(matches['home_goals'], errors='coerce')
    matches['away_goals'] = pd.to_numeric(matches['away_goals'], errors='coerce')
    if matches.empty or matches.isnull().values.any():
        return 0, 0
    goals_scored = matches.apply(lambda row: row['home_goals'] if row['home_team'] == team1 else row['away_goals'], axis=1).mean()
    goals_conceded = matches.apply(lambda row: row['away_goals'] if row['home_team'] == team1 else row['home_goals'], axis=1).mean()
    return goals_scored, goals_conceded

def calculate_venue_stats(team, venue, conn):
    query = f"""
        SELECT home_team, away_team, home_goals, away_goals
        FROM Matches
        WHERE {'home_team' if venue == 'home' else 'away_team'} = '{team}'
        ORDER BY date DESC
        LIMIT 5
    """
    matches = pd.read_sql_query(query, conn)
    matches['home_goals'] = pd.to_numeric(matches['home_goals'], errors='coerce')
    matches['away_goals'] = pd.to_numeric(matches['away_goals'], errors='coerce')
    if matches.empty:
        return 0, 0
    avg_goals = matches['home_goals' if venue == 'home' else 'away_goals'].mean()
    avg_conceded = matches['away_goals' if venue == 'home' else 'home_goals'].mean()
    return avg_goals, avg_conceded

def fetch_fixture_data(fixture):
    conn = sqlite3.connect("premier_league_detailed.db")
    home_team = fixture['home_team']
    away_team = fixture['away_team']
    match_date = fixture['datetime']
    home_recent_goals, home_recent_conceded, home_win_percentage = calculate_recent_form(home_team, match_date, conn)
    away_recent_goals, away_recent_conceded, away_win_percentage = calculate_recent_form(away_team, match_date, conn)
    h2h_goals_scored, h2h_goals_conceded = calculate_h2h(home_team, away_team, conn)
    home_avg_goals, home_avg_conceded = calculate_venue_stats(home_team, 'home', conn)
    away_avg_goals, away_avg_conceded = calculate_venue_stats(away_team, 'away', conn)
    home_metrics = pd.read_sql_query(f"SELECT * FROM TeamMetrics WHERE team = '{home_team}'", conn)
    away_metrics = pd.read_sql_query(f"SELECT * FROM TeamMetrics WHERE team = '{away_team}'", conn)
    conn.close()
    if home_metrics.empty or away_metrics.empty:
        raise ValueError(f"Missing metrics for teams: {home_team}, {away_team}")
    team_metrics_features = [
        home_metrics['avg_xG'].iloc[0], home_metrics['avg_goals'].iloc[0],
        home_metrics['defensive_xGA'].iloc[0], home_metrics['home_avg_xG'].iloc[0],
        home_metrics['last5_avg_xG'].iloc[0], away_metrics['avg_xG'].iloc[0],
        away_metrics['avg_goals'].iloc[0], away_metrics['defensive_xGA'].iloc[0],
        away_metrics['away_avg_xG'].iloc[0], away_metrics['last5_avg_xG'].iloc[0],
        home_metrics['h2h_avg_xG'].iloc[0], away_metrics['h2h_avg_xG'].iloc[0],
        home_metrics['h2h_avg_goals'].iloc[0], away_metrics['h2h_avg_goals'].iloc[0],
        home_metrics['avg_xG'].iloc[0] - home_metrics['avg_goals'].iloc[0],
        away_metrics['avg_xG'].iloc[0] - away_metrics['avg_goals'].iloc[0]
    ]
    input_data = [
        home_recent_goals, home_recent_conceded, home_win_percentage,
        away_recent_goals, away_recent_conceded, away_win_percentage,
        h2h_goals_scored, h2h_goals_conceded,
        home_avg_goals, home_avg_conceded,
        away_avg_goals, away_avg_conceded
    ] + team_metrics_features
    return np.array(input_data).reshape(1, -1)

def predict_fixture(fixture):
    input_data = fetch_fixture_data(fixture)
    scaled_input = scaler.transform(input_data)
    predicted_goals = model.predict(scaled_input)[0]
    thresholds = [0.5, 1.5, 2.5, 3.5, 4.5, 5.5]
    probabilities = {f"over_{t}": max(0, 1 - np.exp(-predicted_goals / t)) for t in thresholds}
    probabilities.update({f"under_{t}": 1 - probabilities[f"over_{t}"] for t in thresholds})
    return predicted_goals, probabilities

def predict_fixtures_in_date_range(start_date, end_date):
    conn = sqlite3.connect("premier_league_detailed.db")
    fixtures = pd.read_sql_query("SELECT * FROM Fixtures", conn)
    conn.close()
    
    # Ensure datetime column is properly typed
    fixtures['datetime'] = pd.to_datetime(fixtures['datetime'])
    
    # Convert start_date and end_date to match datetime64[ns]
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)
    
    # Filter by date range
    filtered_fixtures = fixtures[(fixtures['datetime'] >= start_date) & (fixtures['datetime'] <= end_date)]
    
    results = []
    for _, fixture in filtered_fixtures.iterrows():
        predicted_goals, probabilities = predict_fixture(fixture)
        results.append({
            "Match": f"{fixture['home_team']} vs {fixture['away_team']}",
            "Date": fixture['datetime'],
            "Predicted Goals": predicted_goals,
            "Probabilities": probabilities
        })
    return results

# Tahmin ve oran analizi için fonksiyonlar
def predict_fixture(fixture):
    input_data = fetch_fixture_data(fixture)
    scaled_input = scaler.transform(input_data)
    predicted_goals = model.predict(scaled_input)[0]
    
    thresholds = [0.5, 1.5, 2.5, 3.5, 4.5, 5.5]
    probabilities = {f"over_{t}": max(0, 1 - np.exp(-predicted_goals / t)) for t in thresholds}
    probabilities.update({f"under_{t}": 1 - probabilities[f"over_{t}"] for t in thresholds})
    
    return predicted_goals, probabilities

def best_bet(thresholds, probabilities, user_odds):
    best_choice = None
    best_value = -1
    for threshold, prob in probabilities.items():
        if threshold in user_odds:
            expected_value = prob * user_odds[threshold]
            if expected_value > best_value:
                best_value = expected_value
                best_choice = threshold
    return best_choice, best_value

# Streamlit App
st.title("Ust Olur Mu?")
st.write("Enter a date range to predict match outcomes.")

start_date = st.date_input("Start Date", value=datetime(2025, 1, 14))
end_date = st.date_input("End Date", value=datetime(2025, 1, 18))

if st.button("Predict Matches"):
    predictions = predict_fixtures_in_date_range(start_date, end_date)
    for pred in predictions:
        st.markdown(f"### {pred['Match']} - {pred['Date']}")
        st.write(f"Predicted Goals: {pred['Predicted Goals']:.2f}")

        # Alt/üst oranları ve kullanıcı oranlarını yan yana tablo olarak göster
        thresholds = [threshold.replace("over_", "Over ").replace("under_", "Under ") for threshold in pred["Probabilities"].keys()]
        probabilities = [f"{v*100:05.2f}%" for v in pred["Probabilities"].values()]
        
        # Kullanıcıdan oranları input olarak alın
        user_odds = []
        cols = st.columns(len(thresholds))
        for i, threshold in enumerate(thresholds):
            with cols[i]:
                st.write(threshold)
                st.write(probabilities[i])
                user_odds.append(st.number_input(f"Odds ({threshold})", value=1.0, min_value=1.0, step=0.1, key=f"odds_{pred['Match']}_{threshold}"))

        # Odds'u dictionary formatına dönüştür
        user_odds_dict = dict(zip(thresholds, user_odds))
        
        # Best bet calculation
        best_choice, best_value = best_bet(thresholds, pred["Probabilities"], user_odds_dict)
        if best_choice:
            st.success(f"Best Bet: {best_choice} with Expected Value: {best_value:.2f}")
        else:
            st.warning("No bet selected based on the given odds.")