import pandas as pd
import numpy as np
import sqlite3
import joblib
from datetime import datetime

# Model ve scaler'ı yükle
model = joblib.load("best_model.pkl")
scaler = joblib.load("scaler.pkl")

# Recent form, H2H ve venue stats hesaplayıcılar
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
    # Hatalı verileri temizle veya düzelt
    matches['home_goals'] = pd.to_numeric(matches['home_goals'], errors='coerce')
    matches['away_goals'] = pd.to_numeric(matches['away_goals'], errors='coerce')

    if matches.empty:
        return 0, 0, 0  # Default değerler
    
    goals_scored = matches.apply(lambda row: row['home_goals'] if row['home_team'] == team else row['away_goals'], axis=1).mean()
    goals_conceded = matches.apply(lambda row: row['away_goals'] if row['home_team'] == team else row['home_goals'], axis=1).mean()
    win_percentage = matches.apply(lambda row: 1 if (row['home_team'] == team and row['home_goals'] > row['away_goals']) or 
                                             (row['away_team'] == team and row['away_goals'] > row['home_goals']) else 0, axis=1).mean()
    return goals_scored, goals_conceded, win_percentage

def calculate_h2h(team1, team2, conn):
    print(f"Calculating H2H stats for {team1} vs {team2}")
    query = f"""
        SELECT home_team, away_team, home_goals, away_goals
        FROM Matches
        WHERE (home_team = '{team1}' AND away_team = '{team2}')
           OR (home_team = '{team2}' AND away_team = '{team1}')
        ORDER BY date DESC
        LIMIT 5
    """
    matches = pd.read_sql_query(query, conn)
    print(f"Raw H2H matches for {team1} vs {team2}:\n{matches}")

    # Convert to numeric
    matches['home_goals'] = pd.to_numeric(matches['home_goals'], errors='coerce')
    matches['away_goals'] = pd.to_numeric(matches['away_goals'], errors='coerce')
    print(f"Processed H2H matches for {team1} vs {team2}:\n{matches}")

    # Handle missing values
    if matches.empty or matches.isnull().values.any():
        print(f"No valid H2H data available for {team1} vs {team2}")
        return 0, 0

    goals_scored = matches.apply(lambda row: row['home_goals'] if row['home_team'] == team1 else row['away_goals'], axis=1).mean()
    goals_conceded = matches.apply(lambda row: row['away_goals'] if row['home_team'] == team1 else row['home_goals'], axis=1).mean()
    print(f"H2H stats for {team1} vs {team2}: Goals Scored: {goals_scored}, Goals Conceded: {goals_conceded}")
    return goals_scored, goals_conceded

def calculate_venue_stats(team, venue, conn):
    print(f"Calculating venue stats for team: {team}, venue: {venue}")
    query = f"""
        SELECT home_team, away_team, home_goals, away_goals
        FROM Matches
        WHERE {'home_team' if venue == 'home' else 'away_team'} = '{team}'
        ORDER BY date DESC
        LIMIT 5
    """
    matches = pd.read_sql_query(query, conn)
    print(f"Raw venue matches for {team} at {venue}:\n{matches}")

    # Convert columns to numeric
    matches['home_goals'] = pd.to_numeric(matches['home_goals'], errors='coerce')
    matches['away_goals'] = pd.to_numeric(matches['away_goals'], errors='coerce')

    print(f"Processed venue matches for {team} at {venue}:\n{matches}")

    if matches.empty:
        print(f"No matches found for {team} at {venue}")
        return 0, 0  # Default values

    avg_goals = matches['home_goals' if venue == 'home' else 'away_goals'].mean()
    avg_conceded = matches['away_goals' if venue == 'home' else 'home_goals'].mean()
    print(f"Venue stats for {team} at {venue}: Avg Goals: {avg_goals}, Avg Conceded: {avg_conceded}")
    return avg_goals, avg_conceded

# Fikstürlerden veri çek ve gerekli parametreleri hesapla
def fetch_fixture_data(fixture):
    conn = sqlite3.connect("premier_league_detailed.db")
    
    home_team = fixture['home_team']
    away_team = fixture['away_team']
    match_date = fixture['datetime']

    # Recent form hesapla
    home_recent_goals, home_recent_conceded, home_win_percentage = calculate_recent_form(home_team, match_date, conn)
    away_recent_goals, away_recent_conceded, away_win_percentage = calculate_recent_form(away_team, match_date, conn)

    # H2H stats hesapla
    h2h_goals_scored, h2h_goals_conceded = calculate_h2h(home_team, away_team, conn)

    # Venue stats hesapla
    home_avg_goals, home_avg_conceded = calculate_venue_stats(home_team, 'home', conn)
    away_avg_goals, away_avg_conceded = calculate_venue_stats(away_team, 'away', conn)

    # TeamMetrics verilerini çek
    home_metrics = pd.read_sql_query(f"SELECT * FROM TeamMetrics WHERE team = '{home_team}'", conn)
    away_metrics = pd.read_sql_query(f"SELECT * FROM TeamMetrics WHERE team = '{away_team}'", conn)
    
    conn.close()

    if home_metrics.empty or away_metrics.empty:
        raise ValueError(f"Missing metrics for teams: {home_team}, {away_team}")

    # TeamMetrics'ten özellikleri al
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

    # Model için input oluştur
    input_data = [
        home_recent_goals, home_recent_conceded, home_win_percentage,
        away_recent_goals, away_recent_conceded, away_win_percentage,
        h2h_goals_scored, h2h_goals_conceded,
        home_avg_goals, home_avg_conceded,
        away_avg_goals, away_avg_conceded
    ] + team_metrics_features  # Tüm özellikleri birleştir

    print(f"Final input data for fixture {home_team} vs {away_team}: {input_data}")
    return np.array(input_data).reshape(1, -1)

# Tahmin yap
def predict_fixture(fixture):
    input_data = fetch_fixture_data(fixture)
    scaled_input = scaler.transform(input_data)
    predicted_goals = model.predict(scaled_input)[0]
    
    thresholds = [0.5, 1.5, 2.5, 3.5, 4.5, 5.5]
    probabilities = {f"over_{t}": max(0, 1 - np.exp(-predicted_goals / t)) for t in thresholds}
    probabilities.update({f"under_{t}": 1 - probabilities[f"over_{t}"] for t in thresholds})
    
    return predicted_goals, probabilities

# Fikstürleri tahmin et
def predict_fixtures_in_date_range(start_date, end_date):
    conn = sqlite3.connect("premier_league_detailed.db")
    fixtures = pd.read_sql_query("SELECT * FROM Fixtures", conn)
    conn.close()
    
    fixtures['datetime'] = pd.to_datetime(fixtures['datetime'])
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

# Örnek kullanım
start_date = "2025-01-14"
end_date = "2025-01-18"

predictions = predict_fixtures_in_date_range(start_date, end_date)
for pred in predictions:
    print(f"Match: {pred['Match']}, Date: {pred['Date']}")
    print(f"Predicted Goals: {pred['Predicted Goals']:.2f}")
    print("Probabilities:")
    for key, value in pred['Probabilities'].items():
        print(f"  {key}: {value:.2%}")
    print()