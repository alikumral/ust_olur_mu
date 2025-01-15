import asyncio
import aiohttp
import pandas as pd
from understat import Understat
from sqlalchemy import create_engine
import json

# Veritabanı bağlantısı
engine = create_engine('sqlite:///premier_league_detailed.db')

# TeamHistory verilerini işleyip kaydetme
async def process_team_history(team_id, season, history_data):
    history_list = json.loads(history_data)  # JSON string'i listeye dönüştür
    history_records = []
    for record in history_list:
        history_records.append({
            "team_id": team_id,
            "season": season,
            "date": record["date"],
            "h_a": record["h_a"],  # Ev sahibi (h) ya da deplasman (a)
            "xG": record["xG"],
            "xGA": record["xGA"],
            "npxG": record["npxG"],
            "npxGA": record["npxGA"],
            "ppda_att": record["ppda"]["att"],
            "ppda_def": record["ppda"]["def"],
            "deep": record["deep"],
            "deep_allowed": record["deep_allowed"],
            "scored": record["scored"],
            "missed": record["missed"],
            "result": record["result"],
            "wins": record["wins"],
            "draws": record["draws"],
            "loses": record["loses"],
            "pts": record["pts"],
            "npxGD": record["npxGD"]
        })
    return pd.DataFrame(history_records)

# Teams verilerini çekme
async def fetch_teams_data(season):
    async with aiohttp.ClientSession() as session:
        understat = Understat(session)
        teams = await understat.get_teams("epl", season)
        team_data = []
        for team in teams:
            # TeamHistory tablosunu doldur
            history_df = await process_team_history(team["id"], season, json.dumps(team["history"]))
            history_df.to_sql('TeamHistory', con=engine, if_exists='append', index=False)
            
            # Teams tablosunu doldur
            team_data.append({
                "team_id": team["id"],
                "name": team["title"],
                "season": season
            })
        return pd.DataFrame(team_data)

# Matches verilerini çekme
async def fetch_matches_data(season):
    async with aiohttp.ClientSession() as session:
        understat = Understat(session)
        matches = await understat.get_league_results("epl", season)
        match_data = []
        for match in matches:
            match_data.append({
                "match_id": match["id"],
                "date": match["datetime"],
                "home_team_id": match["h"]["id"],
                "home_team": match["h"]["title"],
                "away_team_id": match["a"]["id"],
                "away_team": match["a"]["title"],
                "home_goals": match["goals"]["h"],
                "away_goals": match["goals"]["a"],
                "home_xG": match["xG"]["h"],
                "away_xG": match["xG"]["a"],
                "forecast_win": match["forecast"]["w"],
                "forecast_draw": match["forecast"]["d"],
                "forecast_loss": match["forecast"]["l"],
                "season": season
            })
        return pd.DataFrame(match_data)

# Tüm sezonların verilerini çekme ve SQL'e aktarma
async def fetch_and_save_all_seasons(seasons):
    for season in seasons:
        print(f"Fetching data for season {season}...")
        
        # Teams verileri
        teams_df = await fetch_teams_data(season)
        teams_df.to_sql('Teams', con=engine, if_exists='append', index=False)
        
        # Matches verileri
        matches_df = await fetch_matches_data(season)
        matches_df.to_sql('Matches', con=engine, if_exists='append', index=False)
        
        print(f"Season {season} data saved!")

# Hedef sezonlar
seasons = [2021, 2022, 2023, 2024, 2025]

# Verileri çek ve kaydet
asyncio.run(fetch_and_save_all_seasons(seasons))