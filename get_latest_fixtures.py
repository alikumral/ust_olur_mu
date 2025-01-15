import aiohttp
import asyncio
import sqlite3
import pandas as pd
from understat import Understat

# Fikstürleri çek ve geliştirilmiş yapıda veritabanına kaydet
async def fetch_and_save_fixtures():
    async with aiohttp.ClientSession() as session:
        understat = Understat(session)
        fixtures = await understat.get_league_fixtures("EPL", 2025)
        fixtures_data = []
        for fixture in fixtures:
            fixtures_data.append({
                "id": fixture["id"],
                "home_team_id": fixture["h"]["id"],
                "home_team": fixture["h"]["title"],
                "away_team_id": fixture["a"]["id"],
                "away_team": fixture["a"]["title"],
                "datetime": fixture["datetime"]
            })
        
        # Fikstürleri SQLite'a kaydet
        conn = sqlite3.connect("premier_league_detailed.db")
        pd.DataFrame(fixtures_data).to_sql("Fixtures", conn, if_exists="replace", index=False)
        conn.close()
        print("Geliştirilmiş fikstürler kaydedildi!")

# Async çağrısı
loop = asyncio.get_event_loop()
loop.run_until_complete(fetch_and_save_fixtures())