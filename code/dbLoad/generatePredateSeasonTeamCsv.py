"""
This module makes CSV files with seasonteams data for each team on each date in a given season.
CSVs are dumped into /Users/Ryan/Documents/projects/ncaa-basketball/data/predate_games/
One CSV for each season, with a record for each team on each date of that season.
The CSVs can be used to replicate what data was available on the given date,
for creating/validating prediction models.

Required: 
- Game data in the MongoDB database

That's it - the rest is handled here.
"""

import pandas as pd
# Import to add project folder to sys path
import sys
utils_path = '/Users/Ryan/Documents/projects/ncaa-basketball/code/utils'
if utils_path not in sys.path:
    sys.path.append(utils_path)

from db import get_db
from api import getSeasonsList, preAggSeasonGames

if __name__ == "__main__":
    db = get_db()
    seasons_list = getSeasonsList(_db=db)
    for SEASON in seasons_list:  # TODO remove number limiter
        print(f"Working on {SEASON}")

        # Get entire season's data from DB and work locally
        season_games = pd.DataFrame(
            list(
                db.games.find(
                    {'Season': SEASON},
                    {'_id': 0}
                )
            )
        )

        # Get unique dates of games in data
        game_dates = pd.DataFrame(
            season_games['GameDate'].unique(),
            columns=['GameDate']
        ).sort_values(
            by=['GameDate']
        ).reset_index()

        first_date = True
        # Loop through all the dates in the season
        for DATE in game_dates['GameDate'][1:]:  # Start at second day (no prior data)
            # Get the subset of games that have occured BEFORE the given date
            season_games_predate = season_games.loc[
                season_games['GameDate'] < DATE
            ]

            seasonTeamsPredate = preAggSeasonGames(season_games_predate)
            seasonTeamsPredate['dataPriorTo'] = DATE
            if first_date:
                output = seasonTeamsPredate.copy()
                first_date = False
            else:
                output = output.append(seasonTeamsPredate)
            print(f"Added {len(seasonTeamsPredate)} records to output for date {DATE}")

        path = '/Users/Ryan/Documents/projects/ncaa-basketball/data/predate_games/'+str(SEASON)+'.csv'
        output.to_csv(
            path,
            index=False
        )
        print(f"Successfully created {path}")

