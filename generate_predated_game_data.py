
import pandas as pd
from db import get_db
from st_functions import get_seasons_list
from preAggregateSeasonTeams import pre_aggregate_season

if __name__ == "__main__":
    db = get_db()
    seasons_list = get_seasons_list(_db=db)
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
            print(f"Working on date: {str(DATE)}")
            # TODO remove limiter
            # Get the subset of games that have occured BEFORE the given date
            season_games_predate = season_games.loc[
                season_games['GameDate'] < DATE
            ]

            # TODO potentially modify this for models on in-game stats like possessions
            # For now, only worry about margin
            season_games_ondate = season_games.loc[
                season_games['GameDate'] == DATE
            ][[
                'Season',
                'GameDate',
                'TmName',
                'OppName',
                'TmLoc',
                'TmMargin'
            ]].rename(
                columns={
                    'TmMargin': 'Tm1_GameMargin',
                    'TmName': 'Tm1',
                    'OppName': 'Tm2',
                    'TmLoc': 'Tm1_Loc'
                }
            )
            num_games_on_date = len(season_games_ondate)

            # Aggregate those pre-dated games
            season_teams_predate = pre_aggregate_season(
                _db='',
                szn='',
                how='df',
                df=season_games_predate
            )
            # TODO is this where opponent-adjusting comes in?

            # Create two copies and rename columns to prep for merging
            st_pd_tm1 = season_teams_predate.add_prefix(
                'Tm1_'
            ).rename(
                columns={
                    'Tm1_TmName': 'Tm1',
                    'Tm1_Season': 'Season'
                }
            )
            st_pd_tm2 = season_teams_predate.add_prefix(
                'Tm2_'
            ).rename(
                columns={
                    'Tm2_TmName': 'Tm2',
                    'Tm2_Season': 'Season'
                }
            )
            # Merge in both pre-dated summaries into on-date games
            # use inner joins to only keep records where both teams have a prior game
            season_games_ondate = pd.merge(
                left=season_games_ondate,
                right=st_pd_tm1,
                on=['Tm1', 'Season'],
                how='inner'
            )
            season_games_ondate = pd.merge(
                left=season_games_ondate,
                right=st_pd_tm2,
                on=['Tm2', 'Season'],
                how='inner'
            )

            if first_date:
                season_games_output = season_games_ondate
                first_date = False
            else:
                season_games_output = season_games_output.append(season_games_ondate)
            print(f"Added {len(season_games_ondate)} games to output ({num_games_on_date} total)")

        # TODO write out CSV
        path = 'data/predate_games/'+str(SEASON)+'.csv'
        season_games_output.to_csv(
            path,
            index=False
        )
        print(f"Successfully created {path}")



# TODO there are duplicate records in all games records
# either split by team name or by some other fashion