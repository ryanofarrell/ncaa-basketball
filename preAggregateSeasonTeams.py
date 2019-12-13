

import pandas as pd
from db import get_db
from st_functions import get_seasons_list


def pre_aggregate_season(_db, szn, how='db', df=''):
    assert how in ['db', 'df'], "Incorrect 'how'"
    if how == 'db':
        season_games = pd.DataFrame(list(_db.games.find({'Season': szn})))
    else:
        assert isinstance(df, pd.DataFrame)
        season_games = df
    cols = list(season_games.columns)

    # Get list of columns to aggregate
    metrics_list = ['PF', 'Margin',
                    'FGM', 'FGA',
                    'FG3M', 'FG3A',
                    'FG2M', 'FG2A',
                    'FTA', 'FTM',
                    'Ast', 'ORB',
                    'DRB', 'TRB',
                    'TO', 'Stl',
                    'Blk', 'Foul',
                    'Poss', 'Mins',
                    'Game', 'Win']
    metrics_to_sum = ['GameOT', 'isRegularSeason']
    for prefix in ['Tm', 'Opp']:
        for metric in metrics_list:
            metrics_to_sum.append(prefix+metric)

    columns_to_sum = list(set(metrics_to_sum).intersection(cols))
    [columns_to_sum.append(field) for field in ['Season', 'TmName']]
    # print(columnsToAgg)
    # nonAgg = list(set(cols).difference(metricsToAgg))
    # print(nonAgg)

    # Sum season values for teach team for each key column
    tm_sum_season_teams = season_games[columns_to_sum].groupby(
        ['TmName', 'Season']).sum().reset_index()

    # Flip values to get into games dataframe
    tm_sum_season_teams_flipped = tm_sum_season_teams.rename(
        columns={'TmName': 'OppName'}
    )

    # Merge in each team's summarized values into opponent's side of games
    opp_season_games = pd.merge(
        left=season_games[['Season', 'TmName', 'OppName']],
        right=tm_sum_season_teams_flipped,
        how='inner',
        on=['OppName', 'Season']
    )
    del opp_season_games['OppName']

    # Sum season values for each team's opponents
    # Note all fields in season_games_opp are the opponent's summarized values
    # Groupby then rename all fields to OppSum_<Field>
    opp_sum_season_teams = opp_season_games.groupby(
        by=['TmName', 'Season']
    ).sum().reset_index()
    for col in opp_sum_season_teams.columns:
        if col not in ['Season','TmName']:
            opp_sum_season_teams = opp_sum_season_teams.rename(
                columns={col: 'OppSum_'+col}
            )
    
    # Merge team's values and opponent sum values together
    season_teams_combined = pd.merge(
        left=tm_sum_season_teams,
        right=opp_sum_season_teams,
        how='inner',
        on=['Season', 'TmName']
    )

    return season_teams_combined


if __name__ == "__main__":
    db = get_db()
    db.seasonteams.drop()
    SEASONS = get_seasons_list(_db=db)
    for season in SEASONS:
        print(f"Working on the {season} season.")
        data = pre_aggregate_season(_db=db, szn=season)
        print(f"Converting {len(data)} new season-team records to dict")
        data_dict = data.to_dict('records')
        print(f"Inserting {len(data_dict)} records to database")
        db.seasonteams.insert_many(data_dict, ordered=False)
        print(f"Inserted {len(data_dict)} records.")
