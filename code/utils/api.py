import pandas as pd

#%% Basic database pulls
def getSeasonsList(_db):
    '''
    Returns a list of the seasons in seasonteams
    '''
    pipeline = [{'$group': {'_id': {'Season': '$Season'}}},
                {'$sort': {'_id': -1}}
                ]
    results = _db.games.aggregate(pipeline)
    seasons_list = []
    for itm in results:
        seasons_list.append(itm['_id']['Season'])
    return seasons_list


#%% Data manipulation
def preAggSeasonGames(seasonGames):
    cols = list(seasonGames.columns)

    # Get list of columns to aggregate
    metricsList = ['PF', 'Margin',
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
    metricsToSum = ['GameOT', 'isRegularSeason']
    for prefix in ['Tm', 'Opp']:
        for metric in metricsList:
            metricsToSum.append(prefix+metric)

    colsToSum = list(set(metricsToSum).intersection(cols))
    [colsToSum.append(field) for field in ['Season', 'TmName']]
    # print(columnsToAgg)
    # nonAgg = list(set(cols).difference(metricsToAgg))
    # print(nonAgg)

    # Sum season values for teach team for each key column
    tm_sum_season_teams = seasonGames[colsToSum].groupby(
        ['TmName', 'Season']).sum().reset_index()

    # Flip values to get into games dataframe
    tm_sum_season_teams_flipped = tm_sum_season_teams.rename(
        columns={'TmName': 'OppName'}
    )

    # Merge in each team's summarized values into opponent's side of games
    opp_season_games = pd.merge(
        left=seasonGames[['Season', 'TmName', 'OppName']],
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
