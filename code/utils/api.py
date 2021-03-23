import pandas as pd
from itertools import product

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


def isAscendingRank(pref, metr):
    # More is better: Tm * Positive metric = 1 (descending)
    # More is better: Opp * Negative metric = 1 (descending)
    # Less is better: Tm * Negative metric = -1 (ascending)
    # Less is Better: Opp * Positive metric = -1 (ascending)

    # Opp = -1, Negative = -1, Tm = 1, Pos = 1

    neg_metrics = ['Foul', 'TO']
    metric_val = -1 if metr in neg_metrics else 1
    prefix_val = -1 if pref == 'Opp' else 1

    result = metric_val*prefix_val == -1 

    return result


def opponentAdjust(
        data,
        PREFIXES=['Tm'],
        METRICS=['Margin', 'PF'],
        DENOMS=['per40'],
        includeOARankFields=False,
        includeNormFields=False,
        includeNormRankFields=False
    ):
    """Opponent-adjusts the given prefixes, metrics, and denominators in the provided dataframe

    Arguments:
        data {DataFrame} -- dataframe of pre-aggregated results - may span multiple seasons

    Keyword Arguments:
        PREFIXES {list} -- prefixes to opponent-adjust (default: {['Tm']})
        METRICS {list} -- metrics to opponent-adjust (default: {['Margin', 'PF']})
        DENOMS {list} -- denominators to opponent-adjust (default: {['per40']})
        includeOARankFields {bool} -- if the returned dataframe should also include the rank of the OA metric (default: {False})
        includeNormFields {bool} -- if the returned dataframe should also include the normalized fields (default: {False})
        includeNormRankFields {bool} -- if the returned dataframe should also include the rank of the normalized metric (default: {False})


    Returns:
        [DataFrame] -- The provided dataframe + all opponent-adjusted columns (prefixed with 'OA_')
    """

    for PREFIX, METRIC, DENOM in product(PREFIXES, METRICS, DENOMS):
        assert PREFIX in ['Opp', 'Tm'], 'Invalid prefix'
        assert DENOM in ['per40', 'perPoss', 'perGame'], 'Invalid denom'
        # TODO insert assert that all columns needed exist in data

        OTHER_PREFIX = 'Opp' if PREFIX == 'Tm' else 'Tm'
        DENOM_FIELD = 'Mins' if DENOM == 'per40' else DENOM[-4:]
        NORMALIZE_CONST = 40 if DENOM == 'per40' else 1

        # Perform the actual opponent-adjusting
        data[PREFIX+METRIC+DENOM] = data[PREFIX+METRIC] / data[PREFIX+DENOM_FIELD] * NORMALIZE_CONST
        data['OA_'+PREFIX+METRIC+DENOM] = \
            (data[PREFIX+METRIC+DENOM]) - \
            (
                (data['OppSum_'+OTHER_PREFIX+METRIC] - data[PREFIX+METRIC]) /
                (data['OppSum_'+OTHER_PREFIX+DENOM_FIELD] - data[PREFIX+DENOM_FIELD])
            ) * NORMALIZE_CONST

        # Rank normalized fields if desired
        if includeNormRankFields:
            data['Rnk_'+PREFIX+METRIC+DENOM] = data.groupby(
                'Season'
            )[PREFIX+METRIC+DENOM].rank(
                'min', ascending=isAscendingRank(PREFIX, METRIC)
            )

        # Delete normalized fields if desired
        if not includeNormFields:
            del data[PREFIX+METRIC+DENOM]

        # Rank OA fields if desired
        if includeOARankFields:
            data['Rnk_OA_'+PREFIX+METRIC+DENOM] = data.groupby(
                'Season'
            )['OA_'+PREFIX+METRIC+DENOM].rank(
                'min', ascending=isAscendingRank(PREFIX, METRIC)
            )


    return data

