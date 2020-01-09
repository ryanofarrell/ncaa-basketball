
import pandas as pd
from itertools import product


# TODO switch back the allow_output_mutation=True once bug 
#@st.cache
def cache_game_data(q, f, _db):
    returned_games = pd.DataFrame(list(_db.games.find(q, f)))
    return returned_games


def get_teams_list(_db):
    """Get a list of all the teams with at least one game ever

    Keyword Arguments:
        _db {database connection} -- Connection to MongoDB

    Returns:
        List -- Every team with at least 1 game played ever
    """
    pipeline = [{'$group': {'_id': {'Team': '$TmName'}}},
                {'$sort': {'_id': 1}}
                ]
    results = _db.games.aggregate(pipeline)
    teams_list = []
    for itm in results:
        teams_list.append(itm['_id']['Team'])
    return teams_list


def get_seasons_list(_db):
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


def is_ascending_rank(pref, metr):
    # More is better: Tm * Positive metric = 1 (descending)
    # More is better: Opp * Negative metric = 1 (descending)
    # Less is better: Tm * Negative metric = -1 (ascending)
    # Less is Better: Opp * Positive metric = -1 (ascending)

    # Opp = -1, Negative = -1, Tm = 1, Pos = 1

    neg_metrics = ['Foul', 'TO']
    metric_val = -1 if metr in neg_metrics else 1
    prefix_val = -1 if pref == 'Opp' else 1

    result = True if metric_val*prefix_val == -1 else False

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
                'min', ascending=is_ascending_rank(PREFIX, METRIC)
            )

        # Delete normalized fields if desired
        if not includeNormFields:
            del data[PREFIX+METRIC+DENOM]

        # Rank OA fields if desired
        if includeOARankFields:
            data['Rnk_OA_'+PREFIX+METRIC+DENOM] = data.groupby(
                'Season'
            )['OA_'+PREFIX+METRIC+DENOM].rank(
                'min', ascending=is_ascending_rank(PREFIX, METRIC)
            )


    return data




if __name__ == "__main__":
    pass
