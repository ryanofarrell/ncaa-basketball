
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








if __name__ == "__main__":
    pass
