
from db import get_db
import pandas as pd
from st_functions import return_seasons_list


def preAggregateSeason(_db, season):
    """Returns a sum of all relevant fields as a dataframe. Gets
    games from DB, trims down to relvant columns, and sums by each
    unique season-team combination.

    Arguments:
        _db {database connection} -- Connection to DB;
        season {int} -- the season to aggregate

    Returns:
        DataFrame -- Contains the TmName, Season, and sum of metrics
    """
    gamesDf = pd.DataFrame(list(_db.games.find({'Season': season})))
    cols = list(gamesDf.columns)

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
    metricsToAgg = ['GameOT', 'isRegularSeason']
    for prefix in ['Tm', 'Opp']:
        for metric in metricsList:
            metricsToAgg.append(prefix+metric)

    columnsToAgg = list(set(metricsToAgg).intersection(cols))
    [columnsToAgg.append(field) for field in ['Season', 'TmName']]
    # print(columnsToAgg)
    # nonAgg = list(set(cols).difference(metricsToAgg))
    # print(nonAgg)

    gamesDf = gamesDf[columnsToAgg].groupby(
        ['TmName', 'Season']).sum().reset_index()

    return gamesDf


if __name__ == "__main__":
    db = get_db()
    db.seasonteams.drop()
    seasons = return_seasons_list(_db=db)
    print(seasons)
    for season in seasons:
        print(f"Working on the {season} season.")
        data = preAggregateSeason(_db=db, season=season)
        print(f"Converting {len(data)} new season-team records to dict")
        data_dict = data.to_dict('records')
        print(f"Inserting {len(data_dict)} records to database")
        db.seasonteams.insert_many(data_dict, ordered=False)
        print(f"Inserted {len(data_dict)} records.")