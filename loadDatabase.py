# %% Imports
from datetime import timedelta
from itertools import permutations
from helpers import (
    dfToTable,
    executeSql,
    get_unique_permutations,
    getRelativeFp,
    logger,
    readSql,
)
import os
import pandas as pd
from fnmatch import fnmatch
import numpy as np

# %% CONSTANTS
OTHERPREFIXMAP = {"tm": "opp", "opp": "tm"}

# %% Logging
log = logger(
    fp=getRelativeFp(__file__, f"logs/loadDatabase.log"),
    fileLevel=10,
    consoleLevel=20,
)

# %% Load calendar
@log.timeFuncInfo
def load_calendar():
    "Relies on games to be loaded into database"

    q = """
    select
        season
        ,min(date) as min
        ,max(date) as max
    from games
    group by season
    """
    seasonGameDates = readSql(q)
    season_list: list[int] = []
    dates: list[str] = []
    day_nums: list[int] = []
    for seas, series in seasonGameDates.iterrows():
        season_dates = list(
            pd.date_range(series["min"], series["max"], freq="D").strftime("%Y-%m-%d")
        )
        dates += season_dates
        season_list += [seas] * len(season_dates)
        day_nums += [x + 1 for x in range(len(season_dates))]
    cal = pd.DataFrame({"season": season_list, "date": dates, "day_num": day_nums})

    # drop old, make table, add indexe, load data
    executeSql("drop table if exists calendar")
    q = f"""
    create table calendar (
        season integer not null,
        date TEXT not null,
        day_num integer not null,
        primary key (season asc, date asc)
    )
    """
    executeSql(q)
    for p in get_unique_permutations(cal.columns):
        executeSql(f"CREATE INDEX calendar_{'_'.join(p)} ON calendar ({', '.join(p)})")
    dfToTable(cal, "calendar", "ncaa.db", ifExists="append")


# %% Load teams
@log.timeFuncInfo
def load_teams():
    "No dependencies"

    teams = pd.read_csv("./data/MTeams.csv")
    teams.columns = [x.lower() for x in teams.columns]
    # Load teams into db
    dfToTable(teams, "teams", "ncaa.db", "replace")
    for p in get_unique_permutations(['teamid','teamname']):
        executeSql(f"CREATE INDEX teams_{'_'.join(p)} ON teams ({', '.join(p)})")



# %% Load games
@log.timeFuncInfo
def load_games():
    "Depends on teams"

    # Read on data
    seasons = pd.read_csv("./data/MSeasons.csv")
    games = pd.read_csv("./data/MRegularSeasonDetailedResults.csv")

    # Normalize col names
    games.columns = [x.lower() for x in games.columns]
    seasons.columns = [x.lower() for x in seasons.columns]

    # Get dame dates
    seasons["dayzero"] = pd.to_datetime(seasons["dayzero"])
    games = games.merge(seasons[["season", "dayzero"]], on="season")
    games["date"] = games.apply(
        lambda row: (row["dayzero"] + timedelta(days=row["daynum"])).strftime(
            "%Y-%m-%d"
        ),
        axis=1,
    )
    del games["dayzero"], games["daynum"]

    # Rename columns
    rename_map = {"w": "tm", "l": "opp"}
    rename = {"numot": "num_ot", "wscore": "tm_pts", "lscore": "opp_pts"}
    for col in [
        "teamid",
        "loc",
        "fgm",
        "fga",
        "fgm3",
        "fga3",
        "ftm",
        "fta",
        "or",
        "dr",
        "ast",
        "to",
        "stl",
        "blk",
        "pf",
    ]:
        for prefix in ["w", "l"]:
            rename[f"{prefix}{col}"] = f"{rename_map[prefix]}_{col}"
    games.rename(columns=rename, inplace=True)
    games["opp_loc"] = np.where(
        games["tm_loc"] == "N", "N", np.where(games["tm_loc"] == "A", "H", "A")
    )

    # Add game key
    teams = readSql("select * from teams")
    teams_dict = {}
    for idx, row in teams.iterrows():
        teams_dict[row["teamid"]] = row["teamname"]
    games["game_key"] = games.apply(
        lambda x: f"{teams_dict[x['tm_teamid']]}>{teams_dict[x['opp_teamid']]}", axis=1
    )

    # Add additional columns
    for prefix in ["tm", "opp"]:
        games[f"{prefix}_fga2"] = games[f"{prefix}_fga"] - games[f"{prefix}_fga3"]
        games[f"{prefix}_fgm2"] = games[f"{prefix}_fgm"] - games[f"{prefix}_fgm3"]
        games[f"{prefix}_tr"] = games[f"{prefix}_or"] + games[f"{prefix}_dr"]
        games[f"{prefix}_mins"] = games[f"{prefix}_mins"] = 40 + games["num_ot"] * 5
        games[f"{prefix}_game"] = 1
        games[f"{prefix}_win"] = 1 * (
            games[f"{prefix}_pts"] > games[f"{OTHERPREFIXMAP[prefix]}_pts"]
        )
        games[f"{prefix}_loss"] = games[f"{prefix}_game"] - games[f"{prefix}_win"]
        # games[f"{prefix}_rsgame"] = 1
        games[f"{prefix}_poss"] = (
            (
                games["tm_fga"]
                + 0.4 * games["tm_fta"]
                - 1.07
                * (games["tm_or"] / (games["tm_or"] + games["opp_dr"]))
                * (games["tm_fga"] - games["tm_fgm"])
                + games["tm_to"]
            )
            + (
                games["opp_fga"]
                + 0.4 * games["opp_fta"]
                - 1.07
                * (games["opp_or"] / (games["opp_or"] + games["tm_dr"]))
                * (games["opp_fga"] - games["opp_fgm"])
                + games["opp_to"]
            )
        ) * 0.5
        games[f"{prefix}_margin"] = (
            games[f"{prefix}_pts"] - games[f"{OTHERPREFIXMAP[prefix]}_pts"]
        )
        games[f"{prefix}_availor"] = (
            games[f"{prefix}_or"] + games[f"{OTHERPREFIXMAP[prefix]}_dr"]
        )
    del games["num_ot"]

    # Duplicate and rename
    def rename_col(col):
        if fnmatch(col, "tm_*"):
            return f"opp_{col[3:]}"
        elif fnmatch(col, "opp_*"):
            return f"tm_{col[4:]}"
        else:
            return col

    dup_games = games.copy()
    dup_games.columns = [rename_col(col) for col in dup_games.columns]
    games = pd.concat([games, dup_games], ignore_index=True)
    games = games.sort_values(by=["date", "game_key"]).reset_index(drop=True)

    # Reorder columns for easier analysis
    first_cols = [
        "date",
        "season",
        "game_key",
        "tm_teamid",
        "opp_teamid",
        "tm_loc",
        "opp_loc",
        "tm_pts",
        "opp_pts",
    ]
    remainder_cols = sorted([x for x in games.columns if x not in first_cols])
    games = games[first_cols + remainder_cols]

    # TODO add postseason

    # Drop old, set up new, add indexes, load data
    executeSql("drop table if exists games")
    q = f"""
    create table games (
        season integer not null,
        date TEXT not null,
        game_key TEXT not null,
        tm_teamid integer not null,
        opp_teamid integer not null,
        tm_loc string not null,
        opp_loc string not null,
        tm_pts integer not null,
        opp_pts integer not null,
        {' integer not null, '.join(remainder_cols)} integer not null,
        primary key (season, date asc, game_key asc, tm_teamid asc)
    )
    """
    executeSql(q)

    perms = get_unique_permutations(
        ["date", "season", "game_key", "tm_teamid", "opp_teamid"]
    )
    log.info(f"Creating {len(perms)} indexes on games")
    for p in perms:
        executeSql(f"CREATE INDEX games_{'_'.join(p)} on games ({', '.join(p)})")
    dfToTable(
        games,
        table="games",
        db="ncaa.db",
        ifExists="append",
    )


# %%
if __name__ == "__main__":

    # Handler
    load_teams()
    load_games()
    load_calendar()

# %%
