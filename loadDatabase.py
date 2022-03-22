# %% Imports
from datetime import timedelta
from helpers import dfToTable, readSql
import os
import pandas as pd
from fnmatch import fnmatch
import numpy as np

# %% CONSTANTS
OTHERPREFIXMAP = {"tm": "opp", "opp": "tm"}

# %%
if __name__ == "__main__":

    # Read on data
    games = pd.read_csv("./data/MRegularSeasonDetailedResults.csv")
    seasons = pd.read_csv("./data/MSeasons.csv")
    teams = pd.read_csv("./data/MTeams.csv")

    # Normalize col names
    games.columns = [x.lower() for x in games.columns]
    seasons.columns = [x.lower() for x in seasons.columns]
    teams.columns = [x.lower() for x in teams.columns]

    # Load teams into db
    dfToTable(teams, "teams", "ncaa.db", "replace", ["teamid"])

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
        "season",
        "date",
        "game_key",
        "tm_teamid",
        "opp_teamid",
        "tm_pts",
        "opp_pts",
    ]
    games = games[
        first_cols + sorted([x for x in games.columns if x not in first_cols])
    ]

    # TODO add postseason

    # create database
    dfToTable(
        games,
        table="games",
        db="ncaa.db",
        ifExists="replace",
        indexCols=["season", "date", "game_key", "tm_teamid"],
    )

    # Create calendar table of dates
    cal = pd.DataFrame(
        {
            "date": pd.date_range(
                games["date"].min(), games["date"].max(), freq="D"
            ).strftime("%Y-%m-%d")
        }
    )
    dfToTable(cal, "calendar", "ncaa.db", ifExists="replace", indexCols=["date"])

# %%
