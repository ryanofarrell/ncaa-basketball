# %% Imports
from datetime import timedelta
from typing import Literal
from helpers import (
    dfToTable,
    executeSql,
    get_unique_permutations,
    getRelativeFp,
    logger,
    readSql,
)
import pandas as pd
from fnmatch import fnmatch
import numpy as np
import firebase_admin
from firebase_admin import credentials, firestore
import os

os.environ["FIRESTORE_EMULATOR_HOST"] = "127.0.0.1:8080"

cred = credentials.Certificate("teach-tapes-dev-firebase-adminsdk-65ndp-75822e6236.json")
# cred = credentials.AnonymousCredentials()
app = firebase_admin.initialize_app(cred)
db = firestore.client()


# %% CONSTANTS
OTHERPREFIXMAP = {"tm": "opp", "opp": "tm"}

# %% Logging
log = logger(
    fp=getRelativeFp(__file__, "logs/loadDatabase.log"),
    fileLevel=10,
    consoleLevel=20,
    removeOldFile=True,
)


# %% Reload data to firestore function
def reload_firestore(coll: str, df: pd.DataFrame, id_col: str) -> None:
    log.info(f"Beginning delete of {coll}")
    # Delete all documents in collection with batched deletes
    docs = db.collection(coll).stream()
    batch = db.batch()
    i = 0
    for doc in docs:
        batch.delete(doc.reference)
        i += 1
        if i % 500 == 0:
            batch.commit()
            batch = db.batch()
            log.info(f"Deleted {i} records")
    batch.commit()
    log.info(f"Deleted all in {coll}")
    batch = db.batch()
    for i, row in df.iterrows():
        assert isinstance(i, int), "Row label must be int!"
        doc_ref = db.collection(coll).document(row[id_col])
        batch.set(doc_ref, row.to_dict())
        if i % 500 == 0:
            batch.commit()
            batch = db.batch()
            log.info(f"Committed {i} ({i/len(df):.2%})")
    batch.commit()


# %% Load teams
@log.timeFuncInfo
def load_teams(replace_firestore: bool = False):
    "No dependencies"

    teams = pd.read_csv("./data/MTeams.csv")
    teams.columns = [x.lower() for x in teams.columns]

    teams["slug"] = teams["teamname"].str.replace(" ", "-").replace("[^a-zA-Z0-9\-]", "", regex=True).str.lower()
    assert teams["slug"].unique().__len__() == len(teams), "Non-unique slugs!"

    # Split columns into snake_case
    teams.rename(
        columns={
            "teamid": "team_id",
            "teamname": "team_name",
            "firstd1season": "first_d1_season",
            "lastd1season": "last_d1_season",
        },
        inplace=True,
    )
    if replace_firestore:
        reload_firestore("teams", teams, "slug")

    # Load teams into db
    dfToTable(teams, "teams", "ncaa.db", "replace")
    for p in get_unique_permutations(["team_id", "team_name"]):
        executeSql(f"CREATE INDEX teams_{'_'.join(p)} ON teams ({', '.join(p)})")


# %% Load games
@log.timeFuncInfo
def load_games(season: int | None = None, replace_firestore: bool = False):
    "Depends on teams"

    # Read on data
    seasons = pd.read_csv("./data/MSeasons.csv")
    games = pd.read_csv("./data/MRegularSeasonDetailedResults.csv")

    # Normalize col names
    games.columns = [x.lower() for x in games.columns]
    seasons.columns = [x.lower() for x in seasons.columns]

    if season is not None:
        games = games[games["season"] == season]

    # Get dame dates
    seasons["dayzero"] = pd.to_datetime(seasons["dayzero"])
    games = games.merge(seasons[["season", "dayzero"]], on="season", how="left")
    games["date"] = games.apply(
        lambda row: (row["dayzero"] + timedelta(days=row["daynum"])).strftime("%Y-%m-%d"),
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
    rename = {**rename, "wteamid": "tm_team_id", "lteamid": "opp_team_id"}
    games.rename(columns=rename, inplace=True)
    games["opp_loc"] = np.where(games["tm_loc"] == "N", "N", np.where(games["tm_loc"] == "A", "H", "A"))

    # Add game key
    teams = readSql("select * from teams")
    teams_dict = {}
    for _, row in teams.iterrows():
        teams_dict[row["team_id"]] = row["team_name"]
    games["game_key"] = games.apply(
        lambda x: f"{teams_dict[x['tm_team_id']]}-vs-{teams_dict[x['opp_team_id']]}", axis=1
    )

    # Add additional columns
    for prefix in ["tm", "opp"]:
        games[f"{prefix}_fga2"] = games[f"{prefix}_fga"] - games[f"{prefix}_fga3"]
        games[f"{prefix}_fgm2"] = games[f"{prefix}_fgm"] - games[f"{prefix}_fgm3"]
        games[f"{prefix}_tr"] = games[f"{prefix}_or"] + games[f"{prefix}_dr"]
        games[f"{prefix}_mins"] = games[f"{prefix}_mins"] = 40 + games["num_ot"] * 5
        games[f"{prefix}_game"] = 1
        games[f"{prefix}_win"] = 1 * (games[f"{prefix}_pts"] > games[f"{OTHERPREFIXMAP[prefix]}_pts"])
        games[f"{prefix}_loss"] = games[f"{prefix}_game"] - games[f"{prefix}_win"]
        # games[f"{prefix}_rsgame"] = 1
        games[f"{prefix}_poss"] = (
            (
                games["tm_fga"]
                + 0.4 * games["tm_fta"]
                - 1.07 * (games["tm_or"] / (games["tm_or"] + games["opp_dr"])) * (games["tm_fga"] - games["tm_fgm"])
                + games["tm_to"]
            )
            + (
                games["opp_fga"]
                + 0.4 * games["opp_fta"]
                - 1.07 * (games["opp_or"] / (games["opp_or"] + games["tm_dr"])) * (games["opp_fga"] - games["opp_fgm"])
                + games["opp_to"]
            )
        ) * 0.5
        games[f"{prefix}_margin"] = games[f"{prefix}_pts"] - games[f"{OTHERPREFIXMAP[prefix]}_pts"]
        games[f"{prefix}_availor"] = games[f"{prefix}_or"] + games[f"{OTHERPREFIXMAP[prefix]}_dr"]
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

    # Add slug
    games["slug"] = games.apply(lambda x: f"{x['date']}-{x['game_key']}".lower().replace(" ", "-"), axis=1)

    # Reorder columns for easier analysis
    first_cols = [
        "date",
        "season",
        "game_key",
        "slug",
        "tm_team_id",
        "opp_team_id",
        "tm_loc",
        "opp_loc",
        "tm_pts",
        "opp_pts",
    ]
    remainder_cols = sorted([x for x in games.columns if x not in first_cols])
    games = games[first_cols + remainder_cols]

    # Load data
    if replace_firestore:
        reload_firestore("games", games, "slug")

    # TODO add postseason

    # Drop old, set up new, add indexes, load data
    if season is None:
        executeSql("drop table if exists games")
        q = f"""
        create table games (
            season integer not null,
            date TEXT not null,
            game_key TEXT not null,
            slug TEXT not null,
            tm_team_id integer not null,
            opp_team_id integer not null,
            tm_loc string not null,
            opp_loc string not null,
            tm_pts integer not null,
            opp_pts integer not null,
            {' real not null, '.join(remainder_cols)} real not null,
            primary key (season, date asc, game_key asc, tm_team_id asc)
        )
        """
        executeSql(q)
        perms = get_unique_permutations(["date", "season", "game_key", "tm_team_id"])
        log.info(f"Creating {len(perms)} indexes on games")
        for p in perms:
            executeSql(f"CREATE INDEX games_{'_'.join(p)} on games ({', '.join(p)})")
    else:
        executeSql(f"delete from games where season = {season}")

    dfToTable(
        games,
        table="games",
        db="ncaa.db",
        ifExists="append" if season is not None else "replace",
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
        season_dates = list(pd.date_range(series["min"], series["max"], freq="D").strftime("%Y-%m-%d"))
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


# %% Scrape single vegas season
@log.timeFuncDebug
def single_season_vegas_load(season: int):
    df = pd.read_excel(
        f"https://www.sportsbookreviewsonline.com/scoresoddsarchives/ncaabasketball/ncaa%20basketball%20{season - 1}-{str(season)[-2:]}.xlsx"
    )
    df.columns = [x.lower() for x in df.columns]

    # Drop bad dates - date must be >100
    pre_drop_len = len(df)
    bad_dates = df["date"] < 100
    log.debug(f"Dropping {sum(bad_dates) * 2:,.0f} bad dated records")
    idx_to_drop = []
    for idx, _ in df.loc[bad_dates].iterrows():
        idx_to_drop.append(idx)
        if idx % 2 == 1:  # If odd, drop even number prior
            idx_to_drop.append(idx - 1)
        else:  # If even, drop next number
            idx_to_drop.append(idx + 1)
    df = df.drop(idx_to_drop).reset_index(drop=True)
    assert len(df) == pre_drop_len - sum(bad_dates) * 2, "missing a date or two"

    # Parse date from integer date column
    df[["mo", "dy"]] = df["date"].apply(lambda x: [str(x)[:-2], str(x)[-2:]]).to_list()
    df["year"] = np.where(df["mo"].astype(int) < 7, str(season), str(season - 1))
    df["date"] = pd.to_datetime(df["year"] + "-" + df["mo"] + "-" + df["dy"]).dt.strftime("%Y-%m-%d")
    df.drop(inplace=True, labels=["year", "mo", "dy"], axis=1)

    # Self-join to get all game data in one row
    df = pd.merge(
        df.iloc[[x for x in df.index if x % 2 == 0]].reset_index(drop=True),
        df.iloc[[x for x in df.index if x % 2 == 1]].reset_index(drop=True),
        left_index=True,
        right_index=True,
        suffixes=["_tm", "_opp"],
    )
    preLen = len(df)
    df = df.loc[df["date_tm"] == df["date_opp"]].reset_index(drop=True)
    df = df.loc[np.abs((df["rot_tm"] - df["rot_opp"])) == 1].reset_index(drop=True)
    if len(df) != preLen:
        log.debug(f"Dropped {preLen-len(df)} records due to date mismatch")
    assert (df["date_tm"] == df["date_opp"]).all(), "Mismatch dates"
    assert np.abs((df["rot_tm"] - df["rot_opp"])).max() == 1, "Mismatch rots"

    # remove unused rows, rename some things
    df.drop(
        ["date_opp", "rot_tm", "rot_opp"] + [f"{x}_{y}" for y in ["tm", "opp"] for x in ["1st", "2nd", "2h"]],
        axis=1,
        inplace=True,
    )
    df.rename(columns={"date_tm": "date"}, inplace=True)

    # Get team IDs
    ts = pd.read_csv("data/MTeamSpellings.csv", sep=",", encoding="cp1252")
    ts.columns = [x.lower() for x in ts.columns]
    ts["teamnamespelling_nospace"] = ts["teamnamespelling"].str.replace(" ", "")
    ts = ts.groupby(["teamid", "teamnamespelling_nospace"]).size().reset_index()
    for col in ["team_tm", "team_opp"]:
        df[col] = df[col].str.lower()
        df[col] = df[col].str.replace(".", "", regex=False)
        df[col] = df[col].str.replace("\xa0", "", regex=False)

    df = (
        pd.merge(
            df,
            ts[["teamnamespelling_nospace", "teamid"]],
            left_on=["team_tm"],
            right_on=["teamnamespelling_nospace"],
            how="left",
        )
        .rename(columns={"teamid": "tm_teamid"})
        .drop(["teamnamespelling_nospace"], axis=1)
        .merge(
            ts[["teamnamespelling_nospace", "teamid"]],
            left_on=["team_opp"],
            right_on=["teamnamespelling_nospace"],
            how="left",
        )
        .rename(columns={"teamid": "opp_teamid"})
        .drop(["teamnamespelling_nospace"], axis=1)
    )
    # TODO handle this
    sorted(
        list(
            set(
                list(df.loc[pd.isna(df["tm_teamid"])]["team_tm"].unique())
                + list(df.loc[pd.isna(df["opp_teamid"])]["team_opp"].unique())
            )
        )
    )

    # Drop missing teams
    preLen = len(df)
    df = df.loc[(~pd.isna(df["tm_teamid"])) & (~pd.isna(df["opp_teamid"]))].reset_index(drop=True)
    log.debug(f"Dropped {preLen - len(df)} records due to missing names {(preLen - len(df)) / preLen:.2%}")

    # Figure out open and close
    for col in ["open_tm", "close_tm", "open_opp", "close_opp"]:
        df[col] = np.where(df[col].str.lower().isin(["pk", "p"]), 0, df[col])

    # Drop the NL values (no line)
    # preLen = len(df)
    # df = df.loc[
    #     ~(df[["open_tm", "close_tm", "open_opp", "close_opp"]] == "NL").any(axis=1)
    # ].reset_index(drop=True)
    # print(
    #     f"Dropped {preLen - len(df)} records due to NL vals {(preLen - len(df)) / preLen:.2%}"
    # )

    def get_line(r):
        "tm openline, tm_closeline, game_open_ou, game_close_ou, tm_ml, opp_ml"

        def _get_val_type(val: str | float) -> Literal["line", "ou", "NL"]:
            if isinstance(val, str):
                if val.strip() == "NL":
                    return "NL"
                # print(r)
                return "NL"
                # raise ValueError(val)
            if 0 <= val <= 70:
                return "line"
            if val > 70:
                return "ou"
            return "NL"

        if r["ml_tm"] == "NL":
            tm_ml = None
        else:
            tm_ml = r["ml_tm"]
        if r["ml_opp"] == "NL":
            opp_ml = None
        else:
            opp_ml = r["ml_opp"]

        types = (
            (_get_val_type(r["open_tm"]), _get_val_type(r["open_opp"])),
            (_get_val_type(r["close_tm"]), _get_val_type(r["close_opp"])),
        )

        # OU is on the side the favorite is
        # Open open, close close, tm-opp
        match types[0]:
            case ("ou", "line"):
                tm_openline = r["open_opp"]
                game_openou = r["open_tm"]
            case ("ou", "NL"):
                tm_openline = None
                game_openou = r["open_tm"]
            case ("NL", "line"):
                tm_openline = r["open_opp"]
                game_openou = None
            case ("line", "ou"):
                tm_openline = -r["open_tm"]
                game_openou = r["open_opp"]
            case ("NL", "ou"):
                tm_openline = None
                game_openou = r["open_opp"]
            case ("line", "NL"):
                tm_openline = -r["open_tm"]
                game_openou = None
            case ("NL", "NL"):
                tm_openline = None
                game_openou = None
            case ("line", "line"):
                tm_openline = None
                game_openou = None
            case _:
                print(r)
                raise ValueError(types[0])

        match types[1]:
            case ("ou", "line"):
                tm_closeline = r["close_opp"]
                game_closeou = r["close_tm"]
            case ("ou", "NL"):
                tm_closeline = None
                game_closeou = r["close_tm"]
            case ("NL", "line"):
                tm_closeline = r["close_opp"]
                game_closeou = None
            case ("line", "ou"):
                tm_closeline = -r["close_tm"]
                game_closeou = r["close_opp"]
            case ("NL", "ou"):
                tm_closeline = None
                game_closeou = r["close_opp"]
            case ("line", "NL"):
                tm_closeline = -r["close_tm"]
                game_closeou = None
            case ("NL", "NL"):
                tm_closeline = None
                game_closeou = None
            case ("ou", "ou"):
                tm_closeline = None
                game_closeou = None
            case ("line", "line"):
                tm_closeline = None
                game_closeou = None

            case _:
                print(r)
                raise ValueError(types[1])

        # assert max(np.abs(tm_openline), np.abs(tm_closeline)) < min(
        #     game_closeou, game_openou
        # )
        return (tm_openline, tm_closeline, game_openou, game_closeou, tm_ml, opp_ml)

    df[
        [
            "tm_openline",
            "tm_closeline",
            "game_openou",
            "game_closeou",
            "tm_ml",
            "opp_ml",
        ]
    ] = df.apply(lambda x: get_line(x), axis=1).to_list()

    for col in [
        "tm_openline",
        "tm_closeline",
        "game_openou",
        "game_closeou",
        "tm_ml",
        "opp_ml",
    ]:
        df[col] = df[col].astype(float, errors="ignore")

    # Bring together to games
    games = readSql(f"select * from games where season = {season}")
    df = pd.merge(
        df,
        games[["date", "tm_teamid", "opp_teamid", "season", "game_key"]],
        how="left",
        on=["date", "tm_teamid", "opp_teamid"],
    )

    preLen = len(df)
    df = df.loc[~pd.isna(df["game_key"])].reset_index(drop=True)
    log.debug(f"Dropped {preLen - len(df)} records due to no matching game {(preLen - len(df)) / preLen:.2%}")

    # Duplicate for database
    dup = df.copy()
    dup.rename(
        columns={
            "tm_ml": "opp_ml",
            "opp_ml": "tm_ml",
            "tm_teamid": "opp_teamid",
            "opp_teamid": "tm_teamid",
        },
        inplace=True,
    )
    dup["tm_openline"] *= -1
    dup["tm_closeline"] *= -1
    df = pd.concat([df, dup], ignore_index=True)
    df = df.sort_values(by=["date", "game_key", "tm_teamid"]).reset_index(drop=True)

    df = df[
        [
            "season",
            "date",
            "game_key",
            "tm_teamid",
            "opp_teamid",
            "tm_openline",
            "tm_closeline",
            "tm_ml",
            "opp_ml",
            "game_openou",
            "game_closeou",
        ]
    ]
    return df


# %% Scrape all vegas seasons
@log.timeFuncInfo
def load_vegas():

    executeSql("drop table if exists vegas_lines")
    q = f"""
    create table vegas_lines (
        season integer not null,
        date TEXT not null,
        game_key TEXT not null,
        tm_teamid integer not null,
        opp_teamid integer not null,
        tm_openline real,
        tm_closeline real,
        tm_ml real,
        opp_ml real,
        game_openou real,
        game_closeou real,

        primary key (season, date asc, game_key asc, tm_teamid asc)
    )
    """
    executeSql(q)
    perms = get_unique_permutations(["date", "season", "game_key", "tm_teamid", "opp_teamid"])
    log.info(f"Creating {len(perms)} indexes on vegas_lines")
    for p in perms:
        executeSql(f"CREATE INDEX vegas_{'_'.join(p)} on vegas_lines ({', '.join(p)})")

    for season in range(2008, 2023):
        log.info(season)
        df = single_season_vegas_load(season)
        dfToTable(
            df,
            table="vegas_lines",
            db="ncaa.db",
            ifExists="append",
        )


# %%
if __name__ == "__main__":

    # Handler
    load_teams()
    load_games()
    load_calendar()
    load_vegas()

# %%
