# %% Imports
from typing import Counter

import numpy as np
from helpers import dfToTable, executeSql, getRelativeFp, readSql, logger
import pandas as pd


# %% Logging
log = logger(
    fp=getRelativeFp(__file__, f"logs/teamdates.log"),
    fileLevel=10,
    consoleLevel=20,
)
# %% constants
OTHERPREFIXMAP = {"tm": "opp", "opp": "tm"}

SUMMABLE_LIST = [
    "game",
    "poss",
    "mins",
    "win",
    "loss",
    "pts",
    "ast",
    "availor",
    "blk",
    "dr",
    "fga",
    "fga2",
    "fga3",
    "fgm",
    "fgm2",
    "fgm3",
    "fta",
    "ftm",
    "margin",
    "or",
    "pf",
    "stl",
    "to",
    "tr",
]
OUTPUT_METRIC_DICT = {
    "win_pct": {"ascending": False, "num_col": "win", "den_col": "game"},
    "pts_pposs": {"ascending": False, "num_col": "pts", "den_col": "poss"},
    "ast_pposs": {"ascending": False, "num_col": "ast", "den_col": "poss"},
    "blk_pposs": {"ascending": False, "num_col": "blk", "den_col": "poss"},
    "dr_pposs": {"ascending": False, "num_col": "dr", "den_col": "poss"},
    # "fga_pposs": {"ascending": False, "num_col": "fga", "den_col": "poss"},
    # "fga2_pposs": {"ascending": False, "num_col": "fga2", "den_col": "poss"},
    # "fga3_pposs": {"ascending": False, "num_col": "fga3", "den_col": "poss"},
    # "fgm_pposs": {"ascending": False, "num_col": "fgm", "den_col": "poss"},
    # "fgm2_pposs": {"ascending": False, "num_col": "fgm2", "den_col": "poss"},
    # "fgm3_pposs": {"ascending": False, "num_col": "fgm3", "den_col": "poss"},
    # "fta_pposs": {"ascending": False, "num_col": "fta", "den_col": "poss"},
    # "ftm_pposs": {"ascending": False, "num_col": "ftm", "den_col": "poss"},
    "margin_pposs": {"ascending": False, "num_col": "margin", "den_col": "poss"},
    "or_pposs": {"ascending": False, "num_col": "or", "den_col": "poss"},
    "pf_pposs": {"ascending": True, "num_col": "pf", "den_col": "poss"},
    "stl_pposs": {"ascending": False, "num_col": "stl", "den_col": "poss"},
    "to_pposs": {"ascending": True, "num_col": "to", "den_col": "poss"},
    "tr_pposs": {"ascending": False, "num_col": "tr", "den_col": "poss"},
    "ft_pct": {"ascending": False, "num_col": "ftm", "den_col": "fta"},
    "fg_pct": {"ascending": False, "num_col": "fgm", "den_col": "fga"},
    "fg2_pct": {"ascending": False, "num_col": "fgm2", "den_col": "fga2"},
    "fg3_pct": {"ascending": False, "num_col": "fgm3", "den_col": "fga3"},
    "astto_ratio": {"ascending": False, "num_col": "ast", "den_col": "to"},
    "or_pct": {"ascending": False, "num_col": "or", "den_col": "availor"},
}

# %% Create sql for injection
metric_cols = [
    f"{prefix}_{metric}" for prefix in ["tm", "opp"] for metric in SUMMABLE_LIST
]
gb_sql = ""
for m in metric_cols:
    gb_sql += f"sum({m}) as {m}, "
gb_sql = gb_sql[:-2]

# %% Function to loop through seasons
@log.timeFuncInfo
def get_teamdates(season: int):
    # Get games for given season, cross join games to all dates to
    q = f"""
    with seasonDates as (
        select
            season
            ,min(date) as minDate
            ,max(date) as maxDate
            from games
            where season = {season}
    )
    ,dates as (
        select distinct 
            s.season
            ,c.date
        from calendar c
        left join seasonDates s
        where c.date between minDate and maxDate
    )
    ,teamList as (
        select distinct season, tm_teamid
        from games
        where season = {season}
    )
    ,allDatesAllTeams as (
        select
            dates.season
            ,dates.date
            ,teamList.tm_teamid
        from dates
        cross join teamList
        on teamList.season = dates.season
    )
    select 
        adat.*
        ,t.teamname
        ,{gb_sql}
    from allDatesAllTeams adat
    left join games g
        on adat.season = g.season
        and adat.tm_teamid = g.tm_teamid
        and adat.date > g.date
    left join teams t
        on adat.tm_teamid = t.teamid
    group by adat.season, adat.date, adat.tm_teamid
    order by adat.season, adat.tm_teamid, adat.date
    """
    df = readSql(q, db="ncaa.db")

    # Get all games for filtering
    q = f"""
    select 
        t.teamname
        ,g.*
    from games g
    left join teams t
    on g.tm_teamid = t.teamid
    where season = {season}
    """
    games = readSql(q)

    log.info("Executed sql")

    # Get predate opponent stats
    total_rows = len(df)

    # Function to get sum of opponent state prior to the give row's date
    def opp_stats(row: pd.Series):
        "Removes the team's games from the data, and duplicates by number of games v opp"

        if (row.name + 1) % 10000 == 0:
            log.info(f"Progress in apply: {(row.name + 1) / total_rows:.2%}")

        # Get list of all opponents (includes duplicate values)
        opp_list = games.loc[
            (games["date"] < row["date"]) & (games["tm_teamid"] == row["tm_teamid"])
        ]["opp_teamid"].to_list()

        # Create DF of opponents-counts combinations, merge into all games by said opps
        opp_df = pd.DataFrame(Counter(opp_list).items(), columns=["tm_teamid", "n"])
        opp_predate_games = games.loc[
            (games["date"] < row["date"])
            & (games["tm_teamid"].isin(opp_list))
            & (games["opp_teamid"] != row["tm_teamid"])
        ].merge(opp_df, how="left", on="tm_teamid")
        # Duplicate games per number of times they were played, get stats
        opp_predate_stats = opp_predate_games.loc[
            opp_predate_games.index.repeat(opp_predate_games["n"])
        ][metric_cols].sum()

        return opp_predate_stats

    df[[f"opp_{x}" for x in metric_cols]] = df.apply(lambda row: opp_stats(row), axis=1)

    # Add output metrics to df
    # Pre-create columns for performance warning reasons
    output_metric_cols = [
        f"{t}{p}_{m}"
        for m in OUTPUT_METRIC_DICT.keys()
        for p in ["tm", "opp"]
        for t in ["", "oa_", "rnk_", "rnk_oa_"]
    ]

    df = pd.concat([df, pd.DataFrame(columns=output_metric_cols)], axis=1)

    output_metrics = []
    for metric, metric_details in OUTPUT_METRIC_DICT.items():
        log.debug(metric)
        for prefix in ["tm", "opp"]:
            output_metrics += [
                f"{prefix}_{metric}",
                f"oa_{prefix}_{metric}",
                f"rnk_{prefix}_{metric}",
                f"rnk_oa_{prefix}_{metric}",
            ]
            # Ascending metrics get flipped for the opponent's acumen at them
            calc_ascending = metric_details["ascending"]
            if prefix == "opp":
                calc_ascending = not calc_ascending

            # Get metric, OA metric
            df[f"{prefix}_{metric}"] = (
                df[f"{prefix}_{metric_details['num_col']}"]
                / df[f"{prefix}_{metric_details['den_col']}"]
            )
            df[f"oa_{prefix}_{metric}"] = df[f"{prefix}_{metric}"] - (
                df[f"opp_{OTHERPREFIXMAP[prefix]}_{metric_details['num_col']}"]
                / df[f"opp_{OTHERPREFIXMAP[prefix]}_{metric_details['den_col']}"]
            )

            # Round to 6 decimals
            df[f"{prefix}_{metric}"] = np.round(df[f"{prefix}_{metric}"], 6)
            df[f"oa_{prefix}_{metric}"] = np.round(df[f"oa_{prefix}_{metric}"], 6)

            # Rank
            df[f"rnk_{prefix}_{metric}"] = df.groupby(["date"])[
                f"{prefix}_{metric}"
            ].rank(ascending=calc_ascending, method="min")
            df[f"rnk_oa_{prefix}_{metric}"] = df.groupby(["date"])[
                f"oa_{prefix}_{metric}"
            ].rank(ascending=calc_ascending, method="min")

    # Trim to final columns
    out_first_cols = [
        "teamid",
        "game",
        "win",
        "loss",
        "poss",
        "mins",
    ]
    renames = {}
    for col in out_first_cols:
        renames[f"tm_{col}"] = col
    df.rename(columns=renames, inplace=True)

    out = df[["season", "date"] + out_first_cols + output_metric_cols]

    return out


# %%
if __name__ == "__main__":

    # Drop old table
    q = "drop table if exists teamdates"
    executeSql(q, "ncaa.db")

    # Loop through seasons in games
    for season in readSql("select distinct season from games")["season"]:
        log.info(f"{season} season")
        df = get_teamdates(season=season)

        dfToTable(
            df,
            table="teamdates",
            db="ncaa.db",
            ifExists="append",
            indexCols=["season", "date", "teamid"],
        )

# %%
