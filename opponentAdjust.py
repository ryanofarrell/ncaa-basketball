# %% Imports
from typing import Counter
from helpers import readSql
import pandas as pd
from fnmatch import fnmatch


# %% constants
SEASON = 2022
OTHERPREFIXMAP = {"tm": "opp", "opp": "tm"}

# %%
q = """
select
    t.teamname
    ,g.*
from games g
left join teams t
on g.tm_teamid = t.teamid
where t.teamname = 'Florida'
order by date
"""
t = readSql(q, db="ncaa.db")

# %% Create sql for injection
nonmetric_list = ["game", "poss", "mins", "rsgame", "win"]
metrics_dict = {
    "pts": True,
    "ast": True,
    "blk": True,
    "dr": True,
    "fga": True,
    "fga2": True,
    "fga3": True,
    "fgm": True,
    "fgm2": True,
    "fgm3": True,
    "fta": True,
    "ftm": True,
    "margin": True,
    "or": True,
    "pf": False,
    "stl": True,
    "to": False,
    "tr": True,
}
metric_list = list(metrics_dict.keys())

metric_cols = [
    f"{prefix}_{metric}" for prefix in ["tm", "opp"] for metric in metric_list
]
nonmetric_cols = [
    f"{prefix}_{nonmetric}" for prefix in ["tm", "opp"] for nonmetric in nonmetric_list
]
gb_sql = ""
for m in metric_cols + nonmetric_cols:
    gb_sql += f"sum({m}) as {m}, "
gb_sql = gb_sql[:-2]
# %%
q = f"""
with dates as (
    select distinct season, date
    from games
    where season = {SEASON}
)
,teamList as (
    select distinct season, tm_teamid
    from games
    where season = {SEASON}
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
where season = {SEASON}
"""
games = readSql(q)


# %%
total_rows = len(df)
# Function to get sum of opponent state prior to the give row's date
def opp_stats(row):
    "Removes the team's games from the data, and duplicates by number of games v opp"

    if (row.name + 1) % 500 == 0:
        print(f"{(row.name + 1) / total_rows:.2%}")

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
    ][metric_cols + nonmetric_cols].sum()

    return opp_predate_stats


df[[f"opp_{x}" for x in metric_cols + nonmetric_cols]] = df.apply(
    lambda row: opp_stats(row), axis=1
)


# normalize the team metrics
for m in metric_list:
    for prefix in ["tm", "opp"]:
        for norm in ["poss"]:
            if norm == m:
                continue
            df[f"{prefix}_{m}_p{norm}"] = df[f"{prefix}_{m}"] / df[f"{prefix}_{norm}"]
            del df[f"{prefix}_{m}"]
            # print(f"{prefix}_{m}_p{norm}")

# NORMALIZE OPPONENT metrics
for m in metric_list:
    for prefix in ["tm", "opp"]:
        for norm in ["poss"]:
            if norm == m:
                continue
            df[f"opp_{prefix}_{m}_p{norm}"] = (
                df[f"opp_{prefix}_{m}"] / df[f"opp_{prefix}_{norm}"]
            )
            del df[f"opp_{prefix}_{m}"]

# Get opponent-adjusted values
for m in metric_list:
    for prefix in ["tm", "opp"]:
        for norm in ["poss"]:
            if norm == m:
                continue

            df[f"oa_{prefix}_{m}_p{norm}"] = (
                df[f"{prefix}_{m}_p{norm}"]
                - df[f"opp_{OTHERPREFIXMAP[prefix]}_{m}_p{norm}"]
            )


# %% Save to database
# Remove extra columns
oa_cols = [x for x in df.columns if (fnmatch(x, "opp_tm_*") or fnmatch(x, "opp_opp_*"))]
opp_nonmetric_cols = [f"opp_{nonmetric}" for nonmetric in nonmetric_list]
for col in (
    [
        "teamname",
    ]
    + oa_cols
    + opp_nonmetric_cols
):
    del df[col]

# Rename
renames = {}
for col in nonmetric_list:
    renames[f"tm_{col}"] = col
df.rename(columns=renames, inplace=True)

# Add custom columns # TODO need to OA these in above process
for prefix in ["tm", "opp"]:
    df[f"{prefix}_ft_pct"] = df[f"{prefix}_ftm_pposs"] / df[f"{prefix}_fta_pposs"]
    df[f"{prefix}_fg_pct"] = df[f"{prefix}_fgm_pposs"] / df[f"{prefix}_fga_pposs"]
    df[f"{prefix}_fg2_pct"] = df[f"{prefix}_fgm2_pposs"] / df[f"{prefix}_fga2_pposs"]
    df[f"{prefix}_fg3_pct"] = df[f"{prefix}_fgm3_pposs"] / df[f"{prefix}_fga3_pposs"]

# %%
import plotly.express as px

px.line(df, x="date", y=["tm_margin_pposs", "oa_tm_margin_pposs"], color="teamid")
# %%
teamdata = games.loc[games["teamname"] == "Florida"].merge(
    df, how="left", on=["season", "date", "teamname", "tm_teamid"]
)
# %%
