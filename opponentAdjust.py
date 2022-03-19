# %% Imports
from helpers import readSql
import pandas as pd


# %% constants
SEASON = 2022

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
norm_list = [
    "game",
    "poss",
    "mins",
    "rsgame",
]
metric_list = [
    "pts",
    "ast",
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
    "win",
]

metric_cols = [
    f"{prefix}_{metric}" for prefix in ["tm", "opp"] for metric in metric_list
]
norm_cols = [f"{prefix}_{norm}" for prefix in ["tm", "opp"] for norm in norm_list]
gb_sql = ""
for m in metric_cols + norm_cols:
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
# %%
# for m in metric_list:
#     for prefix in ["tm", "opp"]:
#         for norm in ["poss"]:
#             if norm == m:
#                 continue
#             df[f"{prefix}_{m}_p{norm}"] = df[f"{prefix}_{m}"] / df[f"{prefix}_{norm}"]
#             del df[f"{prefix}_{m}"]
#             # print(f"{prefix}_{m}_p{norm}")

# %%
small = df.iloc[:10, :].reset_index(drop=True)
# Get all games for filtering
q = f"""
select *
from games
where season = {SEASON}
"""
games = readSql(q)

# Function to get sum of opponent state prior to the give row's date
def opp_stats(row):
    opp_list = games.loc[
        (games["date"] < row["date"]) & (games["tm_teamid"] == row["tm_teamid"])
    ]["opp_teamid"].to_list()
    opp_predate_stats = games.loc[
        (games["date"] < row["date"]) & (games["tm_teamid"].isin(opp_list))
    ][metric_cols + norm_cols].sum()
    return opp_predate_stats


small[[f"opp_{x}" for x in metric_cols + norm_cols]] = small.apply(
    lambda row: opp_stats(row), axis=1
)

# Rename all team stats to tm_<col> e.g. tm_tm_pts
# renames = {}
# for m in metric_cols + norm_cols:
#     renames[m] = f"tm_{m}"
# small.rename(columns=renames, inplace=True)

# Opponent adjust

# %%
for m in metric_list:
    for prefix in ["tm", "opp"]:
        for norm in ["poss"]:
            if norm == m:
                continue
            small[f"{prefix}_{m}_p{norm}"] = (
                small[f"{prefix}_{m}"] / small[f"{prefix}_{norm}"]
            )
            del small[f"{prefix}_{m}"]
            # print(f"{prefix}_{m}_p{norm}")


# %%
