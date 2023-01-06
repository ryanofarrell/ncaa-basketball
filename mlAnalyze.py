# %% Imports
import pandas as pd
import numpy as np
from helpers import qdf, readSql
import plotly.express as px
import plotly.graph_objects as go


# %% Get and visualize regression models
q = """
select *
from predictions p
where substr(id, 3, 1) == 'r'
"""
regs = readSql(q)


# Add some columns
regs["model_line"] = -regs["pred_margin"]
regs["veg-model"] = regs["tm_closeline"] - regs["model_line"]
regs["tm_val_ats"] = np.where(
    regs["tm_margin_ats"] > 0, 1, np.where(regs["tm_margin_ats"] < 0, -1, 0)
)
regs["model_error"] = regs["tm_margin"] - regs["pred_margin"]
regs["tm_abs_margin_ats"] = np.abs(regs["tm_margin_ats"])
regs["abs_model_error"] = np.abs(regs["model_error"])
regs["model_win"] = np.where(
    regs["veg-model"] >= 0, regs["tm_val_ats"], -1 * regs["tm_val_ats"]
)

regs["ats_pred_val"] = np.where(
    regs["pred_margin_ats"] >= 0, regs["tm_val_ats"], -1 * regs["tm_val_ats"]
)

# %%
agg = qdf(
    regs,
    """
select
    id
    ,self.season
    ,avg(abs_model_error) as mean_error
    ,sum(model_win) as sum_model_wins
    ,sum(ats_pred_val) as ats_pred_val
from self
group by id, self.season
union all

select 
    'vegas' as id
    ,season
    ,avg(tm_abs_margin_ats) as mean_error 
    ,0 as sum_model_wins
    ,0 as ats_pred_val
from self
group by season
order by id, self.season


""",
)
# Average error
fig = px.line(agg, x="season", y="mean_error", color="id")
fig.show()
# model wins
fig = px.line(agg, x="season", y="ats_pred_val", color="id")
fig.show()


# %% ats predictions
df = regs.loc[regs["id"] == "gbr_1000_slowlearn"].reset_index(drop=True)
# df = regs.loc[regs["id"] == "rfr_10000"].reset_index(drop=True)
px.scatter(df, x="pred_margin_ats", y="tm_margin_ats")
fig = px.histogram(df, x="pred_margin_ats", y="ats_pred_val", histfunc="avg", nbins=40)
fig.show()

fig = px.line(
    df.groupby(["season"])["ats_pred_val"].sum().reset_index(),
    x="season",
    y="ats_pred_val",
)
fig.show()

# %% 0.5 or higher
out = pd.DataFrame()
for i in range(30):
    thresh = (i + 1) / 10
    agg = (
        df.loc[~df["pred_margin_ats"].between(-thresh, thresh)]
        .groupby(["season"])["ats_pred_val"]
        .agg(["sum", "mean", "count"])
        .reset_index()
    )
    agg["thresh"] = thresh
    out = pd.concat([out, agg])

px.line(out, x="season", y="mean", color="thresh")
px.line(out, x="season", y="sum", color="thresh")
px.line(out, x="season", y="count", color="thresh")

# %% how frequently to predictions not match?
selfmerge = (
    df.groupby(["season", "game_key", "date"])["pred_margin_ats"]
    .apply(list)
    .reset_index()
)
selfmerge["one_side"] = selfmerge["pred_margin_ats"].apply(lambda x: x[0])
selfmerge["other_side"] = selfmerge["pred_margin_ats"].apply(
    lambda x: x[1] if len(x) == 2 else np.nan
)
selfmerge["same_side"] = selfmerge["one_side"] * selfmerge["other_side"] <= 0

selfmerge["same_side"].mean()

# %% Given thresh, look at actions
THRESH = 0.8
dfAct = df.loc[~df["pred_margin_ats"].between(-THRESH, THRESH)].reset_index()
dfAct = dfAct.merge(
    dfAct.groupby(["season", "game_key", "date"])["pred_margin_ats"]
    .apply(list)
    .reset_index()
    .rename(columns={"pred_margin_ats": "bothActions"}),
    on=["season", "game_key", "date"],
)
dfAct = dfAct.loc[dfAct["bothActions"].apply(lambda x: len(x) == 2)].reset_index(
    drop=True
)
agg = dfAct.groupby(["season", "date"])["ats_pred_val"].agg(["sum", "mean", "count"])
agg /= 2
agg.reset_index(inplace=True)
agg["cumsum"] = agg.groupby("season")["sum"].cumsum()

fig = px.line(agg, x="date", y=["count", "cumsum"], color="season")
fig.show()
# %% Select a config
q = "select distinct config from predictions"
cfg = readSql(q)["config"].to_list()
print("Please select a config option...")
for idx, val in enumerate(cfg):
    print(f"{idx}   |  {val}")
selected = int(input("Enter a num"))
sel_cfg = cfg[selected]

# %%
q = f"""
select 
    p.*
    ,v.tm_closeline
    ,v.tm_ml
    ,v.opp_ml
from predictions p
inner join vegas_lines v
    on p.season = v.season
    and p.date = v.date
    and p.game_key = v.game_key
    and p.tm_teamid = v.tm_teamid
where tm_closeline is not null
and config = "{sel_cfg}"

"""
df = readSql(q)
df["model_line"] = -df["pred_margin"]
df["veg-model"] = df["tm_closeline"] - df["model_line"]
df["correct"] = 1 * (df["tm_win"] == df["pred_win"])
df["margin_ats"] = df["tm_margin"] + df["tm_closeline"]
df["tm_win_ats"] = np.where(
    df["margin_ats"] > 0, 1, np.where(df["margin_ats"] < 0, -1, 0)
)
df["certainty"] = np.abs(df["pred_win_prob"] - 0.5) * 2
df["model_error"] = df["tm_margin"] - df["pred_margin"]
df["veg_error"] = df["margin_ats"]
df["abs_model_error"] = np.abs(df["model_error"])
df["abs_veg_error"] = np.abs(df["veg_error"])
# veg-model is 0 when vigas and model both agree on line
# It is > 0 when model thinks better of tm than vegas does, expect team to win ATS
# It is < 0 when model thinks less of tm than vegas does - expect team to lose ATS
df["model_win"] = np.where(
    df["veg-model"] >= 0, df["tm_win_ats"], -1 * df["tm_win_ats"]
)
df["model_win_counts"] = np.where(df["model_win"] == 1, 1, 0)
# %%
fig = px.histogram(df, x="veg-model", y="model_win", histfunc="sum", nbins=20)
fig.show()
fig = px.histogram(df, x="veg-model", y="model_win_counts", histfunc="avg", nbins=20)
fig.show()
# %%
season_results = (
    df.groupby("season")
    .agg(
        {
            "correct": "mean",
            "abs_model_error": ["mean", "median"],
            "abs_veg_error": ["mean", "median"],
            "model_win": ["sum", "count"],
        }
    )
    .reset_index()
)
season_results.columns = [f"{x}_{y}" for x, y in season_results.columns]
fig = px.line(
    season_results, x="season_", y=["abs_model_error_mean", "abs_veg_error_mean"]
)
fig.show()
fig = px.line(season_results, x="season_", y=["model_win_sum", "model_win_count"])
fig.show()
# %%
