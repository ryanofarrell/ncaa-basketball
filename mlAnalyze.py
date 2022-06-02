# %% Imports
import numpy as np
from helpers import readSql
import plotly.express as px


# %% Get and visualize regression models
q = """
select *
from predictions p
where substr(id, 3, 1) == 'r'
"""
regs = readSql(q)

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
df["model_win"] = np.where(df["veg-model"] >= 0, df["tm_win_ats"], -df["tm_win_ats"])
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
