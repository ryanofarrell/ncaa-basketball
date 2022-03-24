# %% Imports
import numpy as np
from helpers import readSql
import plotly.express as px

# %%
df = readSql("select * from predictions")
df["correct"] = 1 * (df["tm_win"] == df["pred_win"])
df["certainty"] = np.abs(df["pred_win_prob"] - 0.5) * 2
df["error"] = df["tm_margin"] - df["pred_margin"]
df["abs_error"] = np.abs(df["error"])
# %%
px.histogram(df, x="pred_win_prob", y="tm_margin", histfunc="avg", nbins=25)

# %%
px.scatter(df, x="pred_margin", y="tm_margin")
# %%
season_results = (
    df.groupby("season")
    .agg({"correct": "mean", "abs_error": ["mean", "median"]})
    .reset_index()
)
season_results.columns = [f"{x}_{y}" for x, y in season_results.columns]
px.line(season_results, x="season_", y=["abs_error_mean", "abs_error_median"])
# %%
