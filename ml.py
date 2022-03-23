# %% Imports
import numpy as np
import pandas as pd
from helpers import readSql
from opponentAdjust import OUTPUT_METRIC_DICT
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy.cluster.hierarchy import dendrogram, linkage
from matplotlib import pyplot as plt
import plotly.express as px

# %% Prep SQL for team vs opp
sql_cols = [
    f"oa_{prefix}_{m}" for m in OUTPUT_METRIC_DICT.keys() for prefix in ["tm", "opp"]
]

tm1_sql_cols = [f"{x} as tm1_{x}" for x in sql_cols]
tm2_sql_cols = [f"{x} as tm2_{x}" for x in sql_cols]
# %%
q = f"""
select
    g.date
    ,g.season
    ,g.game_key
    ,g.tm_teamid
    ,g.opp_teamid
    ,g.tm_pts
    ,g.opp_pts
    ,g.tm_win
    ,td1.{', td1.'.join(tm1_sql_cols)}
    ,td2.{', td2.'.join(tm2_sql_cols)}
from games g
left join teamdates td1
    on g.date = td1.date
    and g.tm_teamid = td1.teamid
left join teamdates td2
    on g.date = td2.date
    and g.opp_teamid = td2.teamid
left join calendar c
    on c.date = g.date
where c.day_num >= 60
order by g.date, game_key, tm_teamid
-- limit 1000
"""
raw_td = readSql(q).dropna(how="any")

# %%
td = raw_td.copy()
X_cols = td.columns[8:]
td["rand"] = np.random.random(len(td))
td = (
    td.sort_values(by=["rand"])
    .drop_duplicates(subset=["date", "game_key"], keep="first")
    .drop(["rand"], axis=1)
)
# De-duplicate - apply random number to column, keep first
X = td[X_cols]
y = td["tm_win"]

#
# Scale
x_scale = StandardScaler().fit_transform(X)

# pca
pca = PCA(n_components=0.9, random_state=832828).fit(x_scale)

i = 1
cuml = 0
for ev in pca.explained_variance_ratio_:
    cuml += ev
    print(f"Expl Var % by {i}: {ev:.3%} (cuml: {cuml:.3%})")
    i += 1


x_scale_pca = pca.transform(x_scale)

# %% hierarchical
from sklearn.ensemble import RandomForestClassifier

rfc = RandomForestClassifier(
    n_estimators=5000,
    min_samples_split=5,
    max_features="sqrt",
    bootstrap=True,
    n_jobs=-1,
    max_samples=0.8,
    verbose=True,
    random_state=918375,
)

# Train-est split
train_mask = td["season"] != 2019
test_mask = td["season"] == 2019
X_train = x_scale_pca[train_mask]
X_test = x_scale_pca[test_mask]
y_train = y[train_mask]
y_test = y[test_mask]

rfc.fit(X_train, y_train)
# %%

td_test = td.loc[test_mask]
td_test = td_test[[x for x in td_test if x not in X_cols]]
td_test["margin"] = td_test["tm_pts"] - td_test["opp_pts"]
td_test["abs_margin"] = np.abs(td_test["margin"])
td_test["actl"] = y_test
td_test["pred"] = rfc.predict(X_test)
td_test["correct"] = 1 * (td_test["actl"] == td_test["pred"])
td_test["pred_prob"] = rfc.predict_proba(X_test)[:, 1]
td_test["certainty"] = np.abs(0.5 - td_test["pred_prob"]) * 2

# %%

px.histogram(td_test, x="certainty", y="correct", histfunc="avg", nbins=30)

# %%
