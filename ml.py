# %% Imports
import numpy as np
import pandas as pd
from helpers import dfToTable, executeSql, get_unique_permutations, readSql
from opponentAdjust import OUTPUT_METRIC_DICT
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import plotly.express as px
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor


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

td["tm_margin"] = td["tm_pts"] - td["opp_pts"]

# De-duplicate - apply random number to column, keep first
td["rand"] = np.random.random(len(td))
td = (
    td.sort_values(by=["rand"])
    .drop_duplicates(subset=["date", "game_key"], keep="first")
    .drop(["rand"], axis=1)
)
assert len(td) == 0.5 * len(raw_td), "Dropped != half games"

X = td[X_cols]
assert len(X.columns) == 4 * len(OUTPUT_METRIC_DICT), "INCORRECT X COLS"

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
print(f"Dropped num variables by {1 - x_scale_pca.shape[1] / X.shape[1]:.2%} via PCA")

# %% Loop through years and train
all_years = sorted(td["season"].unique().tolist())

if input("Drop/remake old table? y/(n): ") == "y":
    print(f"Dropping old table")
    executeSql("drop table if exists predictions")
    q = f"""
    create table predictions (
        config string not null,
        season integer not null,
        date string not null,
        game_key string not null,
        tm_teamid integer not null,
        opp_teamid integer not null,
        tm_pts integer not null,
        opp_pts integer not null,
        tm_margin integer not null,
        tm_win integer not null,
        pred_win integer not null,
        pred_win_prob real not null,
        pred_margin real not null,
        primary key (config, date asc, game_key asc)
    )
    """
    executeSql(q)
    for p in get_unique_permutations(["config", "date", "game_key"]):
        executeSql(
            f"CREATE INDEX predictions_{'_'.join(p)} on predictions ({', '.join(p)})"
        )


config = {
    "model_type": "regression",
    "y_val": "tm_margin",
    "n_samples": 15000,
    "min_split": 5,
    "max_feat": "sqrt",
    "max_samp": 0.8,
}
y = td[config["y_val"]]
for test_year in all_years:
    print(f"Testing {test_year} season")
    if config["model_type"] == "regression":
        rf_class = RandomForestRegressor
    elif config["model_type"] == "classification":
        rf_class = RandomForestClassifier
    else:
        raise ValueError(f"Unconfigd model_type: {config['model_type']}")

    rf = rf_class(
        n_estimators=config["n_samples"],
        min_samples_split=config["min_split"],
        max_features=config["max_feat"],
        bootstrap=True,
        n_jobs=-1,
        max_samples=config["max_samp"],
        verbose=True,
        random_state=918375,
    )

    # Train-est split
    train_mask = td["season"] != test_year
    test_mask = td["season"] == test_year
    X_train = x_scale_pca[train_mask]
    X_test = x_scale_pca[test_mask]
    y_train = y[train_mask]
    y_test = y[test_mask]

    rf.fit(X_train, y_train)

    # Save output
    td_test = td.loc[test_mask][[x for x in td.columns if x not in X_cols]]
    td_test = td_test[[x for x in td_test if x not in X_cols]]
    td_test["config"] = str(list(config.items()))[1:-1]
    if config["model_type"] == "classification":
        td_test["pred_win"] = rf.predict(X_test)
        td_test["pred_margin"] = 0
        td_test["pred_win_prob"] = rf.predict_proba(X_test)[:, 1]
    else:  # Classification
        td_test["pred_margin"] = rf.predict(X_test)
        td_test["pred_win_prob"] = 0
        td_test["pred_win"] = 1 * (td_test["pred_margin"] > 0)

    dfToTable(td_test, "predictions", "ncaa.db", "append")
