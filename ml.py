# %% Imports
import numpy as np
import pandas as pd
from helpers import dfToTable, executeSql, get_unique_permutations, readSql
from opponentAdjust import OUTPUT_METRIC_DICT
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import plotly.express as px
from sklearn.ensemble import (
    GradientBoostingClassifier,
    RandomForestClassifier,
    RandomForestRegressor,
    GradientBoostingRegressor,
)

RANDOMSEED = 837592
# %% Prep SQL for team vs opp
def get_raw_data():
    sql_cols = [
        f"oa_{prefix}_{m}"
        for m in OUTPUT_METRIC_DICT.keys()
        for prefix in ["tm", "opp"]
    ]

    tm1_sql_cols = [f"{x} as tm1_{x}" for x in sql_cols]
    tm2_sql_cols = [f"{x} as tm2_{x}" for x in sql_cols]
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
        ,v.tm_closeline
        ,v.tm_ml
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
    left join vegas_lines v
        on g.season = v.season
        and g.date = v.date
        and g.game_key = v.game_key
        and g.tm_teamid = v.tm_teamid
    where c.day_num >= 60
    order by g.date, g.game_key, g.tm_teamid
    -- limit 1000
    """
    raw_td = readSql(q)
    raw_td.dropna(
        how="any",
        subset=[col for col in raw_td.columns if col[:3] in ["tm1", "tm2"]],
        inplace=True,
    )
    return raw_td


# %%
def prompt_remake_predictions_table():

    if input("Drop/remake old table? y/(n): ").lower() != "y":
        return

    print(f"Dropping old table")
    executeSql("drop table if exists predictions")
    q = f"""
    create table predictions (
        id string not null,
        config string not null,
        season integer not null,
        date string not null,
        game_key string not null,
        tm_teamid integer not null,
        opp_teamid integer not null,
        tm_closeline real not null,
        tm_ml real not null,
        tm_pts integer not null,
        opp_pts integer not null,
        tm_margin integer not null,
        tm_margin_ats integer not null,
        tm_win integer not null,
        tm_win_ats integer not null,
        pred_win_prob real not null,
        pred_win_ats_prob real not null,
        pred_margin real not null,
        pred_margin_ats real not null,
        primary key (id, config, date asc, game_key asc, tm_teamid asc)
    )
    """
    executeSql(q)
    for p in get_unique_permutations(["id", "config", "date", "game_key", "tm_teamid"]):
        executeSql(
            f"CREATE INDEX predictions_{'_'.join(p)} on predictions ({', '.join(p)})"
        )


# %%
if __name__ == "__main__":
    # Get raw data
    td = get_raw_data()

    # Prep some columns
    X_cols = [col for col in td.columns if col[:3] in ["tm1", "tm2"]]
    td["tm_margin"] = td["tm_pts"] - td["opp_pts"]
    td["tm_margin_ats"] = td["tm_pts"] - td["opp_pts"] + td["tm_closeline"]
    td["tm_win_ats"] = td["tm_margin_ats"] > 0
    # Only include data with a line to evertually evaluate the outcomes
    td["tm_ml"] = td["tm_ml"].astype(int, errors="ignore")
    td = td.loc[~(pd.isna(td["tm_closeline"]) | pd.isna(td["tm_ml"]))].reset_index(
        drop=True
    )

    # Set up X matrices
    X = td[X_cols]
    X_INCL_LINE = td[X_cols + ["tm_closeline"]]
    assert len(X.columns) == 4 * len(OUTPUT_METRIC_DICT), "INCORRECT X COLS"

    # Scale
    x_scale = StandardScaler().fit_transform(X)
    x_scale_incl_line = StandardScaler().fit_transform(X_INCL_LINE)

    # pca
    x_scale_pca = PCA(n_components=0.99, random_state=832828).fit_transform(x_scale)
    print(
        f"Dropped num variables by {1 - x_scale_pca.shape[1] / X.shape[1]:.2%} via PCA"
    )
    x_scale_pca_incl_line = PCA(n_components=0.99, random_state=832828).fit_transform(
        x_scale_incl_line
    )
    print(
        f"Dropped num variables by {1 - x_scale_pca.shape[1] / X_INCL_LINE.shape[1]:.2%} via PCA"
    )

    # Set up models
    reg_models = {
        "rfr_10000": RandomForestRegressor(
            n_estimators=10000,
            min_samples_split=5,
            max_features="sqrt",
            max_samples=0.8,
            random_state=RANDOMSEED,
            verbose=True,
            n_jobs=-1,
        ),
        "rfr_5000": RandomForestRegressor(
            n_estimators=5000,
            min_samples_split=5,
            max_features="sqrt",
            max_samples=0.8,
            random_state=RANDOMSEED,
            verbose=True,
            n_jobs=-1,
        ),
        "rfr_1000_verysubsampled": RandomForestRegressor(
            n_estimators=1000,
            min_samples_split=5,
            max_features="sqrt",
            max_samples=0.2,
            random_state=RANDOMSEED,
            verbose=True,
            n_jobs=-1,
        ),
        "rfr_1000_shallow": RandomForestRegressor(
            n_estimators=1000,
            max_depth=5,
            max_features="sqrt",
            max_samples=0.8,
            random_state=RANDOMSEED,
            verbose=True,
            n_jobs=-1,
        ),
        "gbr_5000": GradientBoostingRegressor(
            n_estimators=5000,
            learning_rate=0.1,
            subsample=0.8,
            min_samples_split=5,
            max_features="sqrt",
            random_state=RANDOMSEED,
            verbose=True,
        ),
        "gbr_2500": GradientBoostingRegressor(
            n_estimators=2500,
            learning_rate=0.1,
            subsample=0.8,
            min_samples_split=5,
            max_features="sqrt",
            random_state=RANDOMSEED,
            verbose=True,
        ),
        "gbr_1000_slowlearn": GradientBoostingRegressor(
            n_estimators=1000,
            learning_rate=0.01,
            subsample=0.8,
            min_samples_split=5,
            max_features="sqrt",
            random_state=RANDOMSEED,
            verbose=True,
        ),
        "gbr_1000_shallow": GradientBoostingRegressor(
            n_estimators=1000,
            learning_rate=0.1,
            subsample=0.8,
            max_depth=5,
            max_features="sqrt",
            random_state=RANDOMSEED,
            verbose=True,
        ),
    }
    # Set up models
    class_models = {
        "rfc_10000": RandomForestClassifier(
            n_estimators=10000,
            min_samples_split=5,
            max_features="sqrt",
            max_samples=0.8,
            random_state=RANDOMSEED,
            verbose=True,
            n_jobs=-1,
        ),
        "rfc_5000": RandomForestClassifier(
            n_estimators=5000,
            min_samples_split=5,
            max_features="sqrt",
            max_samples=0.8,
            random_state=RANDOMSEED,
            verbose=True,
            n_jobs=-1,
        ),
        "rfc_1000_verysubsampled": RandomForestClassifier(
            n_estimators=1000,
            min_samples_split=5,
            max_features="sqrt",
            max_samples=0.2,
            random_state=RANDOMSEED,
            verbose=True,
            n_jobs=-1,
        ),
        "rfc_1000_shallow": RandomForestClassifier(
            n_estimators=1000,
            max_depth=5,
            max_features="sqrt",
            max_samples=0.8,
            random_state=RANDOMSEED,
            verbose=True,
            n_jobs=-1,
        ),
        "gbc_5000": GradientBoostingClassifier(
            n_estimators=5000,
            learning_rate=0.1,
            subsample=0.8,
            min_samples_split=5,
            max_features="sqrt",
            random_state=RANDOMSEED,
            verbose=True,
        ),
        "gbc_2500": GradientBoostingClassifier(
            n_estimators=2500,
            learning_rate=0.1,
            subsample=0.8,
            min_samples_split=5,
            max_features="sqrt",
            random_state=RANDOMSEED,
            verbose=True,
        ),
        "gbc_1000_slowlearn": GradientBoostingClassifier(
            n_estimators=1000,
            learning_rate=0.01,
            subsample=0.8,
            min_samples_split=5,
            max_features="sqrt",
            random_state=RANDOMSEED,
            verbose=True,
        ),
        "gbc_1000_shallow": GradientBoostingClassifier(
            n_estimators=1000,
            learning_rate=0.1,
            subsample=0.8,
            max_depth=5,
            max_features="sqrt",
            random_state=RANDOMSEED,
            verbose=True,
        ),
    }

    # Remake predictions table if yes
    prompt_remake_predictions_table()

    all_years = sorted(td["season"].unique().tolist())
    for test_year in all_years:
        print(f"Testing {test_year} season")

        # Train-test split
        train_mask = td["season"] != test_year
        test_mask = td["season"] == test_year
        X_train = x_scale_pca[train_mask]
        X_test = x_scale_pca[test_mask]
        X_train_incl_line = x_scale_pca_incl_line[train_mask]
        X_test_incl_line = x_scale_pca_incl_line[test_mask]

        for id, model in reg_models.items():
            already_done = (
                readSql("Select distinct season, id from predictions")
                .to_numpy()
                .tolist()
            )
            if [test_year, id] in already_done:
                print(f"Skipping {id} for {test_year}")
                continue
            # Set up output
            td_test = td.loc[test_mask][[x for x in td.columns if x not in X_cols]]
            td_test["id"] = id
            td_test["config"] = str(model.get_params())

            # Get margin prediction
            y = td["tm_margin"]
            y_train = y[train_mask]
            y_test = y[test_mask]
            model.fit(X_train, y_train)
            td_test["pred_margin"] = model.predict(X_test)

            # Get ATS margin
            y = td["tm_margin_ats"]
            y_train = y[train_mask]
            y_test = y[test_mask]
            model.fit(X_train_incl_line, y_train)
            td_test["pred_margin_ats"] = model.predict(X_test_incl_line)

            td_test["pred_win_prob"] = 0
            td_test["pred_win_ats_prob"] = 0

            dfToTable(td_test, "predictions", "ncaa.db", "append")

        for id, model in class_models.items():
            already_done = (
                readSql("Select distinct season, id from predictions")
                .to_numpy()
                .tolist()
            )
            if [test_year, id] in already_done:
                print(f"Skipping {id} for {test_year}")
                continue

            # Set up output
            td_test = td.loc[test_mask][[x for x in td.columns if x not in X_cols]]
            td_test["id"] = id
            td_test["config"] = str(model.get_params())

            # Get win prediction
            y = td["tm_win"]
            y_train = y[train_mask]
            y_test = y[test_mask]
            model.fit(X_train, y_train)
            td_test["pred_win_prob"] = model.predict_proba(X_test)[:, 1]

            # Get ATS win margin
            y = td["tm_win_ats"]
            y_train = y[train_mask]
            y_test = y[test_mask]
            model.fit(X_train_incl_line, y_train)
            td_test["pred_win_ats_prob"] = model.predict_proba(X_test_incl_line)[:, 1]

            td_test["pred_margin"] = 0
            td_test["pred_margin_ats"] = 0

            dfToTable(td_test, "predictions", "ncaa.db", "append")
