
# %% Imports
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import r2_score, roc_curve, roc_auc_score
from itertools import product

# Import to add project folder to sys path
import sys
utils_path = '/Users/Ryan/Documents/projects/ncaa-basketball/code/utils'
if utils_path not in sys.path:
    sys.path.append(utils_path)

from api import opponentAdjust
from db import get_db
from misc import getRelativeFp


# %% Declare parameters
metrics = [
    'PF', 'Margin',
    'FGM', 'FGA',
    'FG3M', 'FG3A',
    'FG2M', 'FG2A',
    'FTA', 'FTM',
    'Ast', 'ORB',
    'DRB', 'TRB',
    'TO', 'Stl',
    'Blk', 'Foul'
]
prefixes = ['Tm', 'Opp']
attrCols = [
    'Season',
    'TmName',
    'dataPriorTo',
    'isRegularSeason',
    ''
]


# %% Read in raw data
def getTd():
    """Reads in local csv of team-date data as of specified dates

    Returns:
        DataFrame: team-date data
    """
    firstYr = True
    for yr in range(2003, 2021):
        print(f"Reading in {yr}'s data")
        fp = getRelativeFp(__file__, f'../data/predate_games/{yr}.csv')
        if firstYr:
            raw = pd.read_csv(fp)
            firstYr = False
        else:
            raw = raw.append(pd.read_csv(fp))
    return raw


# %% Add additional columns to team-date data
def addColsToTd(_td, _metrics=metrics, _prefixes=prefixes):
    """Add additional columns to the team-date DataFrame

    Args:
        _td (DataFrame): Team-date DataFrame
        _metrics (list, optional): list of metrics to pass to opponent-
            adjusting function. Defaults to metrics.
        _prefixes (list, optional): list of prefixes to pass to opponent-
            adjusting function. Defaults to prefixes.

    Returns:
        DataFrame: Team-date data with additional columns
    """
    _td = opponentAdjust(
        _td,
        _prefixes,
        _metrics,
        includeOARankFields=False,
        includeNormRankFields=False,
        includeNormFields=True
    )
    _td['TmWinPct'] = _td['TmWin'] / _td['TmGame']
    _td['TmPossPerGame'] = _td['TmPoss'] / _td['TmGame']

    return _td


# %% Clean team-date data
def cleanTd(_td, minGames=10, _metrics=metrics, _prefixes=prefixes):
    """Removes extra columns, removes data prior to specified number of games,
    changes datetime data type

    Args:
        _td (DataFrame): team-date data
        minGames (int, optional): Minimum games played to keep data.
            Defaults to 10.
        _metrics (list, optional): Metrics to drop raw data of.
            Defaults to metrics.
        _prefixes (list, optional): Prefixes to drop raw data of.
            Defaults to prefixes.


    Returns:
        DataFrame: Cleaned data
    """
    # Make list of columns to ignore
    colsToDrop = [
        'OppGame',
        'GameOT'
    ]
    for metr in _metrics + ['Mins', 'Win', 'Poss']:
        for pref in _prefixes:
            colsToDrop.append(f'{pref}{metr}')
            colsToDrop.append(f'OppSum_{pref}{metr}')

    keptCols = [
        col for col in _td.columns
        if (col not in colsToDrop) & (col[:7] != 'OppSum_')
    ]

    _td = _td[keptCols]

    # Limit team-dates to only those with >= minGames
    _td = _td.loc[_td['TmGame'] >= minGames]

    # Change field to datetime
    _td['dataPriorTo'] = pd.to_datetime(_td['dataPriorTo'])

    return _td


# %% Get game data for analysis
def getGames():
    """Get 2003+ game data from database

    Returns:
        DataFrame: Game data
    """
    db = get_db()
    q = {'Season': {'$gte': 2003}}
    fields = {
        '_id': 0,
        'TmName': 1,
        'OppName': 1,
        'Season': 1,
        'GameDate': 1,
        'TmMargin': 1,
        'GameVegasLine': 1,
        'TmLoc': 1
    }
    raw = pd.DataFrame(
        list(
            db.games.find(
                q,
                fields
            )
        )
    )
    return raw


# %% De-dupe, remove outliers
def cleanGames(_games):
    """De-dupes game records, keeping only home teams and lower-name neutral games,
    rename a few columns, remove outlier lines, and change datatypes.

    Args:
        _games (DataFrame): game data

    Returns:
        DataFrame: Cleaned game data
    """
    # _games = _games.loc[(
    #     (_games['TmLoc'] == 'H') |
    #     ((_games['TmLoc'] == 'N') & (_games['TmName'] < _games['OppName']))
    # )]
    _games = _games.loc[np.abs(_games['GameVegasLine']) <= 50]
    _games.rename(inplace=True, columns={
        'GameDate': 'dataPriorTo'
    })
    _games['dataPriorTo'] = pd.to_datetime(_games['dataPriorTo'])
    _games['Tm_isHome'] = np.where(_games['TmLoc'] == 'H', 1, 0)

    return _games


# %% Add add'l columns to games
def addColsToGames(_games):
    # GameVegasLine is the expected Tm margin. Positive means vegas favored Tm.
    # If TmVegasMargin > 0: TmMargin > GameVegasLine: Team outperformed vegas - bet on team.
    # If TmVegasMargin < 0: TmMargin < GameVegasLine, team did worse than vegas expected - bet on Opp
    _games['tmVegasMargin'] = _games['TmMargin'] - _games['GameVegasLine']
    _games['tmAtsWinner'] = (_games['tmVegasMargin'] > 0) * 1

    return _games


# %% Merge in team-date data to game records
def addTdToGames(_td, _games):
    tdCols = ['TmName', 'Season', 'dataPriorTo']
    dfTm = _td.copy()
    dfTm.columns = [
        f'Tm_{x}' if x not in tdCols else x for x in dfTm.columns
    ]
    dfOpp = _td.copy()
    dfOpp.columns = [
        f'Opp_{x}' if x not in tdCols else x for x in dfOpp.columns
        ]
    dfOpp.rename(inplace=True, columns={'TmName': 'OppName'})

    _games = pd.merge(
        left=_games,
        right=dfTm,
        on=['TmName', 'Season', 'dataPriorTo'],
        how='inner'
    )
    _games = pd.merge(
        left=_games,
        right=dfOpp,
        on=['OppName', 'Season', 'dataPriorTo'],
        how='inner'
    )

    return _games


# %% Eval margin predictions
def evalModel(predMargin, actualMargin, methodName, verbose=False):
    print(f"{methodName}:")
    # Accuracy of margin
    R2 = r2_score(
        y_true=actualMargin,
        y_pred=predMargin
    )

    MAE = np.mean(np.abs(actualMargin - predMargin))

    # Correctly picking winner
    sumWins = np.sum(
        (predMargin * actualMargin > 0)*1
    )
    sumLosses = np.sum(
        (predMargin * actualMargin < 0)*1
    )
    sumTies = np.sum(
        (predMargin * actualMargin == 0)*1
    )
    if verbose:
        print(f"MAE: {MAE:.2f}")
        print(f"R^2: {R2:.4f}")
        print(f"Correct Winner Record: {sumWins:,.0f} - {sumLosses:,.0f} - {sumTies:,.0f}")
        print(f"Win Pct: {sumWins/len(predMargin):.3%}")
        print(f"Win Pct excluding pushes: {sumWins/(sumWins + sumLosses):.3%}")
        print('\n')

    return R2, MAE


# %% Create function for classification model 
def evalClassificationModel(predClass, actualClass, isPush, methodName, predProb, showCurve=False, verbose=False):
    print(f"{methodName}:")

    predWin = ((predClass == actualClass) & (isPush == 0))*1
    predLoss = ((predClass != actualClass) & (isPush == 0))*1
    predPush = (isPush == 1)*1

    w = np.sum(predWin)/(np.sum(predWin)+np.sum(predLoss))
    b_auc = roc_auc_score(actualClass, predClass)
    p_auc = roc_auc_score(actualClass, predProb)

    if verbose:
        print(f"Record: {np.sum(predWin)} - {np.sum(predLoss)} - {np.sum(predPush)}")
        print(f"Net wins: {np.sum(predWin) - np.sum(predLoss)}")
        print(f"Win Percent: {w:.2%}")
        print(f"Binary AUC: {b_auc:.2%}")
        print(f"Probability AUC: {p_auc:.2%}")

    if showCurve:
        fpr, tpr, thresholds = roc_curve(actualClass, predProb)
        fig = go.Figure(go.Scatter(x=fpr, y=tpr))
        fig.show()

    return w, b_auc, p_auc


# %% Main function
if __name__ == '__main__':
    # Get team-date data
    td = addColsToTd(getTd())
    td = cleanTd(td)

    # Get games data, add team details
    games = cleanGames(getGames())
    games = addColsToGames(games)
    gamesCols = games.columns
    games = addTdToGames(td, games)

    gamesX = games[[col for col in games.columns if col not in gamesCols]]
    gamesy = games[gamesCols]

    ####################################
    ## Predicting Game Margin
    ####################################
    # Set baseline: Vegas lines to predict margin
    evalModel(
        predMargin=gamesy['GameVegasLine'],
        actualMargin=gamesy['TmMargin'],
        methodName='Vegas',
        verbose=True
    )

    # Now, using Marginper40 diffs
    evalModel(
        predMargin=gamesX['Tm_TmMarginper40'] - gamesX['Opp_TmMarginper40'],
        actualMargin=gamesy['TmMargin'],
        methodName='Margin per 40 Difference',
        verbose=True
    )

    # OA Margin per 40
    evalModel(
        predMargin=gamesX['Tm_OA_TmMarginper40'] - gamesX['Opp_OA_TmMarginper40'],
        actualMargin=gamesy['TmMargin'],
        methodName='OA Margin per 40 Difference',
        verbose=True
    )

    # Now, adding a few different home court advantages to margin and OAM
    buffs = [a/2 for a in range(-5,16)]
    mov_r= []
    mov_m = []
    oa_r = []
    oa_m = []
    for b in buffs:
        # print(a/2)
        r, m = evalModel(
            predMargin=gamesX['Tm_TmMarginper40'] - gamesX['Opp_TmMarginper40'] + b* gamesy['Tm_isHome'],
            actualMargin=gamesy['TmMargin'],
            methodName=f'Margin per 40 Difference + {b}'
        )
        mov_r.append(r)
        mov_m.append(m)

        r, m = evalModel(
            predMargin=gamesX['Tm_OA_TmMarginper40'] - gamesX['Opp_OA_TmMarginper40'] + b* gamesy['Tm_isHome'],
            actualMargin=gamesy['TmMargin'],
            methodName=f'OA Margin per 40 Difference + {b}'
        )
        oa_r.append(r)
        oa_m.append(m)
    
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, subplot_titles=('R2 by Home Court Advantage', 'MAE by Home Court Advantage'))
    fig.add_trace(
        go.Scatter(x=buffs,y=mov_r, name='MoV - R2', legendgroup='MOV', line_color='steelblue'),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=buffs,y=oa_r, name='OA MoV - R2', legendgroup='OA', line_color='crimson'),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=buffs,y=mov_m, name='MoV - MAE', legendgroup='MOV', line_color='steelblue'),
        row=2, col=1
    )
    fig.add_trace(
        go.Scatter(x=buffs,y=oa_m, name='OA MoV - MAE', legendgroup='OA', line_color='crimson'),
        row=2, col=1
    )
    fig.add_trace(
        go.Scatter(x=buffs, y=[0.3504]*len(buffs), name='Vegas - R2', legendgroup='v', line_color='black'),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=buffs, y=[8.31]*len(buffs), name='Vegas - MAE', legendgroup='v', line_color='black'),
        row=2, col=1
    )
    fig.update_xaxes(
        title_text='Home Court Advantage Value',
        row=2
    )
    fig.update_yaxes(title_text='R2', row=1)
    fig.update_yaxes(title_text='MAE', row=2)

    fig.show()

    # Machine learning section
    rf = RandomForestRegressor(
        n_estimators=500,
        n_jobs=-1,
        max_depth=10,
        random_state=114492,
        min_samples_split=6,
        max_features='sqrt',
        bootstrap=True,
        oob_score=False,
        max_samples=0.8,
        # verbose=True
    )
    temp = pd.DataFrame()
    for s in sorted(gamesy['Season'].unique()):
        print(f"Season {s}")
        rf.fit(gamesX.loc[gamesy['Season'] != s], gamesy.loc[gamesy['Season'] != s, 'TmMargin'])
        # gb.predict(gamesX.loc[gamesy['Season'] == s)
        t = pd.DataFrame(data={
            'predMargin': rf.predict(gamesX.loc[gamesy['Season'] == s]),
            'actlMargin': gamesy.loc[gamesy['Season'] == s, 'TmMargin'],
            'season': s
        })
        temp = temp.append(t)
        evalModel(
            predMargin=t['predMargin'],
            actualMargin=t['actlMargin'],
            methodName=f'Season {s} RF Results',
        verbose=True
        )

    evalModel(
        predMargin=temp['predMargin'],
        actualMargin=temp['actlMargin'],
        methodName=f'Total RF Results',
        verbose=True
    )

    df = temp.groupby(['Season'])


    ##############
    ## ATS
    ##############
    # Now, adding a few different home court advantages to margin and OAM
    buffs = [a/2 for a in range(-50,50)]
    mov_w = []
    mov_bauc = []
    mov_pauc = []
    oa_w = []
    oa_bauc = []
    oa_pauc = []
    for b in buffs:
        # print(a/2)
        # Check versus vegas
        # vegasMinusModel (+) when vegas likes Tm better than model (model bet on Opp)
        # vegasMinusModel (-) when vegas likes Tm worse than model (model bet on Tm)
        predMargin = gamesX['Tm_TmMarginper40'] - gamesX['Opp_TmMarginper40'] + b* gamesy['Tm_isHome']
        vegasMinusModel = gamesy['GameVegasLine'] - predMargin
        
        w, bauc, pauc = evalClassificationModel(
            predClass=(vegasMinusModel <= 0) * 1,
            actualClass=gamesy['tmAtsWinner'],
            isPush=(gamesy['tmVegasMargin'] == 0)*1,
            predProb=vegasMinusModel,
            methodName=f'Margin per 40 Difference + {b}'
            # showCurve=True
        )
        mov_w.append(w)
        mov_bauc.append(bauc)
        mov_pauc.append(pauc)


        predMargin = gamesX['Tm_OA_TmMarginper40'] - gamesX['Opp_OA_TmMarginper40'] + b* gamesy['Tm_isHome']
        vegasMinusModel = gamesy['GameVegasLine'] - predMargin
        
        w, bauc, pauc = evalClassificationModel(
            predClass=(vegasMinusModel <= 0) * 1,
            actualClass=gamesy['tmAtsWinner'],
            isPush=(gamesy['tmVegasMargin'] == 0)*1,
            predProb=vegasMinusModel,
            methodName=f'Margin per 40 Difference + {b}'
            # showCurve=True
        )
        oa_w.append(w)
        oa_bauc.append(bauc)
        oa_pauc.append(pauc)
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, subplot_titles=('Win Pct by Home Court Advantage', 'Binary AUC by HCA', 'Probability AUC by HCA'))
    fig.add_trace(
        go.Scatter(x=buffs,y=mov_w, name='MoV - Win Pct', legendgroup='MOV', line_color='steelblue'),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=buffs,y=oa_w, name='OA MoV - Win Pct', legendgroup='OA', line_color='crimson'),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=buffs,y=mov_bauc, name='MoV - Binary AUC', legendgroup='MOV', line_color='steelblue'),
        row=2, col=1
    )
    fig.add_trace(
        go.Scatter(x=buffs,y=oa_bauc, name='OA MoV - Binary AUC', legendgroup='OA', line_color='crimson'),
        row=2, col=1
    )
    fig.add_trace(
        go.Scatter(x=buffs,y=mov_pauc, name='MoV - Probability AUC', legendgroup='MOV', line_color='steelblue'),
        row=3, col=1
    )
    fig.add_trace(
        go.Scatter(x=buffs,y=oa_pauc, name='OA MoV - Probability AUC', legendgroup='OA', line_color='crimson'),
        row=3, col=1
    )



    fig.update_xaxes(
        title_text='Home Court Advantage Value',
        row=3
    )
    fig.update_yaxes(title_text='Win Pct', row=1)
    fig.update_yaxes(title_text='AUC', row=2)
    fig.update_yaxes(title_text='AUC', row=3)

    fig.show()



    # Try Classification based on if Tm wins ATS
    gamesX['GameVegasLine'] = gamesy['GameVegasLine']
    # Train the winning parameters: 10 depth, 6 min samples
    rfc = RandomForestClassifier(
        n_estimators=500,
        criterion='gini',
        max_depth=5,
        min_samples_split=2,
        max_features='sqrt',
        bootstrap=True,
        # oob_score=True,
        n_jobs=-1,
        random_state=114492,
        # verbose=True,
        max_samples=.8
    )
    temp = pd.DataFrame()
    for s in sorted(gamesy['Season'].unique()):
        print(f"Season {s}")
        rfc.fit(gamesX.loc[gamesy['Season'] != s], gamesy.loc[gamesy['Season'] != s, 'tmAtsWinner'])
        # gb.predict(gamesX.loc[gamesy['Season'] == s)
        t = pd.DataFrame(data={
            'predClass': rfc.predict(gamesX.loc[gamesy['Season'] == s]),
            'predProb': rfc.predict_proba(gamesX.loc[gamesy['Season'] == s])[:,1],
            'actlClass': gamesy.loc[gamesy['Season'] == s, 'tmAtsWinner'],
            'isPush': ((gamesy.loc[gamesy['Season'] == s, 'tmVegasMargin'])==0)*1
        })
        temp = temp.append(t)
        evalClassificationModel(
            predClass=t['predClass'],
            actualClass=t['actlClass'],
            isPush=t['isPush'],
            predProb=t['predProb'],
            methodName=f'{s} Season Results', 
            # showCurve=True
        )
    evalClassificationModel(
        predClass=temp['predClass'],
        actualClass=temp['actlClass'],
        isPush=temp['isPush'],
        predProb=temp['predProb'],
        methodName=f'rfc All Season Results', 
        showCurve=True
    )


    # Gradient Boosting
    gb = GradientBoostingClassifier(
        n_estimators=20,
        max_depth=3,
        random_state=114492,
        verbose=True
    )
    temp = pd.DataFrame()
    for s in sorted(gamesy['Season'].unique()):
        print(f"Season {s}")
        gb.fit(gamesX.loc[gamesy['Season'] != s], gamesy.loc[gamesy['Season'] != s, 'tmAtsWinner'])
        # gb.predict(gamesX.loc[gamesy['Season'] == s)
        t = pd.DataFrame(data={
            'predClass': gb.predict(gamesX.loc[gamesy['Season'] == s]),
            'predProb': gb.predict_proba(gamesX.loc[gamesy['Season'] == s])[:,1],
            'actlClass': gamesy.loc[gamesy['Season'] == s, 'tmAtsWinner'],
            'isPush': ((gamesy.loc[gamesy['Season'] == s, 'tmVegasMargin'])==0)*1
        })
        temp = temp.append(t)
        evalClassificationModel(
            predClass=t['predClass'],
            actualClass=t['actlClass'],
            isPush=t['isPush'],
            predProb=t['predProb'],
            methodName=f'{s} Season Results', 
            # showCurve=True
        )
    evalClassificationModel(
        predClass=temp['predClass'],
        actualClass=temp['actlClass'],
        isPush=temp['isPush'],
        predProb=temp['predProb'],
        methodName=f'GB All Season Results', 
        showCurve=True,
        verbose=True
    )



    # print(f"RF Classifier Correct: {np.mean(gamesy['rfc_Correct']):.2%}")
    # games['rfc_correctCumSum'] = np.where(
    #     games['rfc_betOnTm'] == games['tmAtsWinner'],
    #     1,
    #     -1
    # )
    # games.sort_values(inplace=True, by=['dataPriorTo'])
    # games['modelDateCumSum'] = games['rfc_correctCumSum'].cumsum()
    # fig = px.line(games, x='dataPriorTo', y='modelDateCumSum')
    # fig.show()

    # gamesy['probRounded'] = np.round(rfc.oob_decision_function_[:,1],3)
    # temp2 = gamesy.groupby(['probRounded'])['rfc_correctCumSum'].agg({'rfc_correctCumSum': ['sum', 'count']}).reset_index()
    # temp2.columns = ['probRounded', 'rfc_correctCumSum', 'recordCnt']
    # fig = px.bar(temp2, x='probRounded', y='rfc_correctCumSum')
    # fig.update_layout(
    #     title_text=f"Performance of {predictionCol} by {col}"
    # )

    # gamesy['placeBet'] = (gamesy['probRounded']<= 0.5)*1
    # games['smartWinner'] = games['placeBet']*games['rfc_correctCumSum']
    # print(f"Smart logic correct: {np.sum(games['smartWinner']):.0f}, win pct = {np.sum(games['smartWinner']==1)/np.sum(games['placeBet']):.2%}")

    # temp = games[['tmAtsWinner', 'rfc_tmAtsWinner_1','rfc_betOnTm']]




# %%
gb = GradientBoostingClassifier(
    n_estimators=150,
    max_depth=3,
    random_state=114492,
    verbose=True
)
gb.fit(gamesX, games['tmAtsWinner'])

gamesy['rfc_betOnTm'] = gb.predict(gamesX)
gamesy['rfc_Win'] = ((gamesy['rfc_betOnTm'] == gamesy['tmAtsWinner']) & (gamesy['tmVegasMargin'] != 0))*1
gamesy['rfc_Loss'] = ((gamesy['rfc_betOnTm'] != gamesy['tmAtsWinner']) & (gamesy['tmVegasMargin'] != 0))*1
gamesy['rfc_Push'] = (gamesy['tmVegasMargin'] == 0)*1
print(f"Record: {np.sum(gamesy['rfc_Win'])} - {np.sum(gamesy['rfc_Loss'])} - {np.sum(gamesy['rfc_Push'])}")
print(f"Net wins: {np.sum(gamesy['rfc_Win']) - np.sum(gamesy['rfc_Loss'])}")
print(f"Win Percent: {np.sum(gamesy['rfc_Win'])/(np.sum(gamesy['rfc_Win'])+np.sum(gamesy['rfc_Loss'])):.2%}")


# %%
