
#%% Imports
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler


#%% Define Functions

def get_PBP_data(season):
    df = pd.read_csv('../data/PlayByPlay_'+str(season)+'/Events_'+str(season)+'.csv')
    return df

def get_player_data(season):
    df = pd.read_csv('../data/PlayByPlay_'+str(season)+'/Players_'+str(season)+'.csv')
    return df

def get_team_data(season):
    df = pd.read_csv('../data/Stage2DataFiles/Teams.csv')
    return df

def createSeasonDf(season):
    # Get PBP data
    seasonpbp = get_PBP_data(season)
    players = get_player_data(season)
    teams = get_team_data(season)

    # Add in margin and time remaining
    seasonpbp['SecondsRemaining'] = 2400-seasonpbp['ElapsedSeconds']

    seasonpbp = pd.merge(
        left=seasonpbp,
        left_on=['Season', 'EventPlayerID'],
        right=players[['Season', 'TeamID', 'PlayerID']],
        right_on=['Season', 'PlayerID']
    )

    seasonpbp['WhichTeam'] = np.where(
        seasonpbp['TeamID'] == seasonpbp['WTeamID'],
        'WTeam',
        'LTeam'
    )

    # Determine play-specific points added for each team
    seasonpbp['PlayWPoints'] = pd.to_numeric(np.where(
        ((seasonpbp['EventType'].str[0:4] == 'made') & (seasonpbp['WhichTeam'] == 'WTeam')),
        seasonpbp['EventType'].str[4],
        '0'
    ))
    seasonpbp['PlayLPoints'] = pd.to_numeric(np.where(
        ((seasonpbp['EventType'].str[0:4] == 'made') & (seasonpbp['WhichTeam'] == 'LTeam')),
        seasonpbp['EventType'].str[4],
        '0'
    ))

    # Add a Game Key
    seasonpbp['GameKey'] = seasonpbp['DayNum'].astype(str) \
        + '|' + seasonpbp['WTeamID'].astype(str) \
        + '|' + seasonpbp['LTeamID'].astype(str)


    # Determine running total score in games
    seasonpbp = seasonpbp.sort_values(
        by=['GameKey', 'SecondsRemaining'],
        ascending=[True, False]
    )
    seasonpbp['WPointsRunning'] = seasonpbp.groupby('GameKey')['PlayWPoints'].transform(pd.Series.cumsum)
    seasonpbp['LPointsRunning'] = seasonpbp.groupby('GameKey')['PlayLPoints'].transform(pd.Series.cumsum)

    # Add clutch indicator
    seasonpbp['CurrentMarginAbs'] = abs(seasonpbp['WPointsRunning'] - seasonpbp['LPointsRunning'])

    seasonpbp['IsClutchMoment'] = np.where(
        (
            ((seasonpbp['SecondsRemaining'] < 300) & (seasonpbp['CurrentMarginAbs'] <= 10)) |
            ((seasonpbp['SecondsRemaining'] < 600) & (seasonpbp['CurrentMarginAbs'] <= 15))
        ),
        1,
        0)

    # Add records counter
    seasonpbp['Records'] = 1

    # Aggregate PBP data to each player's totals
    stats = seasonpbp[
        ['EventPlayerID', 'TeamID', 'EventType', 'Records']
    ].groupby(
        by=['EventPlayerID', 'TeamID', 'EventType']
    ).count().unstack('EventType').fillna(0)

    stats.columns = stats.columns.droplevel(0)
    stats = stats.rename( columns={
        'assist': 'Assist',
        'block': 'Block',
        'foul_pers': 'FoulPers', 
        'foul_tech': 'FoulTech', 
        'made1_free': 'FTMade', 
        'made2_dunk': 'FG2Made_Dunk',
        'made2_jump': 'FG2Made_Jump', 
        'made2_lay': 'FG2Made_Lay', 
        'made2_tip': 'FG2Made_Tip', 
        'made3_jump': 'FG3Made', 
        'miss1_free': 'FTMiss',
        'miss2_dunk': 'FG2Miss_Dunk', 
        'miss2_jump': 'FG2Miss_Jump', 
        'miss2_lay': 'FG2Miss_Lay', 
        'miss2_tip': 'FG2Miss_Tip', 
        'miss3_jump': 'FG3Miss',
        'reb_dead': 'Rebound_Dead', 
        'reb_def': 'Rebound_Def', 
        'reb_off': 'Rebound_Off', 
        'steal': 'Steal', 
        'sub_in': 'SubIn', 
        'sub_out': 'SubOut',
        'timeout': 'Timeout', 
        'timeout_tv': 'Timeout_Tv', 
        'turnover': 'Turnover'
    })
    stats.reset_index(inplace=True)
    stats = pd.merge(
        left=stats,
        left_on=['EventPlayerID', 'TeamID'],
        right=players[['PlayerName', 'TeamID', 'PlayerID']],
        right_on=['PlayerID', 'TeamID']
    )
    stats = pd.merge(
        left=stats,
        on=['TeamID'],
        right=teams[['TeamID', 'TeamName']]
    )

    stats = stats.loc[
        stats['PlayerName'] != 'TEAM'
    ]

    # Get total makes and attempts for each FG type
    stats['FG2Made'] = stats[
        [x for x in stats.columns if x[0:8] in ['FG2Made_']]
    ].sum(axis=1)
    stats['FG2Miss'] = stats[
        [x for x in stats.columns if x[0:8] in ['FG2Miss_']]
    ].sum(axis=1)
    stats['FTAttempts'] = stats['FTMade'] + stats['FTMiss']
    stats['FG2Attempts'] = stats['FG2Made'] + stats['FG2Miss']
    stats['FG3Attempts'] = stats['FG3Made'] + stats['FG3Miss']

    stats['FGMade'] = stats[['FG2Made','FG3Made']].sum(axis=1)
    stats['FGAttempts'] = stats[['FG2Attempts','FG3Attempts']].sum(axis=1)

    # Get total rebound stats
    stats['Rebound'] = stats['Rebound_Off'] + stats['Rebound_Def']

    # Get total points
    stats['FTPoints'] = stats['FTMade']
    stats['FG2Points'] = 2 * stats['FG2Made']
    stats['FG3Points'] = 3 * stats['FG3Made']
    stats['Points'] = stats[['FTPoints','FG2Points','FG3Points']].sum(axis=1)

    # Get FT, FG2, FG3 percents, as well as FG2 sub-percents
    stats['FTPct'] = stats['FTMade'] / stats['FTAttempts']
    stats['FG2Pct'] = stats['FG2Made'] / stats['FG2Attempts']
    stats['FG3Pct'] = stats['FG3Made'] / stats['FG3Attempts']
    for field in [x for x in stats.columns if x[0:8] == 'FG2Made_']:
        stats['FG2Pct_'+field[8:]] = stats[field] / (stats['FG2Miss_'+field[8:]] + stats[field])

    #Get percent of points be each method of scoring
    stats['PoP_FT'] = stats['FTPoints'] / stats['Points']
    stats['PoP_FG2'] = stats['FG2Points'] / stats['Points']
    stats['PoP_FG3'] = stats['FG3Points'] / stats['Points']
    for field in [x for x in stats.columns if x[0:8] == 'FG2Made_']:
        stats['PoP_FG2_'+field[8:]] = 2 * stats[field] / stats['Points']

    # Import metric weights
    metricWeights = {
        'Assist': 0.24561605945314355,
        'Block': 0.2913149423305327,
        'Rebound_Def': 1.4015734163656688,
        'FG3Attempts': 0.010632721754413665,
        'FG3Made': 0.32666311740864595,
        'FGAttempts': -1.5660483057500763,
        'FGMade': 1.0417011863300154,
        'FTAttempts': -0.654577237820814,
        'FTMade': 0.514528138861222,
        'FoulPers': -0.09527342961131798,
        'Rebound_Off': 1.5593318779997507,
        'Points': 0.36681611725432,
        'Steal': 1.6295532984362322,
        'Turnover': -1.3731004953979695
    }

    scaledStats = pd.DataFrame(
        StandardScaler().fit_transform(stats[
            [x for x in stats.columns if x[-2:] != 'ID' and x[-4:] != 'Name']
        ]),
        columns=[x for x in stats.columns if x[-2:] != 'ID' and x[-4:] != 'Name'],
        index=stats.index
    )

    scaledStats['Composite'] = 0
    for key in metricWeights.keys():
        scaledStats['Composite'] += scaledStats[key] * metricWeights[key]
    
    stats = pd.merge(
        left=stats,
        right=scaledStats['Composite'],
        left_index=True,
        right_index=True
    )

    stats['Season'] = season


    return stats

#%% Run function
out = pd.DataFrame()
for x in np.arange(2010,2020):
    print(f"Working on {x}")
    out = out.append(createSeasonDf(x))

out['CompositePctl'] = (out['Composite'] - np.min(out['Composite'])) / (np.max(out['Composite'])- np.min(out['Composite']))

del out['EventPlayerID']
leftColumns = ['Season', 'TeamName', 'PlayerName', 'TeamID', 'PlayerID']
out = out[leftColumns + [x for x in out.columns if x not in leftColumns]]

out.to_csv('PlayerData.csv')


