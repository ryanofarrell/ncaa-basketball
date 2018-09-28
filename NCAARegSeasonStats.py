"""
Created on Mon Jul 30 20:39:24 2018

@author: Ryan O'Farrell
"""

import pandas as pd
import time
#import datetime as datetime
#from sklearn.model_selection import train_test_split
#from sklearn.ensemble import RandomForestRegressor
#import numpy as np

begin = time.time()

###############################################################################
###############################################################################
# Ingest all data up to end of 2018 season, clean up, and get required fields
###############################################################################
###############################################################################
# rsgc = regular season games compact, rsgd = regular season games detailed
rsgc = pd.read_csv('/Users/Ryan/Google Drive/ncaa-basketball-data/2018-kaggle-update/RegularSeasonCompactResults.csv')
rsgd = pd.read_csv('/Users/Ryan/Google Drive/ncaa-basketball-data/2018-kaggle-update/RegularSeasonDetailedResults.csv')

seasons = pd.read_csv(filepath_or_buffer = '/Users/Ryan/Desktop/DataFiles/Seasons.csv')
teams = pd.read_csv(filepath_or_buffer = '/Users/Ryan/Desktop/DataFiles/Teams.csv')
    
# Merge in day 0 to rsgc & rsgd, add days to day 0 to get date of game, delete extra columns
rsgc = pd.merge(rsgc,seasons[['Season','DayZero']],on='Season')
rsgc['DayZero'] = pd.to_datetime(rsgc['DayZero'],format='%m/%d/%Y')
rsgc['DayNum'] = pd.to_timedelta(rsgc['DayNum'],unit='d')
rsgc['GameDate'] = rsgc['DayZero'] + rsgc['DayNum']
del rsgc['DayNum'], rsgc['DayZero']

rsgd = pd.merge(rsgd,seasons[['Season','DayZero']],on='Season')
rsgd['DayZero'] = pd.to_datetime(rsgd['DayZero'],format='%m/%d/%Y')
rsgd['DayNum'] = pd.to_timedelta(rsgd['DayNum'],unit='d')
rsgd['GameDate'] = rsgd['DayZero'] + rsgd['DayNum']
del rsgd['DayNum'], rsgd['DayZero']

# Merge together compact and detailed results when possible, delete old dataframes
rsg = pd.merge(left = rsgc, right = rsgd, how = 'left', on = ['GameDate','Season','WTeamID','LTeamID',
                                                              'WScore','LScore','WLoc','NumOT'])
del rsgc, rsgd

# Create detailedgame field in rsg to indicate if the game has details or not
rsg['DetailedGame'] = 0
rsg.loc[(rsg['WFGM'] > 0),'DetailedGame'] = 1

# Create high-level counts of games, detailed games, and a count missing details
rsg_summary1 = rsg[['Season','GameDate']].groupby(['Season']).agg('count').reset_index()
rsg_summary2 = rsg[['Season','DetailedGame']].groupby(['Season']).agg('sum').reset_index()
rsg_summary = pd.merge(rsg_summary1,rsg_summary2,how = 'inner', on = ['Season'])
rsg_summary = rsg_summary.rename(columns={'GameDate':'GameCount','DetailedGame':'DetailedGameCount'})
rsg_summary['MissingDetails'] = rsg_summary['GameCount'] - rsg_summary['DetailedGameCount']
del rsg_summary1, rsg_summary2, rsg['DetailedGame']

# Duplicate rsg into loser rsg
lrsg = rsg.copy()

# Rename columns for rsg
rsg = rsg.rename(columns = {'WTeamID':'TmID','WScore':'TmPF','LTeamID':'OppID','LScore':'OppPF','WLoc':'TmLoc'})
rsg = rsg.rename(columns = {'WFGM':'TmFGM','WFGA':'TmFGA','WFGM3':'TmFGM3','WFGA3':'TmFGA3','WFTM':'TmFTM','WFTA':'TmFTA'})
rsg = rsg.rename(columns = {'WOR':'TmOR','WDR':'TmDR','WAst':'TmAst'})
rsg = rsg.rename(columns = {'WTO':'TmTO','WStl':'TmStl','WBlk':'TmBlk','WPF':'TmFoul'})
rsg = rsg.rename(columns = {'LFGM':'OppFGM','LFGA':'OppFGA','LFGM3':'OppFGM3','LFGA3':'OppFGA3','LFTM':'OppFTM','LFTA':'OppFTA'})
rsg = rsg.rename(columns = {'LOR':'OppOR','LDR':'OppDR','LAst':'OppAst'})
rsg = rsg.rename(columns = {'LTO':'OppTO','LStl':'OppStl','LBlk':'OppBlk','LPF':'OppFoul'})
rsg['TmWin'] = 1


# Rename columns for lrsg
lrsg = lrsg.rename(columns = {'WTeamID':'OppID','WScore':'OppPF','LTeamID':'TmID','LScore':'TmPF'})
lrsg = lrsg.rename(columns = {'WFGM':'OppFGM','WFGA':'OppFGA','WFGM3':'OppFGM3','WFGA3':'OppFGA3','WFTM':'OppFTM','WFTA':'OppFTA'})
lrsg = lrsg.rename(columns = {'WOR':'OppOR','WDR':'OppDR','WAst':'OppAst'})
lrsg = lrsg.rename(columns = {'WTO':'OppTO','WStl':'OppStl','WBlk':'OppBlk','WPF':'OppFoul'})
lrsg = lrsg.rename(columns = {'LFGM':'TmFGM','LFGA':'TmFGA','LFGM3':'TmFGM3','LFGA3':'TmFGA3','LFTM':'TmFTM','LFTA':'TmFTA'})
lrsg = lrsg.rename(columns = {'LOR':'TmOR','LDR':'TmDR','LAst':'TmAst'})
lrsg = lrsg.rename(columns = {'LTO':'TmTO','LStl':'TmStl','LBlk':'TmBlk','LPF':'TmFoul'})
lrsg['TmWin'] = 0


# Adjust locations in loser rsg
lrsg.loc[(lrsg['WLoc'] == 'H'),'TmLoc'] = 'A'
lrsg.loc[(lrsg['WLoc'] == 'A'),'TmLoc'] = 'H'
lrsg.loc[(lrsg['WLoc'] == 'N'),'TmLoc'] = 'N'
del lrsg['WLoc']

# Append lrsg to rsg, delete lrsg,
rsg = rsg.append(lrsg)
del lrsg

# Bring in team names for both Tm and Opp
rsg = pd.merge(rsg,teams[['TeamID','TeamName']],left_on='TmID',right_on='TeamID')
del rsg['TeamID']
rsg = rsg.rename(columns = {'TeamName':'TmName'})
rsg = pd.merge(rsg,teams[['TeamID','TeamName']],left_on='OppID',right_on='TeamID')
del rsg['TeamID']
rsg = rsg.rename(columns = {'TeamName':'OppName'})

# Add countable field for number of games
rsg['TmGame'] = 1

# Add field for number of minutes
rsg['GameMins'] = 40 + rsg['NumOT']*5

# Add field for Total Rebounds
rsg['TmTR'] = rsg['TmOR'] + rsg['TmDR']
rsg['OppTR'] = rsg['OppOR'] + rsg['OppDR']

# Count number of FGA2/FGM2
rsg['TmFGM2'] = rsg['TmFGM'] - rsg['TmFGM3']
rsg['TmFGA2'] = rsg['TmFGA'] - rsg['TmFGA3']
rsg['OppFGM2'] = rsg['OppFGM'] - rsg['OppFGM3']
rsg['OppFGA2'] = rsg['OppFGA'] - rsg['OppFGA3']

# Get field goal percentages
rsg['TmFGPct'] = rsg['TmFGM'] / rsg['TmFGA']
rsg['TmFG3Pct'] = rsg['TmFGM3'] / rsg['TmFGA3']
rsg['TmFG2Pct'] = rsg['TmFGM2'] / rsg['TmFGA2']
rsg['TmFTPct'] = rsg['TmFTM'] / rsg['TmFTA']
rsg['OppFGPct'] = rsg['OppFGM'] / rsg['OppFGA']
rsg['OppFG3Pct'] = rsg['OppFGM3'] / rsg['OppFGA3']
rsg['OppFG2Pct'] = rsg['OppFGM2'] / rsg['OppFGA2']
rsg['OppFTPct'] = rsg['OppFTM'] / rsg['OppFTA']

# Calculate game margin
rsg['TmMargin'] = rsg['TmPF'] - rsg['OppPF']
rsg['OppMargin'] = -rsg['TmMargin']

# Two prefixes: Tm (Team) and Opp (Opponent)
# Core metrics: PF (points for); Margin; FGM (field goals made); FGA (field goals attempted); FGPct (Field goal percent)
# cont...FGM3 (3pt field goals made); FGA3 (3pt field goals attempted); FG3Pct (3pt field goal percent)
# cont...FGM2 (2pt field goals made); FGA2 (2pt field goals attempted); FG2Pct (2pt field goal percent)
# cont...Ast (assists); OR (offensive rebounds); DR (devensive rebounds); TR (total rebounds)
# cont...FTA (free throws attempted); FTM (free throws made); FTPct (Free throw percent)
# cont...TO (turn overs); Stl (steals); Blk (blocks); Foul (foul)
metrics = ['PF','Margin','FGM','FGA','FGPct',
            'FGM3','FGA3','FG3Pct','FGM2','FGA2','FG2Pct','Ast','OR','DR','TR',
            'FTA','FTM','FTPct','TO','Stl','Blk','Foul']

# Add in per-40 stats to rsg
for x in {'Opp','Tm'}:
    for column in metrics:
        rsg[x + column + 'per40'] = rsg[x + column] / rsg['GameMins'] * 40
del column, x

# Benchmark time
totaltime = time.time()-begin
if totaltime < 60:
    print('Total Process Time: ' + str(round((totaltime),2)) + ' sec')
else:
    print('Total Process Time: ' + str(round((totaltime)/60,2)) + ' min')

del begin, totaltime