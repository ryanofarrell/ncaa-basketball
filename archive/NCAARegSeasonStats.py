"""
Created on Mon Jul 30 20:39:24 2018

@author: Ryan O'Farrell
"""

import pandas as pd
import time
import datetime as datetime
#from sklearn.model_selection import train_test_split
#from sklearn.ensemble import RandomForestRegressor
#import numpy as np

from db import get_db



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

# Calculate field goal percentages in each game
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
metrics = ['PF','Margin','FGM','FGA',
            'FGM3','FGA3','FGM2','FGA2','Ast','OR','DR','TR',
            'FTA','FTM','TO','Stl','Blk','Foul']

# Getting in game per-40 for latter opponent adjusting
for x in {'Opp','Tm'}:
    for column in metrics:
        rsg[x + column + 'per40'] = rsg[x + column] / rsg['GameMins'] * 40
del column, x

# Create summable fields
summables = ['GameMins','TmWin','TmGame']
for x in {'Opp','Tm'}:
    for column in metrics:
        summables.append(x + column)
del column, x

# Create seasonteams dataframe, getting in season stats
seasonteams = rsg.groupby(['TmID','TmName','Season'])[summables].sum().reset_index()


# Per-40 adjust the season stats, for later compare to in-game stats
for x in {'Opp','Tm'}:
    for column in metrics:
        seasonteams[x + column + 'per40'] = seasonteams[x + column] / seasonteams['GameMins'] * 40
del column, x

# Calculate season-long percentages
seasonteams['TmFGPct'] = seasonteams['TmFGM'] / seasonteams['TmFGA']
seasonteams['TmFG3Pct'] = seasonteams['TmFGM3'] / seasonteams['TmFGA3']
seasonteams['TmFG2Pct'] = seasonteams['TmFGM2'] / seasonteams['TmFGA2']
seasonteams['TmFTPct'] = seasonteams['TmFTM'] / seasonteams['TmFTA']
seasonteams['OppFGPct'] = seasonteams['OppFGM'] / seasonteams['OppFGA']
seasonteams['OppFG3Pct'] = seasonteams['OppFGM3'] / seasonteams['OppFGA3']
seasonteams['OppFG2Pct'] = seasonteams['OppFGM2'] / seasonteams['OppFGA2']
seasonteams['OppFTPct'] = seasonteams['OppFTM'] / seasonteams['OppFTA']

# Double Check for columns showing up in both
#rsg_cols = pd.DataFrame(list(rsg)).reset_index()
#seasonteams_cols = pd.DataFrame(list(seasonteams)).reset_index()
#col_diffs = pd.merge(rsg_cols, seasonteams_cols, on=[0],how='outer')

# Benchmark time
poatime = time.time()-begin
if poatime < 60:
    print('Pre-Opponent-Adjust Time: ' + str(round((poatime),2)) + ' sec')
else:
    print('Pre-Opponent-Adjust Time: ' + str(round((poatime)/60,2)) + ' min')


###############################################################################
###############################################################################
# Define opponentadjust UDF (for ease of opponent-adjusting metrics)
###############################################################################
###############################################################################
def opponentadjust(OAmetric):
    global rsg, seasonteams

    # Figure out the prefix, core metric, for use later
    if OAmetric[:2] == 'Tm':
        prefix = OAmetric[:2]
        otherprefix = 'Opp'
        coremetric = OAmetric[2:]
    if OAmetric[:3] == 'Opp':
        prefix = OAmetric[:3]
        otherprefix = 'Tm'
        coremetric = OAmetric[3:]
    # print (coremetric + prefix)

    # From iteams put average PF into opponent side of irsg
    # Example, Opp_AvgPF_Against, Opp_AvgPA_Against
    # If I am OAing TmPFper40 (my offense proficiency), I want to get OppPFper40 for my opponent
    # So, for a TmName, get their OppPFper40, aka their PAper40
    tempseasonteams = seasonteams[['TmName','Season',otherprefix+coremetric]]

    # Rename my opponent's metric to say it's *their* average <insert metric>
    # Rename to OppAvg_OppScoreper40 (it's my opponent's average opponents (me) score per 40)
    tempseasonteams = tempseasonteams.rename(columns = {otherprefix+coremetric:'OppAvg_'+otherprefix+coremetric})

    # Merge in this info into irsg, for the opponent in irsg
    rsg = pd.merge(rsg,tempseasonteams,left_on=['OppName','Season'],right_on=['TmName','Season'],how='left',suffixes=('','_y'))
    del rsg['TmName_y']

    # In irsg, determine for that game how the Tm did vs Opp_Avg's
    # Example, GameOppAdj_TmPFper40 = TmPFper40 - OppAvg_OppPFper40
    rsg['GameOppAdj_'+OAmetric] = rsg[OAmetric] - rsg['OppAvg_'+otherprefix+coremetric]

    # switch it for when you start with an opponent
    if prefix == 'Opp':
        rsg['GameOppAdj_'+OAmetric] = rsg['GameOppAdj_'+OAmetric] * -1

    # In iteamstemp, sum the opponent-adjusted metric and get a new average
    # Example, sum(GameOppAdj_TmPFper40) gets you the TOTAL OA_PFper40
    seasonteamstemp = rsg.groupby(['TmName','Season'])['GameOppAdj_'+OAmetric].sum().reset_index()

    # bring that value back into iteams
    seasonteams = pd.merge(seasonteams,seasonteamstemp,left_on=['TmName','Season'],right_on=['TmName','Season'],how='left',suffixes=('','_y'))
    seasonteams = seasonteams.rename(columns = {'GameOppAdj_'+OAmetric:'OA_'+OAmetric})
    seasonteams['OA_'+OAmetric] = seasonteams['OA_'+OAmetric] / seasonteams['GameMins'] * 40
#    del iteams['TmName_y']

# Opponent-adjust all metrics
for x in {'Opp','Tm'}:
    for column in metrics:
        opponentadjust(x + column + 'per40')
del column, x

# Opponent-adjust percentages
for x in {'Opp','Tm'}:
    for column in ('FG','FG3','FG2','FT'):
        opponentadjust(x + column + 'Pct')
del column, x


# Benchmark time
prewritetime = time.time()-begin
if prewritetime < 60:
    print('OA Time: ' + str(round((prewritetime),2)) + ' sec')
else:
    print('OA Time: ' + str(round((prewritetime)/60,2)) + ' min')

now = str(datetime.datetime.now().strftime('%Y-%m-%d %H-%M-%S'))
rsg.to_csv('/Users/Ryan/Google Drive/ncaa-basketball-data/rsg_' + now + '.csv', index=False)
seasonteams.to_csv('/Users/Ryan/Google Drive/ncaa-basketball-data/seasonteams_' + now + '.csv', index=False)


#testseasonteams = seasonteams.loc[(seasonteams['TmName'] == 'Florida')&(seasonteams['Season'] <= 2018)]
#testseason = seasonteams.loc[(seasonteams['Season'] == 2018)]
#testrsg = rsg.loc[(rsg['TmName'] == 'Florida')&(rsg['Season'] == 2018)]



# Benchmark time
totaltime = time.time()-begin
if totaltime < 60:
    print('Total Process Time: ' + str(round((totaltime),2)) + ' sec')
else:
    print('Total Process Time: ' + str(round((totaltime)/60,2)) + ' min')

del begin, totaltime, poatime