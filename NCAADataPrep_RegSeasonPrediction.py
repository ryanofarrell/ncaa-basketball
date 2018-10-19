#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 11 20:20:16 2018

@author: Ryan
"""

import pandas as pd
import time
import datetime as datetime

# Set benchmark time to start
begin = time.time()


###############################################################################
# Create "Print Time" UDF
# Inputs: prefix (string to print); timetoprint (numeric time to print)
# Outputs: Prints the prefix plus the timetoprint, in seconds or minutes
###############################################################################
def printtime(prefix,timetoprint):
    if timetoprint < 60:
        print(prefix + str(round((timetoprint),3)) + ' sec')
    else:
        print(prefix + str(round((timetoprint)/60,3)) + ' min')
###############################################################################
###############################################################################

###############################################################################
# Ingest all data up to end of 2018 season, clean up
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

# Create high-level counts of games, detailed games, and a count missing details in rsg_summary
rsg_summary1 = rsg[['Season','GameDate']].groupby(['Season']).agg('count').reset_index()
rsg_summary2 = rsg[['Season','DetailedGame']].groupby(['Season']).agg('sum').reset_index()
rsg_summary = pd.merge(rsg_summary1,rsg_summary2,how = 'inner', on = ['Season'])
rsg_summary = rsg_summary.rename(columns={'GameDate':'GameCount','DetailedGame':'DetailedGameCount'})
rsg_summary['MissingDetails'] = rsg_summary['GameCount'] - rsg_summary['DetailedGameCount']
del rsg_summary1, rsg_summary2, rsg['DetailedGame']


###############################################################################
# Create a record for each team for each game in rsg, rather than a record for each game
###############################################################################
# Duplicate rsg into loser rsg
lrsg = rsg.copy()

# Rename columns in rsg to standardized format
rsg = rsg.rename(columns = {'WTeamID':'TmID','WScore':'TmPF','LTeamID':'OppID','LScore':'OppPF','WLoc':'TmLoc'})
rsg = rsg.rename(columns = {'WFGM':'TmFGM','WFGA':'TmFGA','WFGM3':'TmFGM3','WFGA3':'TmFGA3','WFTM':'TmFTM','WFTA':'TmFTA'})
rsg = rsg.rename(columns = {'WOR':'TmOR','WDR':'TmDR','WAst':'TmAst'})
rsg = rsg.rename(columns = {'WTO':'TmTO','WStl':'TmStl','WBlk':'TmBlk','WPF':'TmFoul'})
rsg = rsg.rename(columns = {'LFGM':'OppFGM','LFGA':'OppFGA','LFGM3':'OppFGM3','LFGA3':'OppFGA3','LFTM':'OppFTM','LFTA':'OppFTA'})
rsg = rsg.rename(columns = {'LOR':'OppOR','LDR':'OppDR','LAst':'OppAst'})
rsg = rsg.rename(columns = {'LTO':'OppTO','LStl':'OppStl','LBlk':'OppBlk','LPF':'OppFoul'})
rsg['TmWin'] = 1

# Rename columns in lrsg to standardized format
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

# Append lrsg to rsg, delete lrsg
rsg = rsg.append(lrsg)
del lrsg

# Bring in team names for both Tm and Opp
rsg = pd.merge(rsg,teams[['TeamID','TeamName']],left_on='TmID',right_on='TeamID')
del rsg['TeamID']
rsg = rsg.rename(columns = {'TeamName':'TmName'})
rsg = pd.merge(rsg,teams[['TeamID','TeamName']],left_on='OppID',right_on='TeamID')
del rsg['TeamID']
rsg = rsg.rename(columns = {'TeamName':'OppName'})

###############################################################################
# Create additional stat records in rsg
###############################################################################
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
metrics = ['PF','Margin','FGM','FGA','FGM3','FGA3','FGM2','FGA2','FTA','FTM',
            'Ast','OR','DR','TR','TO','Stl','Blk','Foul']

# Calculate per-40 metrics for each game in rsg
for x in {'Opp','Tm'}:
    for column in metrics:
        rsg[x + column + 'per40'] = rsg[x + column] / rsg['GameMins'] * 40
del column, x

# Create summable fields, to sum when calculating stats for a teak
summables = ['GameMins','TmWin','TmGame']
for x in {'Opp','Tm'}:
    for column in metrics:
        summables.append(x + column)
del column, x

###############################################################################
# Create rsgdates dataframe
###############################################################################
# Pull out each unique game date/Season combo, count # of games on those dates
rsgdates = rsg[['Season','GameDate','TmName']].groupby(['Season','GameDate']).agg('count').reset_index()
rsgdates = rsgdates.rename(columns = {'TmName':'GameCount'})

# For testing purposes, limit the size to run end-to-end in reasonable time
#rsgdates = rsgdates.loc[1:7]
rsgdates = rsgdates.loc[rsgdates['Season'] == 2018]

###############################################################################
# Define opponentadjust UDF (for ease of opponent-adjusting metrics)
# Inputs: OAmetric (metric to opponent-adjust)
# Outputs: column in iteams that is opponent-adjusted, based on the games in predate_rsg
###############################################################################
def opponentadjust(OAmetric):
    global predate_rsg, iteams
    
    # Figure out the prefix & core metric, for use later
    if OAmetric[:2] == 'Tm':
        prefix = OAmetric[:2]
        otherprefix = 'Opp'
        coremetric = OAmetric[2:]    
    if OAmetric[:3] == 'Opp':
        prefix = OAmetric[:3]
        otherprefix = 'Tm'
        coremetric = OAmetric[3:]
    
    # Into temp_iteams, get the team names, season, and the opposite side's metric 
    # of what is being adjusted
    # For example, when OAing TmPFper40, temp_iteams will contain the team's OppPFper40
    # This is used later for comparing a team's performance to the opponent's average
    temp_iteams = iteams[['TmName','Season',otherprefix+coremetric]]
    
    # Rename my opponent's metric to say it's *their* average <insert metric>
    # Rename to OppAvg_OppPFper40 (it's my opponent's average opponents (me) PF per 40)
    temp_iteams = temp_iteams.rename(columns = {otherprefix+coremetric:'OppAvg_'+otherprefix+coremetric})

    # Merge in this info into predate_rsg, for the opponent in predate_rsg
    predate_rsg = pd.merge(predate_rsg,temp_iteams,left_on=['OppName','Season'],right_on=['TmName','Season'],how='left',suffixes=('','_y'))
    del predate_rsg['TmName_y']
    
    # In predate_rsg, determine for that game how the Tm did vs Opp_Avg's
    # Example, GameOppAdj_TmPFper40 = TmPFper40 - OppAvg_OppPFper40
    # I.e., how did I do in this game vs my opponent's average opponent
    predate_rsg['GameOppAdj_'+OAmetric] = predate_rsg[OAmetric] - predate_rsg['OppAvg_'+otherprefix+coremetric]

    # Inverse it for when you start with an opponent, to make positive numbers good
    if prefix == 'Opp':
        predate_rsg['GameOppAdj_'+OAmetric] = predate_rsg['GameOppAdj_'+OAmetric] * -1
    
    # In iteamstemp, sum the opponent-adjusted metric and get a new average
    # Example, sum(GameOppAdj_TmPFper40) gets you the TOTAL OA_PFper40
    iteamstemp = predate_rsg.groupby(['TmName','Season'])['GameOppAdj_'+OAmetric].sum().reset_index()

    # bring that value back into iteams, adjust for a 40-min game
    iteams = pd.merge(iteams,iteamstemp,left_on=['TmName','Season'],right_on=['TmName','Season'],how='left',suffixes=('','_y'))
    iteams = iteams.rename(columns = {'GameOppAdj_'+OAmetric:'OA_'+OAmetric})
    iteams['OA_'+OAmetric] = iteams['OA_'+OAmetric] / iteams['GameMins'] * 40
#    del iteams['TmName_y']
###############################################################################
###############################################################################

###############################################################################
# Prep for looping
###############################################################################
# Create rsg_pred (rsg for predicting), rsg_pred_out (same + output)
rsg_pred = rsg[['Season','GameDate','TmName','OppName']]
rsg_pred_out = pd.DataFrame()

# Print pre-looping time
prelooptime = time.time()
printtime('Pre-Loop Time: ',prelooptime-begin)

# Set counter, count total loops for time projections
loopcount = 1
totalloops = rsgdates['GameDate'].size

###############################################################################
# Loop through all seasondates to get stats at the time of the games being played
###############################################################################
for row in rsgdates.itertuples():
    
    # Set included season & the day of the game, from the tuple being looped through
    inclseason = row[1]
    dayofgame = row[2]
    
    # Print current season, game, and progress
    print ('Season: ' + str(inclseason) + '; Date: ' + str(dayofgame.strftime('%Y-%m-%d')) +
           '; Loop ' + str(loopcount) + ' of ' + str(totalloops) + ' (' + 
           str(round(loopcount/totalloops*100,2)) + '%)')

    # predate_rsg = regular season games for the given season that occurred before the game date
    # Only include games before the given date to mimic predicting the game before it happens
    predate_rsg = rsg.loc[(rsg['Season'] == inclseason) & (rsg['GameDate'] < dayofgame)]
    
    # Create iteams by summing the summable fields
    iteams = predate_rsg.groupby(['TmID','TmName'])[summables].sum().reset_index()
    
    # Get per-40 season stats into iteams (can't just sum per-40s since it will incorrectly weight some games)
    for x in {'Opp','Tm'}:
        for column in metrics:
            iteams[x + column + 'per40'] = iteams[x + column] / iteams['GameMins'] * 40
    del column, x
    
    # put Season & GameDate into iteams so we know on what to merge back to the baseline games
    iteams['Season'] = inclseason
    iteams['GameDate'] = dayofgame
    
    # Calculate Win Pct
    iteams['TmWin%'] = iteams['TmWin'] / iteams['TmGame']
    
    # Calculate Shooting Percentages
    iteams['TmFGPct'] = iteams['TmFGMper40'] / iteams['TmFGAper40']
    iteams['TmFG3Pct'] = iteams['TmFGM3per40'] / iteams['TmFGA3per40']
    iteams['TmFG2Pct'] = iteams['TmFGM2per40'] / iteams['TmFGA2per40']
    iteams['TmFTPct'] = iteams['TmFTMper40'] / iteams['TmFTAper40']
    
    # Calculate Ast per FG
    iteams['TmAstRate'] = iteams['TmAstper40'] / iteams['TmFGMper40']
    
    # Check if this is a detailed year; if so, opponentadjust everything
    # If not, only opponent-adjust PF, PA, Margin
    if inclseason < 2003:
        for x in {'Opp','Tm'}:
            for column in ['PF','Margin']:
                opponentadjust(x + column + 'per40')
    else:
        for x in {'Opp','Tm'}:
            for column in metrics:
                opponentadjust(x + column + 'per40')
    del column, x
    
    # Merge into rsg_pred the Tm1 data from the finished iteams dataframe
    iteamsTm1 = iteams.add_prefix('Tm1_')
    rsg_pred_tms = pd.merge(rsg_pred,iteamsTm1,left_on=['Season','GameDate','TmName'],right_on=['Tm1_Season','Tm1_GameDate','Tm1_TmName'],how='left')

    # Merge into rsg_pred_tms the Tm2 data from the finished iteams dataframe
    iteamsTm2 = iteams.add_prefix('Tm2_')
    rsg_pred_tms = pd.merge(rsg_pred_tms,iteamsTm2,left_on=['Season','GameDate','OppName'],right_on=['Tm2_Season','Tm2_GameDate','Tm2_TmName'],how='left')

    # Drop rows in rsg_pred_tms where there is no data
    rsg_pred_tms = rsg_pred_tms.dropna(subset=['Tm1_TmID','Tm2_TmID'])
    
    # Add to rsg_pred_out the finished rsg_pred_tms dataframe
    rsg_pred_out = rsg_pred_out.append(rsg_pred_tms) 

    # Project remaining time to get through loops
    totallooptime = time.time() - prelooptime
    loopsleft = totalloops - loopcount
    printtime('Time Left: ',loopsleft * totallooptime / loopcount)
    loopcount = loopcount + 1

###############################################################################
# End looping
###############################################################################
    
# Benchmark time
postlooptime = time.time()
printtime('Total Loop Time: ', postlooptime-prelooptime)

# Clean-up after loops
del totalloops, totallooptime, loopcount, loopsleft
del inclseason, iteamsTm1, iteamsTm2, rsg_pred_tms
del row, rsg_pred, dayofgame

# Write output
now = str(datetime.datetime.now().strftime('%Y-%m-%d %H-%M-%S'))
rsg_pred_out.to_csv('/Users/Ryan/Google Drive/ncaa-basketball-data/rsg_predate_' + now + '.csv', index=False)

# Benchmark write time
postwritetime = time.time()
printtime('Write to CSV time: ',postwritetime-postlooptime)

# Benchmark total time
totaltime = time.time()-begin
printtime('Total Process Time: ',totaltime)

del begin, totaltime, prelooptime, postlooptime, now
