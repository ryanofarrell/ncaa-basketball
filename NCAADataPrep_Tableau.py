#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 11 17:38:36 2018

@author: Ryan
"""

# import libraries
import pandas as pd
#from datetime import datetime, timedelta
from UDFs import printtime
import os
import numpy as np
#from urllib.request import urlopen
#from bs4 import BeautifulSoup
from scipy.stats import rankdata
import time
#import pygsheets

# Set benchmark time to start
begin = time.time()

###############################################################################
# Ingest all data (rsgc and rsgd being the main ones)
###############################################################################
rsgc = pd.read_csv(filepath_or_buffer = '/Users/Ryan/Google Drive/ncaa-basketball-data/2018-kaggle-update/RegularSeasonCompactResults.csv')
rsgd = pd.read_csv('/Users/Ryan/Google Drive/ncaa-basketball-data/2018-kaggle-update/RegularSeasonDetailedResults.csv')
teams = pd.read_csv(filepath_or_buffer = '/Users/Ryan/Google Drive/ncaa-basketball-data/2018-kaggle/Teams.csv')
teamspellings = pd.read_csv(filepath_or_buffer = '/Users/Ryan/Google Drive/ncaa-basketball-data/2018-kaggle/TeamSpellings.csv',encoding = "ISO-8859-1")
seasons = pd.read_csv(filepath_or_buffer = '/Users/Ryan/Google Drive/ncaa-basketball-data/2018-kaggle/Seasons.csv')
TourneySeeds = pd.read_csv(filepath_or_buffer = '/Users/Ryan/Google Drive/ncaa-basketball-data/2018-kaggle/NCAATourneySeeds.csv')
TourneyResults = pd.read_csv(filepath_or_buffer = '/Users/Ryan/Google Drive/HistoricalNCAAData/TourneyCompactResults.csv')

# Deal with these later
#TourneyResults2017 = pd.read_csv(filepath_or_buffer = '/Users/Ryan/Google Drive/HistoricalNCAAData/2017TournamentResults.csv')
#rsgcDates = pd.read_csv(filepath_or_buffer = '/Users/Ryan/Google Drive/HistoricalNCAAData/rsgcDates.csv')
#TourneyGamesDates = pd.read_csv(filepath_or_buffer = '/Users/Ryan/Google Drive/HistoricalNCAAData/TourneyGamesDates.csv')
#TourneySlots = pd.read_csv(filepath_or_buffer = '/Users/Ryan/Google Drive/HistoricalNCAAData/TourneySlots.csv')



###################################################
###### CURRENT YEAR SECTION #######################
###################################################
# TODO update to detailed results
# https://www.sports-reference.com/cbb/boxscores/index.cgi?month=12&day=4&year=2018

# import libraries
from urllib.request import urlopen
from bs4 import BeautifulSoup

# Pull Web Data
webpage = "https://www.masseyratings.com/scores.php?s=305972&sub=11590&all=1"
page = urlopen(webpage)
soup = BeautifulSoup(page,'lxml')
NCAADataStr = str(soup.pre)
del webpage

# Import Data into DF
x = NCAADataStr.split('\n')
rsg_curr = pd.DataFrame(x, columns = ['RawStr'])
del NCAADataStr

# Remove last 4 rows
rsg_curr = rsg_curr[:-4]

# Remove/replace strings
rsg_curr['RawStr'].replace(regex=True,inplace=True,to_replace=r'&amp;',value=r'&')
rsg_curr['RawStr'].replace(regex=True,inplace=True,to_replace=r'<pre>',value=r'')

# Split string into columns
rsg_curr['GameDate'] = rsg_curr['RawStr'].str[:10]
rsg_curr['Team'] = rsg_curr['RawStr'].str[10:36]
rsg_curr['TeamScore'] = rsg_curr['RawStr'].str[36:39]
rsg_curr['Opponent'] = rsg_curr['RawStr'].str[39:65]
rsg_curr['OpponentScore'] = rsg_curr['RawStr'].str[65:68]
rsg_curr['GameOT'] = pd.to_numeric(rsg_curr['RawStr'].str[70:71],errors='coerce')
rsg_curr['GameOT'] = np.nan_to_num(rsg_curr['GameOT'])
del rsg_curr['RawStr']

del x

# Strip Whitespaces
rsg_curr['GameDate'] = rsg_curr['GameDate'].str.strip()
rsg_curr['TeamScore'] = rsg_curr['TeamScore'].str.strip()
rsg_curr['Team'] = rsg_curr['Team'].str.strip()
rsg_curr['Opponent'] = rsg_curr['Opponent'].str.strip()
rsg_curr['OpponentScore'] = rsg_curr['OpponentScore'].str.strip()

# Change column types
rsg_curr[['TeamScore','OpponentScore']] = rsg_curr[['TeamScore','OpponentScore']].apply(pd.to_numeric)
rsg_curr['GameDate'] = pd.to_datetime(rsg_curr['GameDate'])

# Calculate Margin and team locations
rsg_curr['WLoc'] = ''
rsg_curr['Season'] = 2019
rsg_curr.loc[(rsg_curr['Team'].str[:1] == '@'), 'WLoc'] = 'H'
rsg_curr.loc[(rsg_curr['Opponent'].str[:1] == '@'), 'WLoc'] = 'A'
rsg_curr.loc[(rsg_curr['Opponent'].str[:1] != '@') & (rsg_curr['Team'].str[:1] != '@'), 'WLoc'] = 'N'

# Remove @
rsg_curr['Team'].replace(regex=True,inplace=True,to_replace=r'@',value=r'')
rsg_curr['Opponent'].replace(regex=True,inplace=True,to_replace=r'@',value=r'')


# Rename columns for merge
rsg_curr = rsg_curr.rename(columns = {'TeamScore':'WScore','OpponentScore':'LScore',
                                  'GameOT':'NumOT','Location':'TmLoc'})

# Get team IDs into NCAADf
# NOTE cal baptist added to teamspellings, teams
# NOTE pfw added to teamspellings
rsg_curr = rsg_curr.rename(columns = {'Team':'TeamNameSpelling'})
rsg_curr['TeamNameSpelling'] = rsg_curr['TeamNameSpelling'].str.lower()
rsg_curr = pd.merge(rsg_curr,teamspellings[['TeamNameSpelling','TeamID']],on='TeamNameSpelling',how = 'left')
rsg_curr = rsg_curr.rename(columns = {'TeamNameSpelling':'TmName','TeamID':'WTeamID'})

rsg_curr = rsg_curr.rename(columns = {'Opponent':'TeamNameSpelling'})
rsg_curr['TeamNameSpelling'] = rsg_curr['TeamNameSpelling'].str.lower()
rsg_curr = pd.merge(rsg_curr,teamspellings[['TeamNameSpelling','TeamID']],on='TeamNameSpelling',how = 'left')
rsg_curr = rsg_curr.rename(columns = {'TeamNameSpelling':'OppName','TeamID':'LTeamID'})

# Drop non-mapped teams (DII, exhibitions)
#nonmappedteams = rsg_curr.loc[rsg_curr['TeamID'].isnull()].groupby('TeamNameSpelling').agg('count').reset_index()
rsg_curr = rsg_curr.dropna(how = 'any')

# Drop teamnamespelling version of name
del rsg_curr['TmName'], rsg_curr['OppName']

###################################################
###################################################
###################################################


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

# Append current-year data to rsgc
# TODO when detailed results, append to RSGD and RSGC
rsgc = rsgc.append(rsg_curr)

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

# Add field for number of possessions (NCAA NET method)
rsg['TmPoss'] = rsg['TmFGA'] - rsg['TmOR'] + rsg['TmTO'] + .475 * rsg['TmFTA']
rsg['OppPoss'] = rsg['OppFGA'] - rsg['OppOR'] + rsg['OppTO'] + .475 * rsg['OppFTA']


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

# Calculate per-poss metrics for each game in rsg
for x in {'Opp','Tm'}:
    for column in metrics:
        rsg[x + column + 'perPoss'] = rsg[x + column] / rsg[x + 'Poss']
del column, x



# Create summable fields, to sum when calculating stats for a team
summables = ['GameMins','TmWin','TmGame']
for x in {'Opp','Tm'}:
    for column in metrics:
        summables.append(x + column)
del column, x

###############################################################################
# Create rsgdates dataframe
###############################################################################
# Pull out each unique game date/Season combo, count # of games on those dates
# TODO see if i need this section
#rsgdates = rsg[['Season','GameDate','TmName']].groupby(['Season','GameDate']).agg('count').reset_index()
#rsgdates = rsgdates.rename(columns = {'TmName':'GameCount'})

# For testing purposes, limit the size to run end-to-end in reasonable time
#rsgdates = rsgdates.loc[1:7]
#rsgdates = rsgdates.loc[rsgdates['Season'] == 2018]

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


# TODO bring in stuff from other file

# =============================================================================
# # Add MArch 11 games
# March11Games = pd.read_csv(filepath_or_buffer = '/Users/Ryan/Google Drive/HistoricalNCAAData/March11Games.csv')
# rsgc = rsgc.append(March11Games)
# del March11Games
# 
# =============================================================================

# Calculate PF per 40 mins for each game
rsgc['GameMins'] = 40 + rsgc['Numot']*5
rsgc['WTeamPFper40'] = rsgc['Wscore']*40/rsgc['GameMins']
rsgc['LTeamPFper40'] = rsgc['Lscore']*40/rsgc['GameMins']

# Create SeasonTeams DataFrame
SeasonTeams = pd.DataFrame(columns = ['Team_Id', 'Team_Name', 'Season'])
for t in range(1985,2018):
    Teams['Season'] = t
    SeasonTeams = SeasonTeams.append(Teams)
    
# Delete Hardin-Simmons in 2018 season
Teams = Teams[Teams['Team_Name']!='Hardin-Simmons']
Teams['Season'] = 2018
SeasonTeams = SeasonTeams.append(Teams)
del t
SeasonTeams['Season'] = pd.to_numeric(SeasonTeams['Season'])
SeasonTeams['Team_Id'] = pd.to_numeric(SeasonTeams['Team_Id'])

# Create Wins and Losses dataframes
WinsStats = rsgc.groupby(['Wteam', 'Season'])[['Wscore','Lscore','GameMins']].agg(['sum','count']).reset_index()
LossesStats = rsgc.groupby(['Lteam', 'Season'])[['Wscore','Lscore','GameMins']].agg(['sum','count']).reset_index()
WinsStats.columns = ['Team_Id','Season','WinsPF','Wins','WinsPA','del','WinsMins','del2']
LossesStats.columns = ['Team_Id','Season','LossesPA','Losses','LossesPF','del','LossesMins','del2']
del WinsStats['del'], WinsStats['del2'], LossesStats['del'], LossesStats['del2']

# Merge wins and losses together
SeasonTeams = pd.merge(WinsStats, SeasonTeams,on=['Team_Id','Season'],how='right')
SeasonTeams = pd.merge(LossesStats, SeasonTeams,on=['Team_Id','Season'],how='right')
SeasonTeams.fillna(0,inplace=True)
del WinsStats, LossesStats

# Calculate row stats
SeasonTeams['Games'] = SeasonTeams['Wins'] + SeasonTeams['Losses']
SeasonTeams['WinPct'] = SeasonTeams['Wins']/SeasonTeams['Games']
SeasonTeams['Mins'] = SeasonTeams['WinsMins'] + SeasonTeams['LossesMins']
SeasonTeams['PF'] = SeasonTeams['WinsPF'] + SeasonTeams['LossesPF']
SeasonTeams['PA'] = SeasonTeams['WinsPA'] + SeasonTeams['LossesPA']
SeasonTeams['PFper40'] = SeasonTeams['PF']*40/SeasonTeams['Mins']
SeasonTeams['PAper40'] = SeasonTeams['PA']*40/SeasonTeams['Mins']
del SeasonTeams['WinsPF'], SeasonTeams['WinsPA'], SeasonTeams['LossesPF'], SeasonTeams['LossesPA']
SeasonTeams['Marginper40'] = SeasonTeams['PFper40'] - SeasonTeams['PAper40']
SeasonTeams = SeasonTeams.drop(SeasonTeams[SeasonTeams.Mins < 401].index)

# Put seasonteam stats into games
rsgc = rsgc.rename(columns = {'Wteam':'Team_Id'})
rsgc = pd.merge(rsgc,SeasonTeams[['Season','Team_Id','PFper40','PAper40','WinPct']],on=['Season','Team_Id'],how='left')
rsgc = rsgc.rename(columns = {'Team_Id':'Wteam','PFper40':'WteamAvgPFper40','PAper40':'WteamAvgPAper40','WinPct':'WteamWinPct'})

rsgc = rsgc.rename(columns = {'Lteam':'Team_Id'})
rsgc = pd.merge(rsgc,SeasonTeams[['Season','Team_Id','PFper40','PAper40','WinPct']],on=['Season','Team_Id'],how='left')
rsgc = rsgc.rename(columns = {'Team_Id':'Lteam','PFper40':'LteamAvgPFper40','PAper40':'LteamAvgPAper40','WinPct':'LteamWinPct'})

# Calculate game scores and stuff for games
rsgc['WTeamGameOffScore'] = rsgc['WTeamPFper40'] - rsgc['LteamAvgPAper40']
rsgc['LTeamGameOffScore'] = rsgc['LTeamPFper40'] - rsgc['WteamAvgPAper40']
rsgc['WTeamGameDefScore'] = rsgc['LteamAvgPFper40'] - rsgc['LTeamPFper40']
rsgc['LTeamGameDefScore'] = rsgc['WteamAvgPFper40'] - rsgc['WTeamPFper40']
rsgc['WTeamGameOAM'] = rsgc['WTeamGameOffScore'] + rsgc['WTeamGameDefScore']
rsgc['LTeamGameOAM'] = rsgc['LTeamGameOffScore'] + rsgc['LTeamGameDefScore']

# Bring the game info back to the aggregated table
WinsStats = rsgc.groupby(['Wteam', 'Season'])[['WTeamGameOffScore','WTeamGameDefScore','WTeamGameOAM','LteamWinPct']].agg(['sum']).reset_index()
LossesStats = rsgc.groupby(['Lteam', 'Season'])[['LTeamGameOffScore','LTeamGameDefScore','LTeamGameOAM','WteamWinPct']].agg(['sum']).reset_index()
WinsStats.columns = ['Team_Id','Season','WinsOffScoreSum','WinsDefScoreSum','WinsOAMSum','WinsOWP']
LossesStats.columns = ['Team_Id','Season','LossesOffScoreSum','LossesDefScoreSum','LossesOAMSum','LossesOWP']
SeasonTeams = pd.merge(WinsStats, SeasonTeams,on=['Team_Id','Season'],how='right')
SeasonTeams = pd.merge(LossesStats, SeasonTeams,on=['Team_Id','Season'],how='right')
del WinsStats, LossesStats

SeasonTeams['WinsOWP'] = np.nan_to_num(SeasonTeams['WinsOWP'])
SeasonTeams['LossesOWP'] = np.nan_to_num(SeasonTeams['LossesOWP'])

SeasonTeams['OWP'] = (SeasonTeams['WinsOWP'] + SeasonTeams['LossesOWP'])/SeasonTeams['Games']
del SeasonTeams['WinsOWP'], SeasonTeams['LossesOWP']

## FINISH RPI!!
# Put seasonteam stats into games
rsgc = rsgc.rename(columns = {'Wteam':'Team_Id'})
rsgc = pd.merge(rsgc,SeasonTeams[['Season','Team_Id','OWP']],on=['Season','Team_Id'],how='left')
rsgc = rsgc.rename(columns = {'Team_Id':'Wteam','OWP':'WteamOWP'})

rsgc = rsgc.rename(columns = {'Lteam':'Team_Id'})
rsgc = pd.merge(rsgc,SeasonTeams[['Season','Team_Id','OWP']],on=['Season','Team_Id'],how='left')
rsgc = rsgc.rename(columns = {'Team_Id':'Lteam','OWP':'LteamOWP'})

# Bring the game info back to the aggregated table
WinsStats = rsgc.groupby(['Wteam', 'Season'])[['LteamOWP']].agg(['sum']).reset_index()
LossesStats = rsgc.groupby(['Lteam', 'Season'])[['WteamOWP']].agg(['sum']).reset_index()
WinsStats.columns = ['Team_Id','Season','WinsOOWP']
LossesStats.columns = ['Team_Id','Season','LossesOOWP']
SeasonTeams = pd.merge(WinsStats, SeasonTeams,on=['Team_Id','Season'],how='right')
SeasonTeams = pd.merge(LossesStats, SeasonTeams,on=['Team_Id','Season'],how='right')
del WinsStats, LossesStats

SeasonTeams['WinsOOWP'] = np.nan_to_num(SeasonTeams['WinsOOWP'])
SeasonTeams['LossesOOWP'] = np.nan_to_num(SeasonTeams['LossesOOWP'])

SeasonTeams['OOWP'] = (SeasonTeams['WinsOOWP'] + SeasonTeams['LossesOOWP'])/SeasonTeams['Games']
del SeasonTeams['WinsOOWP'], SeasonTeams['LossesOOWP']
SeasonTeams['RPI'] = .25 * SeasonTeams['WinPct'] + .5 * SeasonTeams['OWP'] + .25 * SeasonTeams['OOWP']
del SeasonTeams['WinPct'], SeasonTeams['OWP'], SeasonTeams['OOWP']
del rsgc['WteamOWP'], rsgc['LteamOWP'], rsgc['LteamWinPct'], rsgc['WteamWinPct']


######################################
## CONFIGURE rsgc FOR TABLEAU#####
######################################
rsgc = rsgc.rename(columns = {'Wteam':'Team','Wscore':'TeamScore','Lteam':'Opponent','Lscore':'OpponentScore'})
rsgc = rsgc.rename(columns = {'WTeamName':'TeamName','LTeamName':'OpponentName','Wloc':'Location'})
del rsgc['WTeamPFper40'], rsgc['LTeamPFper40'], rsgc['WteamAvgPFper40'], rsgc['WteamAvgPAper40']
del rsgc['LteamAvgPFper40'], rsgc['LteamAvgPAper40']
rsgc = rsgc.rename(columns = {'WTeamGameOffScore':'TeamOffScore','WTeamGameDefScore':'TeamDefScore','WTeamGameOAM':'TeamOAM'})
rsgc['Result'] = 'W'

# Make Losses DF
RSLosses = pd.DataFrame(columns = ['Date'])
RSLosses['Date'] = rsgc['Date']
RSLosses['Team'] = rsgc['Opponent']
RSLosses['TeamName'] = rsgc['OpponentName']
RSLosses['Opponent'] = rsgc['Team']
RSLosses['OpponentName'] = rsgc['TeamName']
RSLosses['TeamScore'] = rsgc['OpponentScore']
RSLosses['OpponentScore'] = rsgc['TeamScore']
RSLosses['Location'] = rsgc['Location']
RSLosses.loc[(RSLosses['Location'] == 'H'), 'Location'] = 'AA'
RSLosses.loc[(RSLosses['Location'] == 'A'), 'Location'] = 'H'
RSLosses.loc[(RSLosses['Location'] == 'AA'), 'Location'] = 'A'
RSLosses['TeamOffScore'] = rsgc['LTeamGameOffScore']
RSLosses['TeamDefScore'] = rsgc['LTeamGameDefScore']
RSLosses['TeamOAM'] = rsgc['LTeamGameOAM']
RSLosses['Numot'] = rsgc['Numot']
RSLosses['Season'] = rsgc['Season']
RSLosses['Result'] = 'L'
del rsgc['LTeamGameOffScore'], rsgc['LTeamGameDefScore'], rsgc['LTeamGameOAM'], rsgc['GameMins']

# Combine them
rsgc = rsgc.append(RSLosses)
del RSLosses

# Rank Game Percentiles
rsgc['OffPercentile'] = 1 - (len(rsgc)-rankdata(rsgc['TeamOffScore'],method='min'))/len(rsgc)
rsgc['DefPercentile'] = 1 - (len(rsgc)-rankdata(rsgc['TeamDefScore'],method='min'))/len(rsgc)
rsgc['OAMPercentile'] = 1 - (len(rsgc)-rankdata(rsgc['TeamOAM'],method='min'))/len(rsgc)

######################################
######################################
######################################

# Make all columns numeric
SeasonTeams['WinsOffScoreSum'] = np.nan_to_num(SeasonTeams['WinsOffScoreSum'])
SeasonTeams['LossesOffScoreSum'] = np.nan_to_num(SeasonTeams['LossesOffScoreSum'])
SeasonTeams['WinsDefScoreSum'] = np.nan_to_num(SeasonTeams['WinsDefScoreSum'])
SeasonTeams['LossesDefScoreSum'] = np.nan_to_num(SeasonTeams['LossesDefScoreSum'])
SeasonTeams['WinsOAMSum'] = np.nan_to_num(SeasonTeams['WinsOAMSum'])
SeasonTeams['LossesOAMSum'] = np.nan_to_num(SeasonTeams['LossesOAMSum'])

# Calculate new row stats
SeasonTeams['TotalOffScore'] = SeasonTeams['WinsOffScoreSum'] + SeasonTeams['LossesOffScoreSum']
SeasonTeams['TotalDefScore'] = SeasonTeams['WinsDefScoreSum'] + SeasonTeams['LossesDefScoreSum']
SeasonTeams['TotalOAM'] = SeasonTeams['WinsOAMSum'] + SeasonTeams['LossesOAMSum']
SeasonTeams['AvgOffScore'] = SeasonTeams['TotalOffScore']/SeasonTeams['Games']
SeasonTeams['AvgDefScore'] = SeasonTeams['TotalDefScore']/SeasonTeams['Games']
SeasonTeams['AvgOAM'] = SeasonTeams['TotalOAM']/SeasonTeams['Games']
SeasonTeams['SoS'] = SeasonTeams['AvgOAM'] - SeasonTeams['Marginper40']


# Rank on Metrics
Tdf2 = pd.DataFrame(columns = ['OffRank','DefRank','OAMRank','SoSRank','MarginRank','RPIRank','Season','Team_Id'])
for t in range(1985,2019):
    Tdf = SeasonTeams.drop(SeasonTeams[SeasonTeams.Season != t].index)
    Tdf['OffRank'] = Tdf['AvgOffScore'].rank(ascending = False, na_option = 'bottom', method = 'min')
    Tdf['DefRank'] = Tdf['AvgDefScore'].rank(ascending = False, na_option = 'bottom', method = 'min')
    Tdf['OAMRank'] = Tdf['AvgOAM'].rank(ascending = False, na_option = 'bottom', method = 'min')
    Tdf['SoSRank'] = Tdf['SoS'].rank(ascending = False, na_option = 'bottom', method = 'min')
    Tdf['MarginRank'] = Tdf['Marginper40'].rank(ascending = False, na_option = 'bottom', method = 'min')
    Tdf['RPIRank'] = Tdf['RPI'].rank(ascending = False, na_option = 'bottom', method = 'min')
    Tdf = Tdf[['OffRank','DefRank','OAMRank','SoSRank','MarginRank','RPIRank','Season','Team_Id']]
    Tdf2 = Tdf2.append(Tdf)
SeasonTeams = pd.merge(Tdf2, SeasonTeams,on=['Team_Id','Season'],how='right')
del t, Tdf, Tdf2

# Get Opponent Rank into Games
rsgc = rsgc.rename(columns = {'Opponent':'Team_Id'})
rsgc = pd.merge(rsgc,SeasonTeams[['Season','Team_Id','OAMRank']],on=['Season','Team_Id'],how='left')
rsgc = rsgc.rename(columns = {'Team_Id':'Opponent','OAMRank':'OpponentRank'})

# Del extra columns
del SeasonTeams['PF'], SeasonTeams['PA'], SeasonTeams['WinsMins'], SeasonTeams['LossesMins']
del SeasonTeams['LossesOffScoreSum'], SeasonTeams['LossesDefScoreSum'], SeasonTeams['LossesOAMSum']
del SeasonTeams['WinsOffScoreSum'], SeasonTeams['WinsDefScoreSum'], SeasonTeams['WinsOAMSum']
del SeasonTeams['TotalOffScore'], SeasonTeams['TotalDefScore'], SeasonTeams['TotalOAM']

######################################
######################################
######################################


####################################################
## Tourney Section ##
####################################################


# Merge results files
del TourneyResults2017['Unnamed: 0']
TourneyResults2017 = TourneyResults2017.rename(columns = {'Wteam':'Team_Name'})
TourneyResults2017 = pd.merge(TourneyResults2017,Teams,on=['Team_Name'],how='left')
del TourneyResults2017['Team_Name']
TourneyResults2017 = TourneyResults2017.rename(columns = {'Team_Id':'Wteam'})
TourneyResults2017 = TourneyResults2017.rename(columns = {'Lteam':'Team_Name'})
TourneyResults2017 = pd.merge(TourneyResults2017,Teams,on=['Team_Name'],how='left')
TourneyResults2017 = TourneyResults2017.rename(columns = {'Team_Id':'Lteam'})
del TourneyResults2017['Team_Name']
del TourneyResults2017['Season_x'], TourneyResults2017['Season_y']
TourneyResults2017['Season'] = 2017
TourneyResults = TourneyResults.append(TourneyResults2017)
del TourneyResults2017

# Pull seeds into results
TourneySeeds = TourneySeeds.rename(columns = {'Team':'Wteam'})
TourneyResults = pd.merge(TourneyResults,TourneySeeds,on=['Season','Wteam'])
TourneySeeds = TourneySeeds.rename(columns = {'Wteam':'Team'})
TourneyResults = TourneyResults.rename(columns = {'Seed':'WteamSeed'})
TourneySeeds = TourneySeeds.rename(columns = {'Team':'Lteam'})
TourneyResults = pd.merge(TourneyResults,TourneySeeds,on=['Season','Lteam'])
TourneySeeds = TourneySeeds.rename(columns = {'Lteam':'Team_Id'})
TourneyResults = TourneyResults.rename(columns = {'Seed':'LteamSeed'})

# Play-in game flag, seeds
TourneyResults['PlayInFlag'] = np.where((TourneyResults['WteamSeed'].str.len() == 4) & (TourneyResults['LteamSeed'].str.len() == 4), 'Y', '')
TourneyResults['WteamSeed'] = pd.to_numeric(TourneyResults['WteamSeed'].str[1:3])
TourneyResults['LteamSeed'] = pd.to_numeric(TourneyResults['LteamSeed'].str[1:3])
TourneyResults = TourneyResults[TourneyResults['PlayInFlag']!='Y']

# Get num of wins for team in tourney
TourneyWins = TourneyResults.groupby(['Wteam','Season'])[['Wscore']].agg('count').reset_index()
TourneyWins = TourneyWins.rename(columns = {'Wscore':'TourneyWins','Wteam':'Team_Id'})
TourneyLosses = TourneyResults.groupby(['Lteam','Season'])[['Lscore']].agg('count').reset_index()
TourneyLosses = TourneyLosses.rename(columns = {'Lscore':'TourneyLosses','Lteam':'Team_Id'})
SeasonTeams = pd.merge(TourneyWins, SeasonTeams,on=['Team_Id','Season'],how='right')
SeasonTeams = pd.merge(TourneyLosses, SeasonTeams,on=['Team_Id','Season'],how='right')
SeasonTeams.fillna(0,inplace=True)
del TourneyWins, TourneyLosses
SeasonTeams['TourneyGames'] = SeasonTeams['TourneyWins'] + SeasonTeams['TourneyLosses']
SeasonTeams['TourneyApp'] = np.where((SeasonTeams['TourneyGames'] >= 1), 'Y', '')

# Get Tourney Seeds into SeasonTeams
SeasonTeams = pd.merge(TourneySeeds, SeasonTeams,on=['Team_Id','Season'],how='right')
SeasonTeams['Seed'] = pd.to_numeric(SeasonTeams['Seed'].str[1:3])
SeasonTeams['Seed'][(SeasonTeams['TourneyApp'] != 'Y')] = ''
SeasonTeams['Seed'] = pd.to_numeric(SeasonTeams['Seed'])

# Get 2018 Tourney Results In
# =============================================================================
TourneySeeds18 = pd.read_csv(filepath_or_buffer = '/Users/Ryan/Google Drive/HistoricalNCAAData/18TourneySeeds.csv')
SeasonTeams = pd.merge(SeasonTeams, TourneySeeds18, on=['Season','Team_Name'],how='left')
SeasonTeams['Seed'] = SeasonTeams['Seed_x'].fillna(SeasonTeams['Seed_y'])
del SeasonTeams['Seed_x'], SeasonTeams['Seed_y']
SeasonTeams['TourneyApp'] = np.where((SeasonTeams['Seed'] >= 1), 'Y', '')
SeasonTeams['Season'] = pd.to_numeric(SeasonTeams['Season'])
#SeasonTeams['TourneyApp_x'] = SeasonTeams['TourneyApp_x'].replace(r'\s+', np.nan, regex=True)
#SeasonTeams['TourneyApp'] = SeasonTeams['TourneyApp_x'].fillna(SeasonTeams['TourneyApp_y'])
del SeasonTeams['TourneyApp_x'], SeasonTeams['TourneyApp_y']
del TourneySeeds18
# =============================================================================
####################################################
####################################################
####################################################

####################################################
#######Tourney Games for Tableau    ################
####################################################

# Merge day 0 into rsgc
TourneyResults = pd.merge(TourneyResults,Seasons,on='Season',)
del TourneyResults['Regionw']
del TourneyResults['Regionx']
del TourneyResults['Regiony']
del TourneyResults['Regionz']

# NOTE: This takes tons of HP so it is commented for now
#TourneyResults['Dayzero'] =  pd.to_datetime(TourneyResults['Dayzero'])
#temp = TourneyResults['Daynum'].apply(pd.np.ceil).apply(lambda x: pd.Timedelta(x, unit='D'))
#TourneyResults['Date'] = TourneyResults['Dayzero'] + temp
## Export Date Info
#TourneyGamesDates = TourneyResults['Date']
#os.chdir('/Users/Ryan/Desktop/HistoricalNCAAData/')
#TourneyGamesDates.to_csv('TourneyGamesDates.csv',header=True)
del TourneyResults['Dayzero']
del TourneyResults['Daynum']
del TourneyGamesDates['Unnamed: 0']
TourneyResults = pd.merge(TourneyResults,TourneyGamesDates,left_index=True, right_index=True)
del TourneyGamesDates
TourneyResults['Date'] = TourneyResults['Date_x'].fillna(TourneyResults['Date_y'])
del TourneyResults['Date_x'], TourneyResults['Date_y'], TourneyResults['Wloc'], TourneyResults['PlayInFlag']

# Rename things and append
TourneyResults = TourneyResults.rename(columns = {'Wteam':'Team_Id','Wscore':'TeamScore','Lscore':'OpponentScore'})
del Teams['Season']
TourneyResults = pd.merge(TourneyResults, Teams ,on=['Team_Id'],how='left')
TourneyResults = TourneyResults.rename(columns = {'Team_Name':'TeamName','Team_Id':'Team','Lteam':'Team_Id'})
TourneyResults = pd.merge(TourneyResults, Teams ,on=['Team_Id'],how='left')
TourneyResults = TourneyResults.rename(columns = {'Team_Name':'OpponentName','Team_Id':'Opponent'})
TourneyResults['Result'] = 'W'

# Make Losses Df
TourneyLosses = pd.DataFrame(columns = ['Date'])
TourneyLosses['Date'] = TourneyResults['Date']
TourneyLosses['Team'] = TourneyResults['Opponent']
TourneyLosses['TeamName'] = TourneyResults['OpponentName']
TourneyLosses['Opponent'] = TourneyResults['Team']
TourneyLosses['OpponentName'] = TourneyResults['TeamName']
TourneyLosses['TeamScore'] = TourneyResults['OpponentScore']
TourneyLosses['OpponentScore'] = TourneyResults['TeamScore']
TourneyLosses['Numot'] = TourneyResults['Numot']
TourneyLosses['Season'] = TourneyResults['Season']
TourneyLosses['WteamSeed'] = TourneyResults['LteamSeed']
TourneyLosses['LteamSeed'] = TourneyResults['WteamSeed']
TourneyLosses['Result'] = 'L'

# Combine them
TourneyResults = TourneyResults.append(TourneyLosses)
del TourneyLosses
TourneyResults['TourneyGame'] = 'Y'

rsgc = rsgc.append(TourneyResults)
####################################################
####################################################
####################################################


# Test DF
Test = SeasonTeams.loc[(SeasonTeams['Season'] == 2018) & (SeasonTeams['TourneyApp'] == 'Y')]
TestYear = rsgc.loc[(rsgc['Season'] == 2018)]

middle = time.time()
printtime('Pre-Write time: ',middle-begin)


# Write File
#os.chdir('/Users/Ryan/Google Drive/HistoricalNCAAData/')
#SeasonTeams.to_excel(excel_writer = 'SeasonTeams.xlsx', sheet_name = 'SeasonTeams', engine='xlsxwriter')
#rsgc.to_excel(excel_writer = 'rsgc.xlsx', sheet_name = 'rsgc', engine='xlsxwriter')
#TourneyResults.to_excel(excel_writer = 'TourneyResults.xlsx', sheet_name = 'TourneyResults', engine='xlsxwriter')

## Google Sheets Section
#gc = pygsheets.authorize(outh_file='/Users/Ryan/Google Drive/HistoricalNCAAData/client_secret_171083257821-45099lsapu7balflossoqkncmqm8rfdp.apps.googleusercontent.com.json')
#sh = gc.open('PythonNCAAData')
#
## Write SeasonTeams
#wks = sh.sheet1
#wks.rows = SeasonTeams.shape[0] + 1
#wks.cols = SeasonTeams.shape[1]
#wks.set_dataframe(SeasonTeams,(1,1))
#end0 = time.time()
#print ('Post-SeasonTeamsWrite time: ' + str(round(end0 - middle,2)))
#
## Write rsgc
#sh = gc.open('PythonNCAAData1')
#wks = sh.sheet1
#wks.cols = 21
#wks.rows = 10000
#wks.set_dataframe(rsgc.iloc[0:9999,:],(1,1))
#end1 = time.time()
#print ('Post-rsgcWrite time: ' + str(round(end1 - end1,2)))


end = time.time()

printtime('Post-Write time: ',end - middle)
del begin, middle, end
