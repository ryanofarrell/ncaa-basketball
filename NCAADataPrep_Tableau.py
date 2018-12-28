#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 11 17:38:36 2018

@author: Ryan
"""

# TODO Poll data
# TODO weekly tracker of poll info/general ranks
# TODO refactor so don't have to merge everything when just doing current season
# TODO RPI
# TODO check if there are new games or if this is a big ole waste of time

print('Running Season Team and Game Data')

# import libraries
import pandas as pd
#from datetime import datetime
#from UDFs import printtime
#import os
import numpy as np
#from urllib.request import urlopen
#from bs4 import BeautifulSoup
#from scipy.stats import rankdata
import time
#import pygsheets
from UDFs import createreplacecsv, printtime
from bs4 import BeautifulSoup
from urllib.request import urlopen

# Set benchmark time to start
begin = time.time()

# Pick if you want a full re-run, or just do current season
runall = True
# Set current season variable
currseason = 2019

###############################################################################
###############################################################################
###############################################################################
###############################################################################

# Define opponentadjust UDF
def opponentadjust(prefix, coremetric):
    global rsg_workingseason, st_workingseason, workingseason
    
    # Figure out the prefix & core metric, for use later
    if prefix == 'Tm':
        otherprefix = 'Opp'
    elif prefix == 'Opp':
        otherprefix = 'Tm'

    # If the metric that is fed in has any nulls in the current season,
    # fill the three output columns with NaN
    if st_workingseason[otherprefix + coremetric + 'perGame'].isnull().values.any():
        st_workingseason['OA_'+ prefix + coremetric + 'perGame'] = np.NaN
        st_workingseason['OA_'+ prefix + coremetric + 'per40'] = np.NaN
        st_workingseason['OA_'+ prefix + coremetric + 'perPoss'] = np.NaN
        print('Skipped: ' + str(workingseason) + ' - ' + prefix + coremetric)
    # Otherwise, if there is data to use, opponentadjust
    else:
        # Into temo_st_currseason, get the team names, season, and the opposite side's metric
        # of what is being adjusted
        # For example, when OAing TmPFper40, temp_iteams will contain the team's OppPFper40
        # This is used later for comparing a team's performance to the opponent's average
        temp_st_workingseason = st_workingseason[[
            'TmName', 'Season', otherprefix + coremetric + 'perGame']]
        
    
        # Rename my opponent's metric to say it's *their* average <insert metric>
        # Rename to OppAvg_OppPFper40 (it's my opponent's average opponents (me) PF per 40)
        temp_st_workingseason = temp_st_workingseason.rename(
            columns={otherprefix + coremetric + 'perGame': 'OppAvg_' + otherprefix + coremetric + 'perGame'})
        
        # Merge in this info into predate_rsg, for the opponent in predate_rsg
        rsg_workingseason = pd.merge(rsg_workingseason, temp_st_workingseason, 
                                  left_on=['OppName', 'Season'], 
                                  right_on=['TmName', 'Season'], 
                                  how='left', 
                                  suffixes=('', '_y'))
        del rsg_workingseason['TmName_y']
    
        # In predate_rsg, determine for that game how the Tm did vs Opp_Avg's
        # Example, GameOppAdj_TmPFper40 = TmPFper40 - OppAvg_OppPFper40
        # I.e., how did I do in this game vs my opponent's average opponent
        rsg_workingseason['GameOppAdj_' + prefix + coremetric ] = \
            rsg_workingseason[prefix + coremetric ] - \
            rsg_workingseason['OppAvg_' + otherprefix + coremetric + 'perGame']
            
        del rsg_workingseason['OppAvg_' + otherprefix + coremetric + 'perGame' ]
        # Inverse it for when you start with an opponent, to make positive numbers good
        if prefix == 'Opp':
            rsg_workingseason['GameOppAdj_' + prefix + coremetric ] = \
                rsg_workingseason['GameOppAdj_' + prefix + coremetric ] * -1
    
        # In iteamstemp, sum the opponent-adjusted metric and get a new average
        # Example, sum(GameOppAdj_TmPFper40) gets you the TOTAL OA_PFper40
        temp_st_workingseason = rsg_workingseason.groupby(['TmName', 'Season'])[
            'GameOppAdj_' + prefix + coremetric  ].sum().reset_index()
    
        # bring that value back into iteams, adjust for a 40-min game
        st_workingseason = pd.merge(st_workingseason, temp_st_workingseason, 
                                  on=['TmName', 'Season'], 
                                  how='left')
        st_workingseason = st_workingseason.rename(
            columns={'GameOppAdj_' + prefix + coremetric : 'OA_' + prefix + coremetric })
        
        # Get perGame, perPoss and per40 multipliers
        st_workingseason['OA_'+ prefix + coremetric + 'perGame'] = \
            st_workingseason['OA_'+ prefix + coremetric ] / \
            st_workingseason[prefix + 'Game']
        st_workingseason['OA_'+ prefix + coremetric + 'per40'] = \
            st_workingseason['OA_'+ prefix + coremetric ] / \
            st_workingseason[prefix + 'Mins'] * 40
        st_workingseason['OA_'+ prefix + coremetric + 'perPoss'] = \
            st_workingseason['OA_'+ prefix + coremetric ] / \
            st_workingseason[prefix + 'Poss']
            
        # Delete the useless season aggregate
        del st_workingseason['OA_'+ prefix + coremetric ]
        del rsg_workingseason['GameOppAdj_' + prefix + coremetric]
    
        print('Success: ' + str(workingseason) + ' - ' + prefix + coremetric)

#    del iteams['TmName_y']
###############################################################################
###############################################################################
###############################################################################
###############################################################################

# Current season dataframe creation
        
# Ingest global teamspellings dataframe
teamspellings = pd.read_csv(
    filepath_or_buffer='/Users/Ryan/Google Drive/ncaa-basketball-data/2018-kaggle/TeamSpellings.csv',
    encoding="ISO-8859-1")

metrics = [
    'PF', 'Margin', 'FGM', 'FGA', 'FGM3', 'FGA3', 
    'FGM2', 'FGA2', 'FTA', 'FTM','Ast', 'OR', 'DR', 'TR', 'TO', 'Stl', 'Blk', 'Foul'
]

# Create summable fields, to sum when calculating stats for a team
summables = ['TmGame',
             'OppGame',
             'TmWin',
             'OppWin',
             'TmMins',
             'OppMins',
             'TmPoss',
             'OppPoss',
             ]
for x in {'Opp', 'Tm'}:
    for column in metrics:
        summables.append(x + column)
del column, x

rankmetrics = ['TmSoS']
for oa in {'','OA_'}:
    for prefix in {'Opp', 'Tm'}:
        for coremetric in metrics:
            for suffix in {'perGame','per40','perPoss'}:
                rankmetrics.append(oa + prefix + coremetric + suffix)
del oa, prefix, coremetric, suffix


# Things that I want more of:
# PF, Margin, FGM, FGA, FGM3, FGA3, FGM2, FGA2, FTA, FTM, AST, OR, DR, TR, STL, BLK

# Things I want less of: 
# Foul, TO

# TODO update this once caring about other metrics
ascendingrankmetrics = [
        'OppPFper40','OppPFperGame','OppPFperPoss'
        ,'OppMarginper40','OppMarginperGame','OppMarginperPoss']


# TODO update to detailed results
# https://www.sports-reference.com/cbb/boxscores/index.cgi?month=12&day=4&year=2018

# Pull Web Data
webpage = "https://www.masseyratings.com/scores.php?s=305972&sub=11590&all=1"
page = urlopen(webpage)
soup = BeautifulSoup(page, 'lxml')
NCAADataStr = str(soup.pre)
del webpage, page, soup

# Import Data into DF
x = NCAADataStr.split('\n')
rsg_curr = pd.DataFrame(x, columns=['RawStr'])
del NCAADataStr, x

# Remove last 4 rows
rsg_curr = rsg_curr[:-4]

# Remove/replace strings
rsg_curr['RawStr'].replace(
    regex=True, inplace=True, to_replace=r'&amp;', value=r'&')
rsg_curr['RawStr'].replace(
    regex=True, inplace=True, to_replace=r'<pre>', value=r'')

# Split string into columns
rsg_curr['GameDate'] = rsg_curr['RawStr'].str[:10]
rsg_curr['Team'] = rsg_curr['RawStr'].str[10:36]
rsg_curr['TeamScore'] = rsg_curr['RawStr'].str[36:39]
rsg_curr['Opponent'] = rsg_curr['RawStr'].str[39:65]
rsg_curr['OpponentScore'] = rsg_curr['RawStr'].str[65:68]
rsg_curr['GameOT'] = pd.to_numeric(
    rsg_curr['RawStr'].str[70:71], errors='coerce')
rsg_curr['GameOT'] = np.nan_to_num(rsg_curr['GameOT'])
del rsg_curr['RawStr']

# Strip Whitespaces
rsg_curr['GameDate'] = rsg_curr['GameDate'].str.strip()
rsg_curr['TeamScore'] = rsg_curr['TeamScore'].str.strip()
rsg_curr['Team'] = rsg_curr['Team'].str.strip()
rsg_curr['Opponent'] = rsg_curr['Opponent'].str.strip()
rsg_curr['OpponentScore'] = rsg_curr['OpponentScore'].str.strip()

# Change column types
rsg_curr[['TeamScore',
          'OpponentScore']] = rsg_curr[['TeamScore',
                                        'OpponentScore']].apply(pd.to_numeric)
rsg_curr['GameDate'] = pd.to_datetime(rsg_curr['GameDate'])

# Calculate Margin and team locations
rsg_curr['WLoc'] = ''
rsg_curr['Season'] = currseason
rsg_curr.loc[(rsg_curr['Team'].str[:1] == '@'), 'WLoc'] = 'H'
rsg_curr.loc[(rsg_curr['Opponent'].str[:1] == '@'), 'WLoc'] = 'A'
rsg_curr.loc[(rsg_curr['Opponent'].str[:1] != '@') &
             (rsg_curr['Team'].str[:1] != '@'), 'WLoc'] = 'N'

# Remove @
rsg_curr['Team'].replace(regex=True, inplace=True, to_replace=r'@', value=r'')
rsg_curr['Opponent'].replace(
    regex=True, inplace=True, to_replace=r'@', value=r'')

# Rename columns for merge
rsg_curr = rsg_curr.rename(
    columns={
        'TeamScore': 'WScore',
        'OpponentScore': 'LScore',
        'GameOT': 'NumOT',
        'Location': 'TmLoc'
    })

# Get team IDs into rsg_curr
# NOTE cal baptist added to teamspellings, teams
# NOTE pfw added to teamspellings
rsg_curr = rsg_curr.rename(columns={'Team': 'TeamNameSpelling'})
rsg_curr['TeamNameSpelling'] = rsg_curr['TeamNameSpelling'].str.lower()
rsg_curr = pd.merge(
    rsg_curr,
    teamspellings[['TeamNameSpelling', 'TeamID']],
    on='TeamNameSpelling',
    how='left')
rsg_curr = rsg_curr.rename(columns={
    'TeamNameSpelling': 'TmName',
    'TeamID': 'WTeamID'
})

rsg_curr = rsg_curr.rename(columns={'Opponent': 'TeamNameSpelling'})
rsg_curr['TeamNameSpelling'] = rsg_curr['TeamNameSpelling'].str.lower()
rsg_curr = pd.merge(
    rsg_curr,
    teamspellings[['TeamNameSpelling', 'TeamID']],
    on='TeamNameSpelling',
    how='left')
rsg_curr = rsg_curr.rename(columns={
    'TeamNameSpelling': 'OppName',
    'TeamID': 'LTeamID'
})

del teamspellings

    
# Drop non-mapped teams (DII, exhibitions)
rsg_curr = rsg_curr.dropna(how='any')

# Drop teams with less than 5 games
# Set threshold for number of required games (at least X games to be kept in)
currseasongamethreshold = 5

# Count wins, losses for each team, merge together, get total number of games
rsg_curr_win_sum = rsg_curr[['Season', 'WTeamID']].groupby(
    ['WTeamID']).agg('count').reset_index()
rsg_curr_loss_sum = rsg_curr[['Season', 'LTeamID']].groupby(
    ['LTeamID']).agg('count').reset_index()
rsg_curr_sum = pd.merge(left = rsg_curr_win_sum,
                        right = rsg_curr_loss_sum,
                        how = 'outer',
                        left_on = 'WTeamID',
                        right_on = 'LTeamID',
                        suffixes = ('_win','_loss'))
del rsg_curr_win_sum, rsg_curr_loss_sum
rsg_curr_sum[['Season_win','Season_loss']] = rsg_curr_sum[['Season_win','Season_loss']].fillna(0)
rsg_curr_sum['Games'] = rsg_curr_sum['Season_win'] + rsg_curr_sum['Season_loss']

# Count num of games currently, to output how many are dropped
predroplen = len(rsg_curr)

# Merge in game counts, drop those with less than previously-specified threshold
rsg_curr = pd.merge(left = rsg_curr,
                    right = rsg_curr_sum[['WTeamID','Games']],
                    on = 'WTeamID',
                    )
rsg_curr = rsg_curr.loc[rsg_curr['Games'] >= currseasongamethreshold]
del rsg_curr['Games']
rsg_curr = pd.merge(left = rsg_curr,
                    right = rsg_curr_sum[['LTeamID','Games']],
                    on = 'LTeamID',
                    )
rsg_curr = rsg_curr.loc[rsg_curr['Games'] >= currseasongamethreshold]
del rsg_curr['Games']

# Print output of how many games were dropped
print('Dropped games with one team < ' + str(currseasongamethreshold) + ' games: ' +
      str(predroplen - len(rsg_curr)))

# Drop teamnamespelling version of name
del rsg_curr['TmName'], rsg_curr['OppName'], rsg_curr_sum, predroplen
del currseasongamethreshold

# Until getting detailed results, set detailed columns as nan
rsg_curr = rsg_curr.assign(
           WFGM = np.nan
          ,WFGA = np.nan
          ,WFGM3 = np.nan
          ,WFGA3 = np.nan
          ,WFTM = np.nan
          ,WFTA = np.nan
          ,WAst = np.nan
          ,WOR = np.nan
          ,WDR = np.nan
          ,WTO = np.nan
          ,WStl = np.nan
          ,WBlk = np.nan
          ,WPF = np.nan
          
          ,LFGM = np.nan
          ,LFGA = np.nan
          ,LFGM3 = np.nan
          ,LFGA3 = np.nan
          ,LFTM = np.nan
          ,LFTA = np.nan
          ,LAst = np.nan
          ,LOR = np.nan
          ,LDR = np.nan
          ,LTO = np.nan
          ,LStl = np.nan
          ,LBlk = np.nan
          ,LPF = np.nan
          )

###############################################################################
###############################################################################
###############################################################################
###############################################################################

# Handling of different config options (run all seasons, run just current season)

# If only running this year:
if runall == False:
    
    # Set the loop of the rsg dataframe to just the current season
    seasonstoloop = [currseason]  
    
    # Ingest RSG file that is currently generated, limit to just previous seasons
    readinrsg = pd.read_csv('/Users/Ryan/Google Drive/ncaa-basketball-data/rsg.csv')
    rsg_out = readinrsg.loc[readinrsg['Season'] < currseason]
    del readinrsg

    # Ingest seasonteams dataframe, limit to previous seasons
    readinseasonteams = pd.read_csv('/Users/Ryan/Google Drive/ncaa-basketball-data/seasonteams.csv')
    seasonteams_out = readinseasonteams.loc[readinseasonteams['Season'] < currseason]
    del readinseasonteams

    # Set the working rsg to just the current season's data
    rsg_working = rsg_curr

# Othewise if running all seasons
elif runall == True:
    
    # Set the loop of the rsg dataframe to all season
    seasonstoloop = list(range(1985,currseason+1))

    # Initialize rsg_out
    rsg_out = pd.DataFrame()

    # Initialize seasonteams_out
    seasonteams_out = pd.DataFrame()

    # Ingest necessary dataframes
    seasons = pd.read_csv(
        filepath_or_buffer='/Users/Ryan/Google Drive/ncaa-basketball-data/2018-kaggle/Seasons.csv')
    rsgd = pd.read_csv(
        '/Users/Ryan/Google Drive/ncaa-basketball-data/2018-kaggle-update/RegularSeasonDetailedResults.csv'
    )
    rsgc = pd.read_csv(
            '/Users/Ryan/Google Drive/ncaa-basketball-data/2018-kaggle-update/RegularSeasonCompactResults.csv'
    )
    
    # Merge in day 0 to rsgd, add days to day 0 to get date of game, delete extra columns
    rsgd = pd.merge(rsgd, seasons[['Season', 'DayZero']], on='Season')
    rsgd['DayZero'] = pd.to_datetime(rsgd['DayZero'], format='%m/%d/%Y')
    rsgd['DayNum'] = pd.to_timedelta(rsgd['DayNum'], unit='d')
    rsgd['GameDate'] = rsgd['DayZero'] + rsgd['DayNum']
    del rsgd['DayNum'], rsgd['DayZero']
    
    # Merge in day 0 to rsgd, add days to day 0 to get date of game, delete extra columns
    rsgc = pd.merge(rsgc, seasons[['Season', 'DayZero']], on='Season')
    rsgc['DayZero'] = pd.to_datetime(rsgc['DayZero'], format='%m/%d/%Y')
    rsgc['DayNum'] = pd.to_timedelta(rsgc['DayNum'], unit='d')
    rsgc['GameDate'] = rsgc['DayZero'] + rsgc['DayNum']
    del rsgc['DayNum'], rsgc['DayZero']
    
    # Append current-year data to rsgd
    rsgd = rsgd.append(rsg_curr)
    
    rsg_working = pd.merge(
            left = rsgc,
            right = rsgd,
            how = 'outer',
            on = list(rsgc))

    
    del rsgd, rsgc, seasons
    
###############################################################################
###############################################################################
###############################################################################
###############################################################################
    
# Modifying working rsg dataframe
    
# Create detailedgame field in rsg to indicate if the game has details or not
rsg_working['DetailedGame'] = 0
rsg_working.loc[(rsg_working['WFGM'] > 0), 'DetailedGame'] = 1

# Create high-level counts of games, detailed games, and a count missing details in rsg_summary
rsg_summary1 = rsg_working[['Season', 'GameDate']].groupby(
    ['Season']).agg('count').reset_index()
rsg_summary2 = rsg_working[['Season', 'DetailedGame']].groupby(
    ['Season']).agg('sum').reset_index()
rsg_summary = pd.merge(rsg_summary1, rsg_summary2, how='inner', on=['Season'])
rsg_summary = rsg_summary.rename(columns={
    'GameDate': 'GameCount',
    'DetailedGame': 'DetailedGameCount'
})
rsg_summary['MissingDetails'] = rsg_summary['GameCount'] - rsg_summary[
    'DetailedGameCount']
del rsg_summary1, rsg_summary2, rsg_working['DetailedGame']

###############################################################################
# Create a record for each team for each game in rsg, rather than a record for each game
###############################################################################
# Duplicate rsg into loser rsg
lrsg_working = rsg_working.copy()

# Rename columns in rsg to standardized format
rsg_working = rsg_working.rename(
    columns={
        'WTeamID': 'TmID',
        'WScore': 'TmPF',
        'LTeamID': 'OppID',
        'LScore': 'OppPF',
        'WLoc': 'TmLoc',
        
        'WFGM': 'TmFGM',
        'WFGA': 'TmFGA',
        'WFGM3': 'TmFGM3',
        'WFGA3': 'TmFGA3',
        'WFTM': 'TmFTM',
        'WFTA': 'TmFTA',
        'WOR': 'TmOR', 
        'WDR': 'TmDR', 
        'WAst': 'TmAst',
        'WTO': 'TmTO',
        'WStl': 'TmStl',
        'WBlk': 'TmBlk',
        'WPF': 'TmFoul',
        
        'LFGM': 'OppFGM',
        'LFGA': 'OppFGA',
        'LFGM3': 'OppFGM3',
        'LFGA3': 'OppFGA3',
        'LFTM': 'OppFTM',
        'LFTA': 'OppFTA',
        'LOR': 'OppOR', 
        'LDR': 'OppDR', 
        'LAst': 'OppAst',
        'LTO': 'OppTO',
        'LStl': 'OppStl',
        'LBlk': 'OppBlk',
        'LPF': 'OppFoul',
    })
rsg_working['TmWin'] = 1
rsg_working['OppWin'] = 0

# Rename columns in lrsg to standardized format
lrsg_working = lrsg_working.rename(
    columns={
        'WTeamID': 'OppID',
        'WScore': 'OppPF',
        'LTeamID': 'TmID',
        'LScore': 'TmPF',
        
        'WFGM': 'OppFGM',
        'WFGA': 'OppFGA',
        'WFGM3': 'OppFGM3',
        'WFGA3': 'OppFGA3',
        'WFTM': 'OppFTM',
        'WFTA': 'OppFTA',
        'WOR': 'OppOR',
        'WDR': 'OppDR',
        'WAst': 'OppAst',
        'WTO': 'OppTO',
        'WStl': 'OppStl',
        'WBlk': 'OppBlk',
        'WPF': 'OppFoul',
        
        'LFGM': 'TmFGM',
        'LFGA': 'TmFGA',
        'LFGM3': 'TmFGM3',
        'LFGA3': 'TmFGA3',
        'LFTM': 'TmFTM',
        'LFTA': 'TmFTA',
        'LOR': 'TmOR',
        'LDR': 'TmDR',
        'LAst': 'TmAst',
        'LTO': 'TmTO',
        'LStl': 'TmStl',
        'LBlk': 'TmBlk',
        'LPF': 'TmFoul'
    })
lrsg_working['TmWin'] = 0
lrsg_working['OppWin'] = 1

# Adjust locations in loser rsg
lrsg_working.loc[(lrsg_working['WLoc'] == 'H'), 'TmLoc'] = 'A'
lrsg_working.loc[(lrsg_working['WLoc'] == 'A'), 'TmLoc'] = 'H'
lrsg_working.loc[(lrsg_working['WLoc'] == 'N'), 'TmLoc'] = 'N'
del lrsg_working['WLoc']

# Append lrsg to rsg, delete lrsg
rsg_working = rsg_working.append(lrsg_working)
del lrsg_working

# Bring in team names for both Tm and Opp
teams = pd.read_csv(
    filepath_or_buffer='/Users/Ryan/Google Drive/ncaa-basketball-data/2018-kaggle/Teams.csv')

rsg_working = pd.merge(
    rsg_working, teams[['TeamID', 'TeamName']], left_on='TmID', right_on='TeamID')
del rsg_working['TeamID']
rsg_working = rsg_working.rename(columns={'TeamName': 'TmName'})
rsg_working = pd.merge(
    rsg_working, teams[['TeamID', 'TeamName']], left_on='OppID', right_on='TeamID')
del rsg_working['TeamID']
rsg_working = rsg_working.rename(columns={'TeamName': 'OppName'})

###############################################################################
# Create additional stat records in rsg
###############################################################################
# Add countable field for number of games
rsg_working['TmGame'] = 1
rsg_working['OppGame'] = 1

# Add field for number of minutes
rsg_working['TmMins'] = 40 + rsg_working['NumOT'] * 5
rsg_working['OppMins'] = rsg_working['TmMins']

# Add field for Total Rebounds
rsg_working['TmTR'] = rsg_working['TmOR'] + rsg_working['TmDR']
rsg_working['OppTR'] = rsg_working['OppOR'] + rsg_working['OppDR']

# Count number of FGA2/FGM2
rsg_working['TmFGM2'] = rsg_working['TmFGM'] - rsg_working['TmFGM3']
rsg_working['TmFGA2'] = rsg_working['TmFGA'] - rsg_working['TmFGA3']
rsg_working['OppFGM2'] = rsg_working['OppFGM'] - rsg_working['OppFGM3']
rsg_working['OppFGA2'] = rsg_working['OppFGA'] - rsg_working['OppFGA3']

# Calculate field goal percentages in each game
rsg_working['TmFGPct'] = rsg_working['TmFGM'] / rsg_working['TmFGA']
rsg_working['TmFG3Pct'] = rsg_working['TmFGM3'] / rsg_working['TmFGA3']
rsg_working['TmFG2Pct'] = rsg_working['TmFGM2'] / rsg_working['TmFGA2']
rsg_working['TmFTPct'] = rsg_working['TmFTM'] / rsg_working['TmFTA']
rsg_working['OppFGPct'] = rsg_working['OppFGM'] / rsg_working['OppFGA']
rsg_working['OppFG3Pct'] = rsg_working['OppFGM3'] / rsg_working['OppFGA3']
rsg_working['OppFG2Pct'] = rsg_working['OppFGM2'] / rsg_working['OppFGA2']
rsg_working['OppFTPct'] = rsg_working['OppFTM'] / rsg_working['OppFTA']

# Calculate game margin
rsg_working['TmMargin'] = rsg_working['TmPF'] - rsg_working['OppPF']
rsg_working['OppMargin'] = -rsg_working['TmMargin']

# Add field for number of possessions (NCAA NET method)
rsg_working['TmPoss'] = rsg_working['TmFGA'] \
                - rsg_working['TmOR'] \
                + rsg_working['TmTO'] \
                + .475 * rsg_working['TmFTA']
rsg_working['OppPoss'] = rsg_working['OppFGA'] \
                - rsg_working['OppOR'] \
                + rsg_working['OppTO'] \
                + .475 * rsg_working['OppFTA']

# Calculate per-40 and per-poss metrics for each game in rsg_working
for x in {'Opp', 'Tm'}:
    for column in metrics:
        rsg_working[x + column + 'per40'] = rsg_working[x + column] / rsg_working[x + 'Mins'] * 40
        rsg_working[x + column + 'perPoss'] = rsg_working[x + column] / rsg_working[x + 'Poss']
del column, x

# Create the rsg_out dataframe
# TODO get opponent rank into output


# TODO rank game percentiles, give A, B, C, D, F
# # Rank Game Percentiles
# rsgc['OffPercentile'] = 1 - (len(rsgc)-rankdata(rsgc['TeamOffScore'],method='min'))/len(rsgc)
# rsgc['DefPercentile'] = 1 - (len(rsgc)-rankdata(rsgc['TeamDefScore'],method='min'))/len(rsgc)
# rsgc['OAMPercentile'] = 1 - (len(rsgc)-rankdata(rsgc['TeamOAM'],method='min'))/len(rsgc)



rsg_out = rsg_out.append(rsg_working)

###############################################################################
###############################################################################
###############################################################################
###############################################################################

# Loop through and do a bunch of things

# Loop through the specified seasons to loop
for workingseason in seasonstoloop:

    # Limit the rsg data to just the current season
    rsg_workingseason = rsg_working.loc[rsg_working['Season'] == workingseason]

    # Create current season seasonteams, summing all summables
    # TODO why is summables not summing GameMins?????????????
    st_workingseason = rsg_workingseason.groupby(
            ['TmID', 'TmName'])[summables].sum().reset_index()
#    st_workingseason_wtf = rsg_workingseason.groupby(
#            ['TmID', 'TmName'])['TmMins','OppMins'].sum().reset_index()
#    st_workingseason = st_workingseason.merge(
#            st_workingseason_wtf,
#            how='inner',
#            on=['TmID','TmName'])
#    del st_workingseason_wtf

    # Add season column to seasonteams_currseason
    st_workingseason['Season'] = workingseason
    
    # Get per-game season stats into st_currseason (can't just sum per-games since it will incorrectly weight some games)
    for x in {'Opp','Tm'}:
         for column in metrics:
             st_workingseason[x + column + 'perGame'] = st_workingseason[x + column] / st_workingseason[x + 'Game']
             st_workingseason[x + column + 'per40'] = st_workingseason[x + column] / st_workingseason[x + 'Mins'] * 40
             st_workingseason[x + column + 'perPoss'] = st_workingseason[x + column] / st_workingseason[x+'Poss']
    del column, x

    # Opponent adjust all metrics
    for prefix in {'Opp', 'Tm'}:
        for coremetric in metrics:
                opponentadjust(prefix, coremetric)
    del prefix, coremetric
    
    # Get SoS metric
    st_workingseason['TmSoS'] = st_workingseason['OA_TmMarginper40'] - st_workingseason['TmMarginper40']
    
    # TODO is this duplicate with OppWin?
    # Get Losses into st_workingseason
    st_workingseason['TmLoss'] = st_workingseason['TmGame'] - st_workingseason['TmWin']
    
    # Rank all metrics
    for metric in rankmetrics:
        if metric in ascendingrankmetrics:
            # TODO check if values are null, then don't rank
            st_workingseason['Rank_' + metric ] = st_workingseason[metric].rank(method = 'min',
                                                    ascending = True,
                                                    na_option = 'bottom')
        else:
            st_workingseason['Rank_' + metric] = st_workingseason[metric].rank(method = 'min',
                                                    ascending = False,
                                                    na_option = 'bottom')
    
    # SOS & Rank
    
    del metric

    # Append the working seasons output to the total seasonteams output
    seasonteams_out = seasonteams_out.append(st_workingseason)

###############################################################################
###############################################################################
###############################################################################
###############################################################################

# Do tournament information

trd = pd.read_csv(
    filepath_or_buffer='/Users/Ryan/Google Drive/ncaa-basketball-data/2018-kaggle/NCAATourneyDetailedResults.csv')
trc = pd.read_csv(
    filepath_or_buffer='/Users/Ryan/Google Drive/ncaa-basketball-data/2018-kaggle/NCAATourneyCompactResults.csv')
trd18 = pd.read_csv(
    filepath_or_buffer='/Users/Ryan/Google Drive/ncaa-basketball-data/2018TourneyResults.csv')

trd = trd.append(trd18)


tr = pd.merge(
        left = trc,
        right = trd,
        on = list(trc),
        how = 'outer')

del trc, trd

# Create detailedgame field in rsg to indicate if the game has details or not
tr['DetailedGame'] = 0
tr.loc[(tr['WFGM'] > 0), 'DetailedGame'] = 1

###############################################################################
# Create a record for each team for each game in rsg, rather than a record for each game
###############################################################################
# Duplicate rsg into loser rsg
ltr = tr.copy()

# Rename columns in rsg to standardized format
tr = tr.rename(
    columns={
        'WTeamID': 'TmID',
        'WScore': 'TmPF',
        'LTeamID': 'OppID',
        'LScore': 'OppPF',
        'WLoc': 'TmLoc',
        
        'WFGM': 'TmFGM',
        'WFGA': 'TmFGA',
        'WFGM3': 'TmFGM3',
        'WFGA3': 'TmFGA3',
        'WFTM': 'TmFTM',
        'WFTA': 'TmFTA',
        'WOR': 'TmOR', 
        'WDR': 'TmDR', 
        'WAst': 'TmAst',
        'WTO': 'TmTO',
        'WStl': 'TmStl',
        'WBlk': 'TmBlk',
        'WPF': 'TmFoul',
        
        'LFGM': 'OppFGM',
        'LFGA': 'OppFGA',
        'LFGM3': 'OppFGM3',
        'LFGA3': 'OppFGA3',
        'LFTM': 'OppFTM',
        'LFTA': 'OppFTA',
        'LOR': 'OppOR', 
        'LDR': 'OppDR', 
        'LAst': 'OppAst',
        'LTO': 'OppTO',
        'LStl': 'OppStl',
        'LBlk': 'OppBlk',
        'LPF': 'OppFoul',
    })
tr['TmWin'] = 1
tr['OppWin'] = 0

# Rename columns in lrsg to standardized format
ltr = ltr.rename(
    columns={
        'WTeamID': 'OppID',
        'WScore': 'OppPF',
        'LTeamID': 'TmID',
        'LScore': 'TmPF',
        
        'WFGM': 'OppFGM',
        'WFGA': 'OppFGA',
        'WFGM3': 'OppFGM3',
        'WFGA3': 'OppFGA3',
        'WFTM': 'OppFTM',
        'WFTA': 'OppFTA',
        'WOR': 'OppOR',
        'WDR': 'OppDR',
        'WAst': 'OppAst',
        'WTO': 'OppTO',
        'WStl': 'OppStl',
        'WBlk': 'OppBlk',
        'WPF': 'OppFoul',
        
        'LFGM': 'TmFGM',
        'LFGA': 'TmFGA',
        'LFGM3': 'TmFGM3',
        'LFGA3': 'TmFGA3',
        'LFTM': 'TmFTM',
        'LFTA': 'TmFTA',
        'LOR': 'TmOR',
        'LDR': 'TmDR',
        'LAst': 'TmAst',
        'LTO': 'TmTO',
        'LStl': 'TmStl',
        'LBlk': 'TmBlk',
        'LPF': 'TmFoul'
    })
ltr['TmWin'] = 0
ltr['OppWin'] = 1

# Adjust locations in loser rsg
ltr.loc[(ltr['WLoc'] == 'H'), 'TmLoc'] = 'A'
ltr.loc[(ltr['WLoc'] == 'A'), 'TmLoc'] = 'H'
ltr.loc[(ltr['WLoc'] == 'N'), 'TmLoc'] = 'N'
del ltr['WLoc']

# Append lrsg to rsg, delete lrsg
tr = tr.append(ltr)
del ltr

tr = pd.merge(
    tr, teams[['TeamID', 'TeamName']], left_on='TmID', right_on='TeamID')
del tr['TeamID']
tr = tr.rename(columns={'TeamName': 'TmName'})
tr = pd.merge(
    tr, teams[['TeamID', 'TeamName']], left_on='OppID', right_on='TeamID')
del tr['TeamID']
tr = tr.rename(columns={'TeamName': 'OppName'})

###############################################################################
# Create additional stat records in rsg
###############################################################################
# Add countable field for number of games
tr['TmGame'] = 1
tr['OppGame'] = 1

# Add field for number of minutes
tr['TmMins'] = 40 + tr['NumOT'] * 5
tr['OppMins'] = tr['TmMins']

# Add field for Total Rebounds
tr['TmTR'] = tr['TmOR'] + tr['TmDR']
tr['OppTR'] = tr['OppOR'] + tr['OppDR']

# Count number of FGA2/FGM2
tr['TmFGM2'] = tr['TmFGM'] - tr['TmFGM3']
tr['TmFGA2'] = tr['TmFGA'] - tr['TmFGA3']
tr['OppFGM2'] = tr['OppFGM'] - tr['OppFGM3']
tr['OppFGA2'] = tr['OppFGA'] - tr['OppFGA3']

# Calculate field goal percentages in each game
tr['TmFGPct'] = tr['TmFGM'] / tr['TmFGA']
tr['TmFG3Pct'] = tr['TmFGM3'] / tr['TmFGA3']
tr['TmFG2Pct'] = tr['TmFGM2'] / tr['TmFGA2']
tr['TmFTPct'] = tr['TmFTM'] / tr['TmFTA']
tr['OppFGPct'] = tr['OppFGM'] / tr['OppFGA']
tr['OppFG3Pct'] = tr['OppFGM3'] / tr['OppFGA3']
tr['OppFG2Pct'] = tr['OppFGM2'] / tr['OppFGA2']
tr['OppFTPct'] = tr['OppFTM'] / tr['OppFTA']

# Calculate game margin
tr['TmMargin'] = tr['TmPF'] - tr['OppPF']
tr['OppMargin'] = -tr['TmMargin']

# Add field for number of possessions (NCAA NET method)
tr['TmPoss'] = tr['TmFGA'] \
                - tr['TmOR'] \
                + tr['TmTO'] \
                + .475 * tr['TmFTA']
tr['OppPoss'] = tr['OppFGA'] \
                - tr['OppOR'] \
                + tr['OppTO'] \
                + .475 * tr['OppFTA']

# Calculate per-40 and per-poss metrics for each game in tr
for x in {'Opp', 'Tm'}:
    for column in metrics:
        tr[x + column + 'per40'] = tr[x + column] / tr[x + 'Mins'] * 40
        tr[x + column + 'perPoss'] = tr[x + column] / tr[x + 'Poss']
del column, x

# Get seeds into tr dataframe
tourneyseeds = pd.read_csv(
    filepath_or_buffer='/Users/Ryan/Google Drive/ncaa-basketball-data/2018-kaggle-update/NCAATourneySeeds.csv'
)
tourneyseeds.loc[tourneyseeds['Seed'].str.len() == 4,'TmPlayInTeam'] = True
tourneyseeds['Seed'] = tourneyseeds['Seed'].str[1:3].astype('int')
tourneyseeds = tourneyseeds.rename(columns = {'TeamID':'TmID','Seed':'TmTourneySeed'})
tr = pd.merge(tr,
              tourneyseeds,
              how = 'left',
              on = ['Season','TmID'])
tourneyseeds = tourneyseeds.rename(columns = {
                    'TmPlayInTeam':'OppPlayInTeam',
                    'TmID':'OppID',
                    'TmTourneySeed':'OppTourneySeed'
                    })
tr = pd.merge(tr,
              tourneyseeds,
              how = 'left',
              on = ['Season','OppID'])

# Rename in prep for getting in to seasontourney
tourneyseeds = tourneyseeds.rename(columns = {
                    'OppPlayInTeam':'PlayInTeam',
                    'OppID':'TmID',
                    'OppTourneySeed':'TourneySeed'
                    })

    
# TODO un-limit the TR dataframe  
# Drop play-in-games
tr['PlayInGame'] = 0
tr.loc[(tr['TmPlayInTeam'] == True) & (tr['OppPlayInTeam'] ==  True),'PlayInGame'] = 1
#tr = tr.loc[tr['PlayInGame'] != True ]

# Create seasontourney dataframe and summarize some data
seasontourney = tr.groupby(['TmID', 'Season'])[['TmGame','TmWin','PlayInGame']].sum().reset_index()
seasontourney = seasontourney.rename(columns = {
                    'TmWin':'TourneyWin',
                    'TmGame':'TourneyGame'})
    
# Get seed information into seasontourney dataframe
seasontourney = pd.merge(
        left = seasontourney,
        right = tourneyseeds,
        how = 'inner',
        on = ['Season','TmID'])
del tourneyseeds

seasontourney['PlayInWin'] = 0
seasontourney.loc[(seasontourney['PlayInTeam'] == True) & 
                  (seasontourney['TourneyGame'] > 1)
                  ,'PlayInWin'] = 1
seasontourney['TourneyGame'] = seasontourney['TourneyGame'] - seasontourney['PlayInGame']
seasontourney['TourneyWin'] = seasontourney['TourneyWin'] - seasontourney['PlayInWin']


seasonteams_out = pd.merge(
                    left = seasonteams_out,
                    right = seasontourney,
                    how = 'left',
                    on = ['Season','TmID'])


# Get round information into seasontourney
seasonteams_out['TourneyResultStr'] = '-'
seasonteams_out.loc[seasonteams_out['TourneyWin'] == 6,'TourneyResultStr'] = 'Champion'
seasonteams_out.loc[seasonteams_out['TourneyWin'] == 5,'TourneyResultStr'] = 'Runner Up'
seasonteams_out.loc[seasonteams_out['TourneyWin'] == 4,'TourneyResultStr'] = 'Final 4'
seasonteams_out.loc[seasonteams_out['TourneyWin'] == 3,'TourneyResultStr'] = 'Elite 8'
seasonteams_out.loc[seasonteams_out['TourneyWin'] == 2,'TourneyResultStr'] = 'Sweet 16'
seasonteams_out.loc[seasonteams_out['TourneyWin'] == 1,'TourneyResultStr'] = 'Rnd of 32'
seasonteams_out.loc[seasonteams_out['TourneyWin'] == 0,'TourneyResultStr'] = 'Rnd of 64'




# Write output
createreplacecsv('/Users/Ryan/Google Drive/ncaa-basketball-data/rsg.csv',rsg_out)

createreplacecsv('/Users/Ryan/Google Drive/ncaa-basketball-data/seasonteams.csv',seasonteams_out)

end = time.time()
printtime('Post-Write time: ',end - begin)
del end, begin

#inclheaders = ['Season','TmName']
#for metric in rankmetrics:
#    inclheaders.append('Rank_' + metric)
#
#z = st_workingseason[inclheaders]



#testst = seasonteams.loc[seasonteams['TmName'] == 'Florida'][['Season',
#                                                            'TmName',
#                                                            'TmGame',
#                                                            'TmMarginper40',
#                                                            'OA_TmMarginper40',
#                                                            'OA_TmPFper40',
#                                                            'OA_OppPFper40']]
#
#testrsg = rsg.loc[(rsg['Season'] == 2019) & 
#                  (rsg['TmName'] == 'Florida')][[
#                          'GameDate'
#                          , 'TmName'
#                          , 'OppName'
#                          , 'TmPF'
#                          , 'TmMargin'
##                          , 'GameOppAdj_TmPF'
#                          , 'OppPF'
#                          , 'OppMargin'
##                          , 'GameOppAdj_OppPF'
#                          ]]


########################################
## Seasonteams prep for Season Table Dashboard
########################################


# TODO Deal with these later
#TourneyResults2017 = pd.read_csv(filepath_or_buffer = '/Users/Ryan/Google Drive/HistoricalNCAAData/2017TournamentResults.csv')
#rsgcDates = pd.read_csv(filepath_or_buffer = '/Users/Ryan/Google Drive/HistoricalNCAAData/rsgcDates.csv')
#TourneyGamesDates = pd.read_csv(filepath_or_buffer = '/Users/Ryan/Google Drive/HistoricalNCAAData/TourneyGamesDates.csv')
#TourneySlots = pd.read_csv(filepath_or_buffer = '/Users/Ryan/Google Drive/HistoricalNCAAData/TourneySlots.csv')




##
#
#
#
#
# # Merge results files
# del TourneyResults2017['Unnamed: 0']
# TourneyResults2017 = TourneyResults2017.rename(columns = {'Wteam':'Team_Name'})
# TourneyResults2017 = pd.merge(TourneyResults2017,Teams,on=['Team_Name'],how='left')
# del TourneyResults2017['Team_Name']
# TourneyResults2017 = TourneyResults2017.rename(columns = {'Team_Id':'Wteam'})
# TourneyResults2017 = TourneyResults2017.rename(columns = {'Lteam':'Team_Name'})
# TourneyResults2017 = pd.merge(TourneyResults2017,Teams,on=['Team_Name'],how='left')
# TourneyResults2017 = TourneyResults2017.rename(columns = {'Team_Id':'Lteam'})
# del TourneyResults2017['Team_Name']
# del TourneyResults2017['Season_x'], TourneyResults2017['Season_y']
# TourneyResults2017['Season'] = 2017
# TourneyResults = TourneyResults.append(TourneyResults2017)
# del TourneyResults2017
#
# # Pull seeds into results
# TourneySeeds = TourneySeeds.rename(columns = {'Team':'Wteam'})
# TourneyResults = pd.merge(TourneyResults,TourneySeeds,on=['Season','Wteam'])
# TourneySeeds = TourneySeeds.rename(columns = {'Wteam':'Team'})
# TourneyResults = TourneyResults.rename(columns = {'Seed':'WteamSeed'})
# TourneySeeds = TourneySeeds.rename(columns = {'Team':'Lteam'})
# TourneyResults = pd.merge(TourneyResults,TourneySeeds,on=['Season','Lteam'])
# TourneySeeds = TourneySeeds.rename(columns = {'Lteam':'Team_Id'})
# TourneyResults = TourneyResults.rename(columns = {'Seed':'LteamSeed'})
#
# # Play-in game flag, seeds
# TourneyResults['PlayInFlag'] = np.where((TourneyResults['WteamSeed'].str.len() == 4) & (TourneyResults['LteamSeed'].str.len() == 4), 'Y', '')
# TourneyResults['WteamSeed'] = pd.to_numeric(TourneyResults['WteamSeed'].str[1:3])
# TourneyResults['LteamSeed'] = pd.to_numeric(TourneyResults['LteamSeed'].str[1:3])
# TourneyResults = TourneyResults[TourneyResults['PlayInFlag']!='Y']
#
# # Get num of wins for team in tourney
# TourneyWins = TourneyResults.groupby(['Wteam','Season'])[['Wscore']].agg('count').reset_index()
# TourneyWins = TourneyWins.rename(columns = {'Wscore':'TourneyWins','Wteam':'Team_Id'})
# TourneyLosses = TourneyResults.groupby(['Lteam','Season'])[['Lscore']].agg('count').reset_index()
# TourneyLosses = TourneyLosses.rename(columns = {'Lscore':'TourneyLosses','Lteam':'Team_Id'})
# SeasonTeams = pd.merge(TourneyWins, SeasonTeams,on=['Team_Id','Season'],how='right')
# SeasonTeams = pd.merge(TourneyLosses, SeasonTeams,on=['Team_Id','Season'],how='right')
# SeasonTeams.fillna(0,inplace=True)
# del TourneyWins, TourneyLosses
# SeasonTeams['TourneyGames'] = SeasonTeams['TourneyWins'] + SeasonTeams['TourneyLosses']
# SeasonTeams['TourneyApp'] = np.where((SeasonTeams['TourneyGames'] >= 1), 'Y', '')
#
# # Get Tourney Seeds into SeasonTeams
# SeasonTeams = pd.merge(TourneySeeds, SeasonTeams,on=['Team_Id','Season'],how='right')
# SeasonTeams['Seed'] = pd.to_numeric(SeasonTeams['Seed'].str[1:3])
# SeasonTeams['Seed'][(SeasonTeams['TourneyApp'] != 'Y')] = ''
# SeasonTeams['Seed'] = pd.to_numeric(SeasonTeams['Seed'])
#
# # Get 2018 Tourney Results In
# # =============================================================================
# TourneySeeds18 = pd.read_csv(filepath_or_buffer = '/Users/Ryan/Google Drive/HistoricalNCAAData/18TourneySeeds.csv')
# SeasonTeams = pd.merge(SeasonTeams, TourneySeeds18, on=['Season','Team_Name'],how='left')
# SeasonTeams['Seed'] = SeasonTeams['Seed_x'].fillna(SeasonTeams['Seed_y'])
# del SeasonTeams['Seed_x'], SeasonTeams['Seed_y']
# SeasonTeams['TourneyApp'] = np.where((SeasonTeams['Seed'] >= 1), 'Y', '')
# SeasonTeams['Season'] = pd.to_numeric(SeasonTeams['Season'])
# #SeasonTeams['TourneyApp_x'] = SeasonTeams['TourneyApp_x'].replace(r'\s+', np.nan, regex=True)
# #SeasonTeams['TourneyApp'] = SeasonTeams['TourneyApp_x'].fillna(SeasonTeams['TourneyApp_y'])
# del SeasonTeams['TourneyApp_x'], SeasonTeams['TourneyApp_y']
# del TourneySeeds18
# # =============================================================================
# ####################################################
# ####################################################
# ####################################################
#
# ####################################################
# #######Tourney Games for Tableau    ################
# ####################################################
#
# # Merge day 0 into rsgc
# TourneyResults = pd.merge(TourneyResults,Seasons,on='Season',)
# del TourneyResults['Regionw']
# del TourneyResults['Regionx']
# del TourneyResults['Regiony']
# del TourneyResults['Regionz']
#
# # NOTE: This takes tons of HP so it is commented for now
# #TourneyResults['Dayzero'] =  pd.to_datetime(TourneyResults['Dayzero'])
# #temp = TourneyResults['Daynum'].apply(pd.np.ceil).apply(lambda x: pd.Timedelta(x, unit='D'))
# #TourneyResults['Date'] = TourneyResults['Dayzero'] + temp
# ## Export Date Info
# #TourneyGamesDates = TourneyResults['Date']
# #os.chdir('/Users/Ryan/Desktop/HistoricalNCAAData/')
# #TourneyGamesDates.to_csv('TourneyGamesDates.csv',header=True)
# del TourneyResults['Dayzero']
# del TourneyResults['Daynum']
# del TourneyGamesDates['Unnamed: 0']
# TourneyResults = pd.merge(TourneyResults,TourneyGamesDates,left_index=True, right_index=True)
# del TourneyGamesDates
# TourneyResults['Date'] = TourneyResults['Date_x'].fillna(TourneyResults['Date_y'])
# del TourneyResults['Date_x'], TourneyResults['Date_y'], TourneyResults['Wloc'], TourneyResults['PlayInFlag']
#
# # Rename things and append
# TourneyResults = TourneyResults.rename(columns = {'Wteam':'Team_Id','Wscore':'TeamScore','Lscore':'OpponentScore'})
# del Teams['Season']
# TourneyResults = pd.merge(TourneyResults, Teams ,on=['Team_Id'],how='left')
# TourneyResults = TourneyResults.rename(columns = {'Team_Name':'TeamName','Team_Id':'Team','Lteam':'Team_Id'})
# TourneyResults = pd.merge(TourneyResults, Teams ,on=['Team_Id'],how='left')
# TourneyResults = TourneyResults.rename(columns = {'Team_Name':'OpponentName','Team_Id':'Opponent'})
# TourneyResults['Result'] = 'W'
#
# # Make Losses Df
# TourneyLosses = pd.DataFrame(columns = ['Date'])
# TourneyLosses['Date'] = TourneyResults['Date']
# TourneyLosses['Team'] = TourneyResults['Opponent']
# TourneyLosses['TeamName'] = TourneyResults['OpponentName']
# TourneyLosses['Opponent'] = TourneyResults['Team']
# TourneyLosses['OpponentName'] = TourneyResults['TeamName']
# TourneyLosses['TeamScore'] = TourneyResults['OpponentScore']
# TourneyLosses['OpponentScore'] = TourneyResults['TeamScore']
# TourneyLosses['Numot'] = TourneyResults['Numot']
# TourneyLosses['Season'] = TourneyResults['Season']
# TourneyLosses['WteamSeed'] = TourneyResults['LteamSeed']
# TourneyLosses['LteamSeed'] = TourneyResults['WteamSeed']
# TourneyLosses['Result'] = 'L'
#
# # Combine them
# TourneyResults = TourneyResults.append(TourneyLosses)
# del TourneyLosses
# TourneyResults['TourneyGame'] = 'Y'
#
# rsgc = rsgc.append(TourneyResults)
# ####################################################
# ####################################################
# ####################################################
#
#
# # Test DF
# Test = SeasonTeams.loc[(SeasonTeams['Season'] == 2018) & (SeasonTeams['TourneyApp'] == 'Y')]
# TestYear = rsgc.loc[(rsgc['Season'] == 2018)]
#
# middle = time.time()
# printtime('Pre-Write time: ',middle-begin)
#
#
#
