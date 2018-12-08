#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 11 17:38:36 2018

@author: Ryan
"""

# import libraries
import pandas as pd
#from datetime import datetime, timedelta

import os
import numpy as np
#from urllib.request import urlopen
#from bs4 import BeautifulSoup
from scipy.stats import rankdata
import time
#import pygsheets

begin = time.time()

# Import raw data files
RSGames = pd.read_csv(filepath_or_buffer = '/Users/Ryan/Google Drive/HistoricalNCAAData/RegularSeasonCompactResults.csv')
Teams = pd.read_csv(filepath_or_buffer = '/Users/Ryan/Google Drive/HistoricalNCAAData/Teams.csv')
Seasons = pd.read_csv(filepath_or_buffer = '/Users/Ryan/Google Drive/HistoricalNCAAData/Seasons.csv')
TourneySlots = pd.read_csv(filepath_or_buffer = '/Users/Ryan/Google Drive/HistoricalNCAAData/TourneySlots.csv')
TourneySeeds = pd.read_csv(filepath_or_buffer = '/Users/Ryan/Google Drive/HistoricalNCAAData/TourneySeeds.csv')
TourneyResults = pd.read_csv(filepath_or_buffer = '/Users/Ryan/Google Drive/HistoricalNCAAData/TourneyCompactResults.csv')
TourneyResults2017 = pd.read_csv(filepath_or_buffer = '/Users/Ryan/Google Drive/HistoricalNCAAData/2017TournamentResults.csv')
RSGamesDates = pd.read_csv(filepath_or_buffer = '/Users/Ryan/Google Drive/HistoricalNCAAData/RSGamesDates.csv')
TourneyGamesDates = pd.read_csv(filepath_or_buffer = '/Users/Ryan/Google Drive/HistoricalNCAAData/TourneyGamesDates.csv')

# Delete Hardin-Simmons
#Teams = Teams[Teams['Team_Name']!='Hardin-Simmons']

# Merge team names into games
RSGames = RSGames.rename(columns = {'Wteam':'Team_Id'})
RSGames = pd.merge(RSGames,Teams,on='Team_Id',)
RSGames = RSGames.rename(columns = {'Team_Name':'WTeamName'})
RSGames = RSGames.rename(columns = {'Team_Id':'Wteam'})

RSGames = RSGames.rename(columns = {'Lteam':'Team_Id'})
RSGames = pd.merge(RSGames,Teams,on='Team_Id',)
RSGames = RSGames.rename(columns = {'Team_Name':'LTeamName'})
RSGames = RSGames.rename(columns = {'Team_Id':'Lteam'})

# Merge day 0 into RSGames
RSGames = pd.merge(RSGames,Seasons,on='Season',)
del RSGames['Regionw']
del RSGames['Regionx']
del RSGames['Regiony']
del RSGames['Regionz']

# Calculate game date

# NOTE: This takes tons of HP so it is commented for now
#RSGames['Dayzero'] =  pd.to_datetime(RSGames['Dayzero'])
#temp = RSGames['Daynum'].apply(pd.np.ceil).apply(lambda x: pd.Timedelta(x, unit='D'))
#RSGames['Date'] = RSGames['Dayzero'] + temp
## Export Date Info
#RSGamesDates = RSGames['Date']
#os.chdir('/Users/Ryan/Desktop/HistoricalNCAAData/')
#RSGamesDates.to_csv('RSGamesDates.csv',header=True)
del RSGames['Dayzero']
del RSGames['Daynum']
del RSGamesDates['Unnamed: 0']
RSGames = pd.merge(RSGames,RSGamesDates,left_index=True, right_index=True)
del RSGamesDates

###################################################
###### CURRENT YEAR SECTION########################
###################################################
# import libraries
from urllib.request import urlopen
from bs4 import BeautifulSoup

# Pull Web Data
webpage = "https://www.masseyratings.com/scores.php?s=298892&sub=11590&all=1"
page = urlopen(webpage)
soup = BeautifulSoup(page,'lxml')
NCAADataStr = str(soup.pre)
del webpage

# Import Data into DF
x = NCAADataStr.split('\n')
NCAADf = pd.DataFrame(x, columns = ['RawStr'])
del NCAADataStr

# Remove last 4 rows
NCAADf = NCAADf[:-4]

# Remove/replace strings
NCAADf['RawStr'].replace(regex=True,inplace=True,to_replace=r'&amp;',value=r'&')
NCAADf['RawStr'].replace(regex=True,inplace=True,to_replace=r'<pre>',value=r'')

# Split string into columns
NCAADf['Date'] = NCAADf['RawStr'].str[:10]
NCAADf['Team'] = NCAADf['RawStr'].str[10:36]
NCAADf['TeamScore'] = NCAADf['RawStr'].str[36:39]
NCAADf['Opponent'] = NCAADf['RawStr'].str[39:65]
NCAADf['OpponentScore'] = NCAADf['RawStr'].str[65:68]
NCAADf['GameOT'] = pd.to_numeric(NCAADf['RawStr'].str[70:71],errors='coerce')
NCAADf['GameOT'] = np.nan_to_num(NCAADf['GameOT'])
del NCAADf['RawStr']

del x

# Strip Whitespaces
NCAADf['Date'] = NCAADf['Date'].str.strip()
NCAADf['TeamScore'] = NCAADf['TeamScore'].str.strip()
NCAADf['Team'] = NCAADf['Team'].str.strip()
NCAADf['Opponent'] = NCAADf['Opponent'].str.strip()
NCAADf['OpponentScore'] = NCAADf['OpponentScore'].str.strip()

# Change column types
NCAADf[['TeamScore','OpponentScore']] = NCAADf[['TeamScore','OpponentScore']].apply(pd.to_numeric)
#NCAADf['Date'] = NCAADf['Date'].dt.date

# Calculate Margin and team locations
NCAADf['Location'] = ''
NCAADf['Season'] = 2018
NCAADf.loc[(NCAADf['Team'].str[:1] == '@'), 'Location'] = 'H'
NCAADf.loc[(NCAADf['Opponent'].str[:1] == '@'), 'Location'] = 'A'
NCAADf.loc[(NCAADf['Opponent'].str[:1] != '@') & (NCAADf['Team'].str[:1] != '@'), 'Location'] = 'N'

# Remove @
NCAADf['Team'].replace(regex=True,inplace=True,to_replace=r'@',value=r'')
NCAADf['Opponent'].replace(regex=True,inplace=True,to_replace=r'@',value=r'')


# Rename columns for merge
NCAADf = NCAADf.rename(columns = {'TeamScore':'Wscore','OpponentScore':'Lscore','GameOT':'Numot','Location':'Wloc'})

# Get team IDs into NCAADf
NCAADf = NCAADf.rename(columns = {'Team':'Team_Name'})
NCAADf = pd.merge(NCAADf,Teams,on='Team_Name')
NCAADf = NCAADf.rename(columns = {'Team_Name':'WTeamName','Team_Id':'Wteam'})

NCAADf = NCAADf.rename(columns = {'Opponent':'Team_Name'})
NCAADf = pd.merge(NCAADf,Teams,on='Team_Name',)
NCAADf = NCAADf.rename(columns = {'Team_Name':'LTeamName','Team_Id':'Lteam'})


# Append to RSGames
RSGames = RSGames.append(NCAADf)
del NCAADf

###################################################
###################################################
###################################################

# =============================================================================
# # Add MArch 11 games
# March11Games = pd.read_csv(filepath_or_buffer = '/Users/Ryan/Google Drive/HistoricalNCAAData/March11Games.csv')
# RSGames = RSGames.append(March11Games)
# del March11Games
# 
# =============================================================================

# Calculate PF per 40 mins for each game
RSGames['GameMins'] = 40 + RSGames['Numot']*5
RSGames['WTeamPFper40'] = RSGames['Wscore']*40/RSGames['GameMins']
RSGames['LTeamPFper40'] = RSGames['Lscore']*40/RSGames['GameMins']

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
WinsStats = RSGames.groupby(['Wteam', 'Season'])[['Wscore','Lscore','GameMins']].agg(['sum','count']).reset_index()
LossesStats = RSGames.groupby(['Lteam', 'Season'])[['Wscore','Lscore','GameMins']].agg(['sum','count']).reset_index()
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
RSGames = RSGames.rename(columns = {'Wteam':'Team_Id'})
RSGames = pd.merge(RSGames,SeasonTeams[['Season','Team_Id','PFper40','PAper40','WinPct']],on=['Season','Team_Id'],how='left')
RSGames = RSGames.rename(columns = {'Team_Id':'Wteam','PFper40':'WteamAvgPFper40','PAper40':'WteamAvgPAper40','WinPct':'WteamWinPct'})

RSGames = RSGames.rename(columns = {'Lteam':'Team_Id'})
RSGames = pd.merge(RSGames,SeasonTeams[['Season','Team_Id','PFper40','PAper40','WinPct']],on=['Season','Team_Id'],how='left')
RSGames = RSGames.rename(columns = {'Team_Id':'Lteam','PFper40':'LteamAvgPFper40','PAper40':'LteamAvgPAper40','WinPct':'LteamWinPct'})

# Calculate game scores and stuff for games
RSGames['WTeamGameOffScore'] = RSGames['WTeamPFper40'] - RSGames['LteamAvgPAper40']
RSGames['LTeamGameOffScore'] = RSGames['LTeamPFper40'] - RSGames['WteamAvgPAper40']
RSGames['WTeamGameDefScore'] = RSGames['LteamAvgPFper40'] - RSGames['LTeamPFper40']
RSGames['LTeamGameDefScore'] = RSGames['WteamAvgPFper40'] - RSGames['WTeamPFper40']
RSGames['WTeamGameOAM'] = RSGames['WTeamGameOffScore'] + RSGames['WTeamGameDefScore']
RSGames['LTeamGameOAM'] = RSGames['LTeamGameOffScore'] + RSGames['LTeamGameDefScore']

# Bring the game info back to the aggregated table
WinsStats = RSGames.groupby(['Wteam', 'Season'])[['WTeamGameOffScore','WTeamGameDefScore','WTeamGameOAM','LteamWinPct']].agg(['sum']).reset_index()
LossesStats = RSGames.groupby(['Lteam', 'Season'])[['LTeamGameOffScore','LTeamGameDefScore','LTeamGameOAM','WteamWinPct']].agg(['sum']).reset_index()
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
RSGames = RSGames.rename(columns = {'Wteam':'Team_Id'})
RSGames = pd.merge(RSGames,SeasonTeams[['Season','Team_Id','OWP']],on=['Season','Team_Id'],how='left')
RSGames = RSGames.rename(columns = {'Team_Id':'Wteam','OWP':'WteamOWP'})

RSGames = RSGames.rename(columns = {'Lteam':'Team_Id'})
RSGames = pd.merge(RSGames,SeasonTeams[['Season','Team_Id','OWP']],on=['Season','Team_Id'],how='left')
RSGames = RSGames.rename(columns = {'Team_Id':'Lteam','OWP':'LteamOWP'})

# Bring the game info back to the aggregated table
WinsStats = RSGames.groupby(['Wteam', 'Season'])[['LteamOWP']].agg(['sum']).reset_index()
LossesStats = RSGames.groupby(['Lteam', 'Season'])[['WteamOWP']].agg(['sum']).reset_index()
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
del RSGames['WteamOWP'], RSGames['LteamOWP'], RSGames['LteamWinPct'], RSGames['WteamWinPct']


######################################
## CONFIGURE RSGAMES FOR TABLEAU#####
######################################
RSGames = RSGames.rename(columns = {'Wteam':'Team','Wscore':'TeamScore','Lteam':'Opponent','Lscore':'OpponentScore'})
RSGames = RSGames.rename(columns = {'WTeamName':'TeamName','LTeamName':'OpponentName','Wloc':'Location'})
del RSGames['WTeamPFper40'], RSGames['LTeamPFper40'], RSGames['WteamAvgPFper40'], RSGames['WteamAvgPAper40']
del RSGames['LteamAvgPFper40'], RSGames['LteamAvgPAper40']
RSGames = RSGames.rename(columns = {'WTeamGameOffScore':'TeamOffScore','WTeamGameDefScore':'TeamDefScore','WTeamGameOAM':'TeamOAM'})
RSGames['Result'] = 'W'

# Make Losses DF
RSLosses = pd.DataFrame(columns = ['Date'])
RSLosses['Date'] = RSGames['Date']
RSLosses['Team'] = RSGames['Opponent']
RSLosses['TeamName'] = RSGames['OpponentName']
RSLosses['Opponent'] = RSGames['Team']
RSLosses['OpponentName'] = RSGames['TeamName']
RSLosses['TeamScore'] = RSGames['OpponentScore']
RSLosses['OpponentScore'] = RSGames['TeamScore']
RSLosses['Location'] = RSGames['Location']
RSLosses.loc[(RSLosses['Location'] == 'H'), 'Location'] = 'AA'
RSLosses.loc[(RSLosses['Location'] == 'A'), 'Location'] = 'H'
RSLosses.loc[(RSLosses['Location'] == 'AA'), 'Location'] = 'A'
RSLosses['TeamOffScore'] = RSGames['LTeamGameOffScore']
RSLosses['TeamDefScore'] = RSGames['LTeamGameDefScore']
RSLosses['TeamOAM'] = RSGames['LTeamGameOAM']
RSLosses['Numot'] = RSGames['Numot']
RSLosses['Season'] = RSGames['Season']
RSLosses['Result'] = 'L'
del RSGames['LTeamGameOffScore'], RSGames['LTeamGameDefScore'], RSGames['LTeamGameOAM'], RSGames['GameMins']

# Combine them
RSGames = RSGames.append(RSLosses)
del RSLosses

# Rank Game Percentiles
RSGames['OffPercentile'] = 1 - (len(RSGames)-rankdata(RSGames['TeamOffScore'],method='min'))/len(RSGames)
RSGames['DefPercentile'] = 1 - (len(RSGames)-rankdata(RSGames['TeamDefScore'],method='min'))/len(RSGames)
RSGames['OAMPercentile'] = 1 - (len(RSGames)-rankdata(RSGames['TeamOAM'],method='min'))/len(RSGames)

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
RSGames = RSGames.rename(columns = {'Opponent':'Team_Id'})
RSGames = pd.merge(RSGames,SeasonTeams[['Season','Team_Id','OAMRank']],on=['Season','Team_Id'],how='left')
RSGames = RSGames.rename(columns = {'Team_Id':'Opponent','OAMRank':'OpponentRank'})

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

# Merge day 0 into RSGames
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

RSGames = RSGames.append(TourneyResults)
####################################################
####################################################
####################################################


# Test DF
Test = SeasonTeams.loc[(SeasonTeams['Season'] == 2018) & (SeasonTeams['TourneyApp'] == 'Y')]
TestYear = RSGames.loc[(RSGames['Season'] == 2018)]

middle = time.time()
print ('Pre-Write time: ' + str(round(middle - begin,2)))

# Write File
os.chdir('/Users/Ryan/Google Drive/HistoricalNCAAData/')
SeasonTeams.to_excel(excel_writer = 'SeasonTeams.xlsx', sheet_name = 'SeasonTeams', engine='xlsxwriter')
#RSGames.to_excel(excel_writer = 'RSGames.xlsx', sheet_name = 'RSGames', engine='xlsxwriter')
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
## Write RSGames
#sh = gc.open('PythonNCAAData1')
#wks = sh.sheet1
#wks.cols = 21
#wks.rows = 10000
#wks.set_dataframe(RSGames.iloc[0:9999,:],(1,1))
#end1 = time.time()
#print ('Post-RSGamesWrite time: ' + str(round(end1 - end1,2)))


end = time.time()


print ('Post-Write time: ' + str(round(end - middle,2)))
del begin, middle, end
