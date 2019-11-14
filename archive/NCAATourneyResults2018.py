# -*- coding: utf-8 -*-


import pandas as pd
from UDFs import createreplacecsv, printtime
from bs4 import BeautifulSoup
from urllib.request import urlopen
import numpy as np


teamspellings = pd.read_csv(
    filepath_or_buffer='/Users/Ryan/Google Drive/ncaa-basketball-data/2018-kaggle/TeamSpellings.csv',
    encoding="ISO-8859-1")

# Pull Web Data
webpage = "https://www.masseyratings.com/scores.php?s=298892&sub=298892&all=1"
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
rsg_curr.loc[rsg_curr['RawStr'].str.contains('NCAA') == True,'TourneyGame'] = 1
rsg_curr['GameOT'] = np.nan_to_num(rsg_curr['GameOT'])
del rsg_curr['RawStr']

# Strip Whitespaces
rsg_curr['GameDate'] = rsg_curr['GameDate'].str.strip()
rsg_curr['TeamScore'] = rsg_curr['TeamScore'].str.strip()
rsg_curr['Team'] = rsg_curr['Team'].str.strip()
rsg_curr['Opponent'] = rsg_curr['Opponent'].str.strip()
rsg_curr['OpponentScore'] = rsg_curr['OpponentScore'].str.strip()

rsg_curr = rsg_curr.loc[rsg_curr['TourneyGame'] == 1]

# Change column types
rsg_curr[['TeamScore',
          'OpponentScore']] = rsg_curr[['TeamScore',
                                        'OpponentScore']].apply(pd.to_numeric)
rsg_curr['GameDate'] = pd.to_datetime(rsg_curr['GameDate'])

# Calculate Margin and team locations
rsg_curr['WLoc'] = ''
rsg_curr['Season'] = 2018
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


# Drop teamnamespelling version of name
del rsg_curr['TmName'], rsg_curr['OppName']

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

createreplacecsv('/Users/Ryan/Google Drive/ncaa-basketball-data/2018TourneyResults.csv',rsg_curr)
