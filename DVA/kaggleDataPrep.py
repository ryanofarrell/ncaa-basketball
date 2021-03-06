#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 11 17:38:36 2018

@author: Ryan
"""
import pandas as pd
import numpy as np
from datetime import datetime


class timer:
    def __init__(self):
        pass

    def start(self):
        self.starttime = datetime.now()
        self.prevsplit = datetime.now()
        print('Timer started...')

    def split(self,descr = 'Time since prev split: '):
        assert type(descr) == str, 'Please pass a string'
        self.now = datetime.now()
        print(descr +  str(self.now - self.prevsplit))
        self.prevsplit = datetime.now()
    def end(self):
        self.now = datetime.now()
        print('Time since start: ' +  str(self.now - self.starttime))

def createreplacecsv(csvtoreplace,dataframetowrite):
    # Assert cavtoreplace is a string and ends in csv
    assert type(csvtoreplace) == str, 'First argument is not a string'
    assert csvtoreplace[-4:] == '.csv', 'First argument is not a csv file'

    # Create new file string
    import re
    regex = '^(\/(?:.+\/)*)(.+)(.csv)'
    currcsvpath = re.match(regex,csvtoreplace).group(1)
    currcsvname = re.match(regex,csvtoreplace).group(2)

    from datetime import datetime
    now = str(datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
    newcsvpath = currcsvpath + 'archive/'
    newcsvname = currcsvname + '_ReplacedOn_' + now + '.csv'

    # Assert the archive path exists
    import os
    assert os.path.exists(newcsvpath), "No 'archive' folder in the provided path"

    # Assert dataframetowrite is a dataframe
    import pandas as pd
    assert type(dataframetowrite) == pd.core.frame.DataFrame, 'Second argument is not a DataFrame'


    print('\nWriting: ' + currcsvname + '.csv...')

    # Move current file to archive folder, rename file
    try:
        os.rename(csvtoreplace, newcsvpath + newcsvname)
        print('Archived current ' + currcsvname + '.csv into archive folder...')
    except FileNotFoundError:
        print('No file archived; no file named ' + currcsvname + '.csv at given path')

    dataframetowrite.to_csv(currcsvpath + currcsvname + '.csv', index=False)
    print('Success! Wrote dataframe to ' + currcsvname + '.csv')

# TODO Poll data
# TODO weekly tracker of poll info/general ranks
# TODO RPI
# TODO check if there are new games or if this is a big ole waste of time


print('Running season calculations\n')

timer = timer()
timer.start()
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
    if st_workingseason[
            otherprefix + coremetric + 'perGame'].isnull().values.any():
        st_workingseason['OA_' + prefix + coremetric + 'perGame'] = np.NaN
        st_workingseason['OA_' + prefix + coremetric + 'per40'] = np.NaN
        st_workingseason['OA_' + prefix + coremetric + 'perPoss'] = np.NaN
        #print('Skipped: ' + str(workingseason) + ' - ' + prefix + coremetric)
    # Otherwise, if there is data to use, opponentadjust
    else:
        # Into temo_st_currseason, get the team names, season, and
        # the opposite side's metric of what is being adjusted
        # For example, when OAing TmPFper40, temp_iteams will contain the
        # team's OppPFper40
        # This is used later for comparing a team's performance to the
        # opponent's average
        temp_st_workingseason = st_workingseason[[
            'TmName', 'Season', otherprefix + coremetric + 'perGame']]

        # Rename my opponent's metric to say it's *their* average <metric>
        # Rename to OppAvg_OppPFper40
        # (it's my opponent's average opponents (me) PF per 40)
        temp_st_workingseason = temp_st_workingseason.rename(
            columns={otherprefix + coremetric + 'perGame':
                     'OppAvg_' + otherprefix + coremetric + 'perGame'})

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
        rsg_workingseason['GameOppAdj_' + prefix + coremetric] = \
            rsg_workingseason[prefix + coremetric] - \
            rsg_workingseason['OppAvg_' + otherprefix + coremetric + 'perGame']

        del rsg_workingseason['OppAvg_' + otherprefix + coremetric + 'perGame']
        # Inverse for when you start with an opponent
        # to make positive numbers good
        if prefix == 'Opp':
            rsg_workingseason['GameOppAdj_' + prefix + coremetric] = \
                rsg_workingseason['GameOppAdj_' + prefix + coremetric] * -1

        # In iteamstemp, sum the opponent-adjusted metric and get a new average
        # Example, sum(GameOppAdj_TmPFper40) gets you the TOTAL OA_PFper40
        temp_st_workingseason = rsg_workingseason.groupby(
                ['TmName', 'Season'])['GameOppAdj_'
                                      + prefix
                                      + coremetric].sum().reset_index()

        # bring that value back into iteams, adjust for a 40-min game
        st_workingseason = pd.merge(st_workingseason, temp_st_workingseason,
                                    on=['TmName', 'Season'],
                                    how='left')
        st_workingseason = st_workingseason.rename(
            columns={'GameOppAdj_' + prefix + coremetric:
                     'OA_' + prefix + coremetric})

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
        del st_workingseason['OA_' + prefix + coremetric]
        del rsg_workingseason['GameOppAdj_' + prefix + coremetric]

#        print('Success: ' + str(workingseason) + ' - ' + prefix + coremetric)

#    del iteams['TmName_y']
###############################################################################
###############################################################################
###############################################################################
###############################################################################


# Future metric init
metrics = ['PF', 'Margin',
           'FGM', 'FGA',
           'FG3M', 'FG3A',
           'FG2M', 'FG2A',
           'FTA', 'FTM',
           'Ast', 'ORB',
           'DRB', 'TRB',
           'TO', 'Stl',
           'Blk', 'Foul']

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

descendingrankmetrics = [
    'TmSoS', 
    'TmTSP', 
    'TmAstTORatio',
    'TmFTPct',
    'TmFG2Pct',
    'TmFG3Pct',
    'TmFGPct',
    'TmORBPct',
    'TmDRBPct',
    'TmOE',
    'TmNE'
]
ascendingrankmetrics = [
    'OppTSP', 
    'OppAstTORatio',
    'OppFTPct',
    'OppFG2Pct',
    'OppFG3Pct',
    'OppFGPct',
    'OppORBPct',
    'OppDRBPct',
    'TmDE'
]
negativecoremetrics = ['Foul', 'TO']

# Positive:
# PF, Margin, FGM, FGA, FGM3, FGA3, FGM2, FGA2
# FTA, FTM, AST, OR, DR, TR, STL, BLK

# Foul, TO

# More = better: Tm * Positive
# More = Better: Opp * NEgative
# LEss = better: Tm * Negative
# LEss = Better: Opp * Positive

# Opp = -1, Negative = -1, Tm = 1, Pos = 1

for oa in {'', 'OA_'}:
    for prefix in {'Opp', 'Tm'}:
        for coremetric in metrics:
            for suffix in {'perGame', 'per40', 'perPoss'}:
                if prefix == 'Opp':
                    # Need special handling because OA stuff in inversed
                    if oa == 'OA_':
                        x = 1
                    else:
                        x = -1
                else:
                    x = 1
                if coremetric in negativecoremetrics:
                    y = -1
                else:
                    y = 1
                z = x*y
                if z == 1:
                    descendingrankmetrics.append(oa + prefix + coremetric + suffix)
                else:
                    ascendingrankmetrics.append(oa + prefix + coremetric + suffix)
del oa, prefix, coremetric, suffix, negativecoremetrics, x, y, z

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
    # TODO fix the mixed dtypes warning
    readinseasonteams = pd.read_csv('/Users/Ryan/Google Drive/ncaa-basketball-data/seasonteams.csv')
    seasonteams_out = readinseasonteams.loc[readinseasonteams['Season'] < currseason]
    del readinseasonteams

    # Set the working rsg to just the current season's data
    rsg_tocalc = rsg_curr
    del rsg_curr

# Othewise if running all seasons
elif runall == True:

    # Set the loop of the rsg dataframe to all season
    seasonstoloop = list(range(1985,currseason+1))

    # Initialize rsg_out
    rsg_out = pd.DataFrame()

    # Initialize seasonteams_out
    seasonteams_out = pd.DataFrame()

    rsgd_prev = pd.read_csv(
        '../data/Stage2DataFiles/RegularSeasonDetailedResults.csv'
    )
    rsgc_prev = pd.read_csv(
            '../data/Stage2DataFiles/RegularSeasonCompactResults.csv'
    )
    seasons = pd.read_csv(
        filepath_or_buffer='../data/Stage2DataFiles/Seasons.csv'
    )


    # Merge in day 0 to rsgd, add days to day 0 to get date of game, delete extra columns
    rsgd_prev = pd.merge(rsgd_prev, seasons[['Season', 'DayZero']], on='Season')
    rsgd_prev['DayZero'] = pd.to_datetime(rsgd_prev['DayZero'], format='%m/%d/%Y')
    rsgd_prev['DayNum'] = pd.to_timedelta(rsgd_prev['DayNum'], unit='d')
    rsgd_prev['GameDate'] = rsgd_prev['DayZero'] + rsgd_prev['DayNum']
    del rsgd_prev['DayNum'], rsgd_prev['DayZero']

    # Merge in day 0 to rsgd, add days to day 0 to get date of game, delete extra columns
    rsgc_prev = pd.merge(rsgc_prev, seasons[['Season', 'DayZero']], on='Season')
    rsgc_prev['DayZero'] = pd.to_datetime(rsgc_prev['DayZero'], format='%m/%d/%Y')
    rsgc_prev['DayNum'] = pd.to_timedelta(rsgc_prev['DayNum'], unit='d')
    rsgc_prev['GameDate'] = rsgc_prev['DayZero'] + rsgc_prev['DayNum']
    del rsgc_prev['DayNum'], rsgc_prev['DayZero']

    rsg_prev = pd.merge(
        left = rsgc_prev,
        right = rsgd_prev,
        how = 'outer',
        on = list(rsgc_prev))
    del rsgc_prev, rsgd_prev

    # Rename all the columns...
    rsg_prev = rsg_prev.rename(columns={
                                'GameDate': 'GameDate',
                                'NumOT': 'GameOT',
                                'WTeamID': 'TmID',
                                'WScore': 'TmPF',
                                'WFGM': 'TmFGM',
                                'WFGA': 'TmFGA',
                                'WFGM2': 'TmFG2M',
                                'WFGA2': 'TmFG2A',
                                'WFGM3': 'TmFG3M',
                                'WFGA3': 'TmFG3A',
                                'WFTM': 'TmFTM',
                                'WFTA': 'TmFTA',
                                'WOR': 'TmORB',
                                'WDR': 'TmDRB',
                                'WTRB': 'TmTRB',
                                'WAst': 'TmAst',
                                'WStl': 'TmStl',
                                'WBlk': 'TmBlk',
                                'WTO': 'TmTO',
                                'WPF': 'TmFoul',
                                'WLoc': 'TmLoc',
                                'LTeamID': 'OppID',
                                'LScore': 'OppPF',
                                'LFGM': 'OppFGM',
                                'LFGA': 'OppFGA',
                                'LFGM2': 'OppFG2M',
                                'LFGA2': 'OppFG2A',
                                'LFGM3': 'OppFG3M',
                                'LFGA3': 'OppFG3A',
                                'LFTM': 'OppFTM',
                                'LFTA': 'OppFTA',
                                'LOR': 'OppORB',
                                'LDR': 'OppDRB',
                                'LTRB': 'OppTRB',
                                'LAst': 'OppAst',
                                'LStl': 'OppStl',
                                'LBlk': 'OppBlk',
                                'LTO': 'OppTO',
                                'LPF': 'OppFoul',
                                'LLoc': 'OppLoc'
                                })

    # Copy, rename, and append the other half of the games to rsg_prev
    lrsg_prev = rsg_prev.copy()
    newnames = pd.DataFrame(list(lrsg_prev),columns = ['OldName'])
    newnames['NewName'] = newnames['OldName']
    newnames.loc[newnames['OldName'].str[0:3] == 'Opp','NewName'] = 'Tm' + newnames['OldName'].str[3:]
    newnames.loc[newnames['OldName'].str[0:2] == 'Tm','NewName'] = 'Opp' + newnames['OldName'].str[2:]
    newnames = newnames.set_index('OldName')['NewName']
    lrsg_prev = lrsg_prev.rename(columns = newnames)
    lrsg_prev['TmLoc'] = 'N'
    lrsg_prev.loc[lrsg_prev['OppLoc'] == 'H', 'TmLoc'] = 'A'
    lrsg_prev.loc[lrsg_prev['OppLoc'] == 'A', 'TmLoc'] = 'H'
    del lrsg_prev['OppLoc']
    rsg_prev = rsg_prev.append(lrsg_prev)
    del lrsg_prev, newnames

    # Handle column differences
    rsg_prev['TmFG2A'] = rsg_prev['TmFGA'] - rsg_prev['TmFG3A']
    rsg_prev['TmFG2M'] = rsg_prev['TmFGM'] - rsg_prev['TmFG3M']
    rsg_prev['OppFG2A'] = rsg_prev['OppFGA'] - rsg_prev['OppFG3A']
    rsg_prev['OppFG2M'] = rsg_prev['OppFGM'] - rsg_prev['OppFG3M']
    rsg_prev['TmTRB'] = rsg_prev['TmORB'] + rsg_prev['TmDRB']
    rsg_prev['OppTRB'] = rsg_prev['OppORB'] + rsg_prev['OppDRB']

    rsg_tocalc = rsg_prev.copy()

timer.split('Read everything in: ')
###############################################################################
###############################################################################
###############################################################################
###############################################################################

# Modifying working rsg dataframe

# Create detailedgame field in rsg to indicate if the game has details or not
rsg_tocalc['DetailedGame'] = 0
rsg_tocalc.loc[(rsg_tocalc['TmFGM'] > 0), 'DetailedGame'] = 1

# Create high-level counts of games, detailed games, and a count missing details in rsg_summary
rsg_summary1 = rsg_tocalc[['Season', 'GameDate']].groupby(
    ['Season']).agg('count').reset_index()
rsg_summary2 = rsg_tocalc[['Season', 'DetailedGame']].groupby(
    ['Season']).agg('sum').reset_index()
rsg_summary = pd.merge(rsg_summary1, rsg_summary2, how='inner', on=['Season'])
rsg_summary = rsg_summary.rename(columns={
    'GameDate': 'GameCount',
    'DetailedGame': 'DetailedGameCount'
})
rsg_summary['MissingDetails'] = rsg_summary['GameCount'] - rsg_summary[
    'DetailedGameCount']

missinggamesseasons = len(rsg_summary.loc[rsg_summary['MissingDetails'] > 0])

print('\n' + str(missinggamesseasons) + ' seasons missing details...')

del rsg_summary1, rsg_summary2, rsg_tocalc['DetailedGame']
del missinggamesseasons

###############################################################################
# Round out game records
###############################################################################

# Bring in team names for both Tm and Opp
teams = pd.read_csv(
    filepath_or_buffer='../data/Stage2DataFiles/Teams.csv')

rsg_tocalc = pd.merge(
    rsg_tocalc, teams[['TeamID', 'TeamName']], left_on='TmID', right_on='TeamID')
del rsg_tocalc['TeamID']
rsg_tocalc = rsg_tocalc.rename(columns={'TeamName': 'TmName'})
rsg_tocalc = pd.merge(
    rsg_tocalc, teams[['TeamID', 'TeamName']], left_on='OppID', right_on='TeamID')
del rsg_tocalc['TeamID']
rsg_tocalc = rsg_tocalc.rename(columns={'TeamName': 'OppName'})

# Add countable field for number of games
rsg_tocalc['TmGame'] = 1
rsg_tocalc['OppGame'] = 1

# Add field for number of minutes
rsg_tocalc['TmMins'] = 40 + rsg_tocalc['GameOT'] * 5
rsg_tocalc['OppMins'] = rsg_tocalc['TmMins']

# Calculate field goal percentages in each game
rsg_tocalc['TmFGPct'] = rsg_tocalc['TmFGM'] / rsg_tocalc['TmFGA']
rsg_tocalc['TmFG3Pct'] = rsg_tocalc['TmFG3M'] / rsg_tocalc['TmFG3A']
rsg_tocalc['TmFG2Pct'] = rsg_tocalc['TmFG2M'] / rsg_tocalc['TmFG2A']
rsg_tocalc['TmFTPct'] = rsg_tocalc['TmFTM'] / rsg_tocalc['TmFTA']
rsg_tocalc['OppFGPct'] = rsg_tocalc['OppFGM'] / rsg_tocalc['OppFGA']
rsg_tocalc['OppFG3Pct'] = rsg_tocalc['OppFG3M'] / rsg_tocalc['OppFG3A']
rsg_tocalc['OppFG2Pct'] = rsg_tocalc['OppFG2M'] / rsg_tocalc['OppFG2A']
rsg_tocalc['OppFTPct'] = rsg_tocalc['OppFTM'] / rsg_tocalc['OppFTA']

# Calculate game margin
rsg_tocalc['TmMargin'] = rsg_tocalc['TmPF'] - rsg_tocalc['OppPF']
rsg_tocalc['OppMargin'] = -rsg_tocalc['TmMargin']

# Calculate win columns
rsg_tocalc['TmWin'] = 0
rsg_tocalc.loc[rsg_tocalc['TmMargin'] > 0,'TmWin'] = 1
rsg_tocalc['OppWin'] = 1 - rsg_tocalc['TmWin']

# Calculate Shooting Ratios
rsg_tocalc['TmFG3AperFG2A'] = rsg_tocalc['TmFG3A'] / rsg_tocalc['TmFG2A']
rsg_tocalc['OppFG3AperFG2A'] = rsg_tocalc['OppFG3A'] / rsg_tocalc['OppFG2A']

# Calculate Rebounding Percentages
rsg_tocalc['TmORBPct'] = rsg_tocalc['TmORB'] / (rsg_tocalc['TmORB'] + rsg_tocalc['OppDRB'])
rsg_tocalc['OppDRBPct'] = 1 - rsg_tocalc['TmORBPct']
rsg_tocalc['OppORBPct'] = rsg_tocalc['OppORB'] / (rsg_tocalc['OppORB'] + rsg_tocalc['TmDRB'])
rsg_tocalc['TmDRBPct'] = 1 - rsg_tocalc['OppORBPct']


# Add field for number of possessions (BasketballReference Method)
# https://www.basketball-reference.com/about/glossary.html
rsg_tocalc['TmPoss'] = (
        0.5 * ((rsg_tocalc['TmFGA']
                + 0.4 * rsg_tocalc['TmFTA']
                - 1.07 * (rsg_tocalc['TmORB'] /
                          (rsg_tocalc['TmORB'] + rsg_tocalc['OppDRB']))
                      * (rsg_tocalc['TmFGA']
                          - rsg_tocalc['TmFGM'])
                + rsg_tocalc['TmTO'])
               + (rsg_tocalc['OppFGA']
                  + 0.4 * rsg_tocalc['OppFTA']
                  - 1.07 * (rsg_tocalc['OppORB'] /
                            (rsg_tocalc['OppORB'] + rsg_tocalc['TmDRB']))
                        * (rsg_tocalc['OppFGA']
                           - rsg_tocalc['OppFGM'])
                  + rsg_tocalc['OppTO'])))

rsg_tocalc['OppPoss'] = (
        0.5 * ((rsg_tocalc['OppFGA']
                + 0.4 * rsg_tocalc['OppFTA']
                - 1.07 * (rsg_tocalc['OppORB'] /
                          (rsg_tocalc['OppORB'] + rsg_tocalc['TmDRB']))
                      * (rsg_tocalc['OppFGA']
                          - rsg_tocalc['OppFGM'])
                + rsg_tocalc['OppTO'])
               + (rsg_tocalc['TmFGA']
                  + 0.4 * rsg_tocalc['TmFTA']
                  - 1.07 * (rsg_tocalc['TmORB'] /
                            (rsg_tocalc['TmORB'] + rsg_tocalc['OppDRB']))
                        * (rsg_tocalc['TmFGA']
                           - rsg_tocalc['TmFGM'])
                  + rsg_tocalc['TmTO'])))

#rsg_tocalc['TmPossNET'] = (rsg_tocalc['TmFGA']
#                        - rsg_tocalc['TmORB']
#                        + rsg_tocalc['TmTO']
#                        + .475 * rsg_tocalc['TmFTA'])
#rsg_tocalc['OppPossNET'] = (rsg_tocalc['OppFGA']
#                         - rsg_tocalc['OppORB']
#                         + rsg_tocalc['OppTO']
#                         + .475 * rsg_tocalc['OppFTA'])

# Calculate per-40 and per-poss metrics for each game in rsg_tocalc
for x in {'Opp', 'Tm'}:
    for column in metrics:
        rsg_tocalc[x + column + 'per40'] = rsg_tocalc[x + column] / rsg_tocalc[x + 'Mins'] * 40
        rsg_tocalc[x + column + 'perPoss'] = rsg_tocalc[x + column] / rsg_tocalc[x + 'Poss']
del column, x

# Create the rsg_out dataframe
# TODO get opponent rank into output


# TODO rank game percentiles, give A, B, C, D, F
# # Rank Game Percentiles
# rsgc['OffPercentile'] = 1 - (len(rsgc)-rankdata(rsgc['TeamOffScore'],method='min'))/len(rsgc)
# rsgc['DefPercentile'] = 1 - (len(rsgc)-rankdata(rsgc['TeamDefScore'],method='min'))/len(rsgc)
# rsgc['OAMPercentile'] = 1 - (len(rsgc)-rankdata(rsgc['TeamOAM'],method='min'))/len(rsgc)

rsg_out = rsg_out.append(rsg_tocalc)

timer.split('Pre-loop split: ')
###############################################################################
###############################################################################
###############################################################################
###############################################################################

# Loop through and do a bunch of things

# Loop through the specified seasons to loop
for workingseason in seasonstoloop:

    print('Working on ' + str(workingseason) + '...')

    # Limit the rsg data to just the current season
    rsg_workingseason = rsg_tocalc.loc[rsg_tocalc['Season'] == workingseason]

    # Set flag for if season has detailed results
    if rsg_workingseason['TmFGA'].isnull().values.any():
        detailedseason = False
    else:
        detailedseason = True

    # Create current season seasonteams, summing all summables
    st_workingseason = rsg_workingseason.groupby(
            ['TmID', 'TmName'])[summables].sum().reset_index()

    # Add season column to seasonteams_currseason
    st_workingseason['Season'] = workingseason

    # Get per-game season stats into st_currseason
    # (can't just sum per-games since it will incorrectly weight some games)
    for x in {'Opp', 'Tm'}:
        for column in metrics:
            st_workingseason[x + column + 'perGame'] = (
                    st_workingseason[x + column]
                    / st_workingseason[x + 'Game'])
            st_workingseason[x + column + 'per40'] = (
                    st_workingseason[x + column]
                    / st_workingseason[x + 'Mins'] * 40)
            if detailedseason is False:
                st_workingseason[x + column + 'perPoss'] = float('NaN')
            else:
                st_workingseason[x + column + 'perPoss'] = (
                        st_workingseason[x + column]
                        / st_workingseason[x + 'Poss'])
    del column, x

    st_workingseason['TmPossperGame'] = st_workingseason['TmPoss'] / st_workingseason['TmGame']
    st_workingseason['MeanPossperGame'] = np.sum(st_workingseason['TmPoss']) / np.sum(st_workingseason['TmGame'])

    # Opponent adjust all metrics
    for prefix in {'Opp', 'Tm'}:
        for coremetric in metrics:
                opponentadjust(prefix, coremetric)
    del prefix, coremetric

    # Get SoS metric
    st_workingseason['TmSoS'] = (st_workingseason['OA_TmMarginper40']
                                 - st_workingseason['TmMarginper40'])

    # Get Losses into st_workingseason
    st_workingseason['TmLoss'] = (st_workingseason['TmGame']
                                  - st_workingseason['TmWin'])

    # True shooting percentage
    if detailedseason is False:
        st_workingseason['TmTSP'] = float('NaN')
        st_workingseason['OppTSP'] = float('NaN')
    else:
        st_workingseason['TmTSP'] = (st_workingseason['TmPF'] /
                                     (2 * (st_workingseason['TmFGA']
                                      + 0.44 * st_workingseason['TmFTA'])))
        st_workingseason['OppTSP'] = (st_workingseason['OppPF'] /
                                      (2 * (st_workingseason['OppFGA']
                                       + 0.44 * st_workingseason['OppFTA'])))

    # Assist/TO
    if detailedseason is False:
        st_workingseason['TmAstTORatio'] = float('NaN')
        st_workingseason['OppAstTORatio'] = float('NaN')
    else:
        st_workingseason['TmAstTORatio'] = (st_workingseason['TmAst']
                                            / st_workingseason['TmTO'])
        st_workingseason['OppAstTORatio'] = (st_workingseason['OppAst']
                                             / st_workingseason['OppTO'])

    if detailedseason is not False:
        # Calculate field goal percentages in each season
        st_workingseason['TmFGPct'] = st_workingseason['TmFGM'] / st_workingseason['TmFGA']
        st_workingseason['TmFG3Pct'] = st_workingseason['TmFG3M'] / st_workingseason['TmFG3A']
        st_workingseason['TmFG2Pct'] = st_workingseason['TmFG2M'] / st_workingseason['TmFG2A']
        st_workingseason['TmFTPct'] = st_workingseason['TmFTM'] / st_workingseason['TmFTA']
        st_workingseason['OppFGPct'] = st_workingseason['OppFGM'] / st_workingseason['OppFGA']
        st_workingseason['OppFG3Pct'] = st_workingseason['OppFG3M'] / st_workingseason['OppFG3A']
        st_workingseason['OppFG2Pct'] = st_workingseason['OppFG2M'] / st_workingseason['OppFG2A']
        st_workingseason['OppFTPct'] = st_workingseason['OppFTM'] / st_workingseason['OppFTA']

        # Calculate Shooting Ratios
        st_workingseason['TmFG3AperFG2A'] = st_workingseason['TmFG3A'] / st_workingseason['TmFG2A']
        st_workingseason['OppFG3AperFG2A'] = st_workingseason['OppFG3A'] / st_workingseason['OppFG2A']

        # Calculate Rebounding Percentages
        st_workingseason['TmORBPct'] = st_workingseason['TmORB'] / (st_workingseason['TmORB'] + st_workingseason['OppDRB'])
        st_workingseason['OppDRBPct'] = 1 - st_workingseason['TmORBPct']
        st_workingseason['OppORBPct'] = st_workingseason['OppORB'] / (st_workingseason['OppORB'] + st_workingseason['TmDRB'])
        st_workingseason['TmDRBPct'] = 1 - st_workingseason['OppORBPct']

        # Calculate OE, DE, NE Fields
        teamratings = pd.read_csv(
            filepath_or_buffer='../data/teamratings.csv'
        ).rename(columns={
            'OE': 'TmOE',
            'DE': 'TmDE',
            'Net': 'TmNE',
            'TeamID': 'TmID'
        })
        st_workingseason = pd.merge(
            left=st_workingseason,
            right=teamratings[['TmID','Season','TmOE','TmDE','TmNE']],
            on=['TmID', 'Season'],
            how='left'
        )
    else:
        pass

    # Rank all metrics
    for metric in ascendingrankmetrics:
        if detailedseason is False:
            st_workingseason['Rank_' + metric] = 0
        else:
            st_workingseason['Rank_' + metric] = (
                    st_workingseason[metric].rank(method='min',
                                                  ascending=True,
                                                  na_option='bottom'))
    for metric in descendingrankmetrics:
        if detailedseason is False:
            st_workingseason['Rank_' + metric] = 0
        else:
            st_workingseason['Rank_' + metric] = (
                    st_workingseason[metric].rank(method='min',
                                                  ascending=False,
                                                  na_option='bottom'))

    del metric

    # Append the working seasons output to the total seasonteams output
    seasonteams_out = seasonteams_out.append(st_workingseason)

del st_workingseason, rsg_workingseason, rsg_tocalc
del descendingrankmetrics, ascendingrankmetrics, detailedseason

seasonteams_out = seasonteams_out.reset_index()
rsg_out = rsg_out.reset_index()
del seasonteams_out['index'], rsg_out['index']

timer.split('Looping time: ')
###############################################################################
###############################################################################
###############################################################################
###############################################################################

# Do tournament information
if runall == True:
    trd = pd.read_csv(
        filepath_or_buffer='../data/Stage2DataFiles/NCAATourneyDetailedResults.csv')
    trc = pd.read_csv(
        filepath_or_buffer='../data/Stage2DataFiles/NCAATourneyCompactResults.csv')
    # trd18 = pd.read_csv( # Removing 3/18/20
    #     filepath_or_buffer='/Users/Ryan/Google Drive/ncaa-basketball-data/2018TourneyResults.csv')

    # Merge in day 0 to rsgd, add days to day 0 to get date of game, delete extra columns
    trd = pd.merge(trd, seasons[['Season', 'DayZero']], on='Season')
    trd['DayZero'] = pd.to_datetime(trd['DayZero'], format='%m/%d/%Y')
    trd['DayNum'] = pd.to_timedelta(trd['DayNum'], unit='d')
    trd['GameDate'] = trd['DayZero'] + trd['DayNum']
    trd['GameDate'] = pd.to_datetime(trd['GameDate']).dt.strftime('%Y-%m-%d')
    del trd['DayNum'], trd['DayZero']

    trc = pd.merge(trc, seasons[['Season', 'DayZero']], on='Season')
    trc['DayZero'] = pd.to_datetime(trc['DayZero'], format='%m/%d/%Y')
    trc['DayNum'] = pd.to_timedelta(trc['DayNum'], unit='d')
    trc['GameDate'] = trc['DayZero'] + trc['DayNum']
    trc['GameDate'] = pd.to_datetime(trc['GameDate']).dt.strftime('%Y-%m-%d')
    del trc['DayNum'], trc['DayZero']

    # trd = trd.append(trd18) # removing 3/18/20

    tr = pd.merge(
            left = trc,
            right = trd,
            on = list(trc),
            how = 'outer')

    del trc, trd#, trd18

    # Rename all the columns...
    tr = tr.rename(columns={'GameDate': 'GameDate',
                            'NumOT': 'GameOT',
                            'WTeamID': 'TmID',
                            'WScore': 'TmPF',
                            'WFGM': 'TmFGM',
                            'WFGA': 'TmFGA',
                            'WFGM2': 'TmFG2M',
                            'WFGA2': 'TmFG2A',
                            'WFGM3': 'TmFG3M',
                            'WFGA3': 'TmFG3A',
                            'WFTM': 'TmFTM',
                            'WFTA': 'TmFTA',
                            'WOR': 'TmORB',
                            'WDR': 'TmDRB',
                            'WTRB': 'TmTRB',
                            'WAst': 'TmAst',
                            'WStl': 'TmStl',
                            'WBlk': 'TmBlk',
                            'WTO': 'TmTO',
                            'WPF': 'TmFoul',
                            'WLoc': 'TmLoc',
                            'LTeamID': 'OppID',
                            'LScore': 'OppPF',
                            'LFGM': 'OppFGM',
                            'LFGA': 'OppFGA',
                            'LFGM2': 'OppFG2M',
                            'LFGA2': 'OppFG2A',
                            'LFGM3': 'OppFG3M',
                            'LFGA3': 'OppFG3A',
                            'LFTM': 'OppFTM',
                            'LFTA': 'OppFTA',
                            'LOR': 'OppORB',
                            'LDR': 'OppDRB',
                            'LTRB': 'OppTRB',
                            'LAst': 'OppAst',
                            'LStl': 'OppStl',
                            'LBlk': 'OppBlk',
                            'LTO': 'OppTO',
                            'LPF': 'OppFoul',
                            'LLoc': 'OppLoc'})

    # Copy, rename, and append the other half of the games to rsg_prev
    ltr = tr.copy()
    newnames = pd.DataFrame(list(ltr),columns = ['OldName'])
    newnames['NewName'] = newnames['OldName']
    newnames.loc[newnames['OldName'].str[0:3] == 'Opp','NewName'] = 'Tm' + newnames['OldName'].str[3:]
    newnames.loc[newnames['OldName'].str[0:2] == 'Tm','NewName'] = 'Opp' + newnames['OldName'].str[2:]
    newnames = newnames.set_index('OldName')['NewName']
    ltr = ltr.rename(columns = newnames)
    ltr['TmLoc'] = 'N'
    ltr.loc[ltr['OppLoc'] == 'H', 'TmLoc'] = 'A'
    ltr.loc[ltr['OppLoc'] == 'A', 'TmLoc'] = 'H'
    del ltr['OppLoc']
    tr = tr.append(ltr)
    del ltr, newnames

    # Handle column differences
    tr['TmFG2A'] = tr['TmFGA'] - tr['TmFG3A']
    tr['TmFG2M'] = tr['TmFGM'] - tr['TmFG3M']
    tr['OppFG2A'] = tr['OppFGA'] - tr['OppFG3A']
    tr['OppFG2M'] = tr['OppFGM'] - tr['OppFG3M']
    tr['TmTRB'] = tr['TmORB'] + tr['TmDRB']
    tr['OppTRB'] = tr['OppORB'] + tr['OppDRB']

    tr = pd.merge(
        tr, teams[['TeamID', 'TeamName']], left_on='TmID', right_on='TeamID')
    del tr['TeamID']
    tr = tr.rename(columns={'TeamName': 'TmName'})
    tr = pd.merge(
        tr, teams[['TeamID', 'TeamName']], left_on='OppID', right_on='TeamID')
    del tr['TeamID']
    tr = tr.rename(columns={'TeamName': 'OppName'})

    # Add countable field for number of games
    tr['TmGame'] = 1
    tr['OppGame'] = 1

    # Add field for number of minutes
    tr['TmMins'] = 40 + tr['GameOT'] * 5
    tr['OppMins'] = tr['TmMins']

    # Calculate field goal percentages in each game
    tr['TmFGPct'] = tr['TmFGM'] / tr['TmFGA']
    tr['TmFG3Pct'] = tr['TmFG3M'] / tr['TmFG3A']
    tr['TmFG2Pct'] = tr['TmFG2M'] / tr['TmFG2A']
    tr['TmFTPct'] = tr['TmFTM'] / tr['TmFTA']
    tr['OppFGPct'] = tr['OppFGM'] / tr['OppFGA']
    tr['OppFG3Pct'] = tr['OppFG3M'] / tr['OppFG3A']
    tr['OppFG2Pct'] = tr['OppFG2M'] / tr['OppFG2A']
    tr['OppFTPct'] = tr['OppFTM'] / tr['OppFTA']

    # Calculate game margin
    tr['TmMargin'] = tr['TmPF'] - tr['OppPF']
    tr['OppMargin'] = -tr['TmMargin']

    # Calculate win columns
    tr['TmWin'] = 0
    tr.loc[tr['TmMargin'] > 0,'TmWin'] = 1
    tr['OppWin'] = 1 - tr['TmWin']


    # Add field for number of possessions (NCAA NET method)
    tr['TmPoss'] = tr['TmFGA'] \
                    - tr['TmORB'] \
                    + tr['TmTO'] \
                    + .475 * tr['TmFTA']
    tr['OppPoss'] = tr['OppFGA'] \
                    - tr['OppORB'] \
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
        filepath_or_buffer='../data/Stage2DataFiles/NCAATourneySeeds.csv'
    )
    tourneyseeds['TmPlayInTeam'] = 0
    tourneyseeds.loc[tourneyseeds['Seed'].str.len() == 4,'TmPlayInTeam'] = 1
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
                        'OppID':'ID',
                        'OppTourneySeed':'TourneySeed'
                        })


    # TODO un-limit the TR dataframe
    # Drop play-in-games
    tr['PlayInGame'] = 0
    tr.loc[(tr['TmPlayInTeam'] == 1) & (tr['OppPlayInTeam'] ==  1),'PlayInGame'] = 1
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
            left_on = ['Season','TmID']
            ,right_on = ['Season','ID'])
    del tourneyseeds, seasontourney['ID']

    seasontourney['PlayInWin'] = 0
    seasontourney.loc[(seasontourney['PlayInTeam'] == 1) &
                      (seasontourney['TourneyGame'] > 1)
                      ,'PlayInWin'] = 1
    seasontourney['TourneyGame'] = seasontourney['TourneyGame'] - seasontourney['PlayInGame']
    seasontourney['TourneyWin'] = seasontourney['TourneyWin'] - seasontourney['PlayInWin']

    # Get round information into seasontourney
    seasontourney['TourneyResultStr'] = '-'
    seasontourney.loc[seasontourney['TourneyWin'] == 6,'TourneyResultStr'] = 'Champion'
    seasontourney.loc[seasontourney['TourneyWin'] == 5,'TourneyResultStr'] = 'Runner Up'
    seasontourney.loc[seasontourney['TourneyWin'] == 4,'TourneyResultStr'] = 'Final 4'
    seasontourney.loc[seasontourney['TourneyWin'] == 3,'TourneyResultStr'] = 'Elite 8'
    seasontourney.loc[seasontourney['TourneyWin'] == 2,'TourneyResultStr'] = 'Sweet 16'
    seasontourney.loc[seasontourney['TourneyWin'] == 1,'TourneyResultStr'] = 'Rnd of 32'
    seasontourney.loc[seasontourney['TourneyWin'] == 0,'TourneyResultStr'] = 'Rnd of 64'

    seasonteams_out = pd.merge(
                        left = seasonteams_out,
                        right = seasontourney,
                        how = 'left',
                        on = ['Season','TmID'])
elif runall == False:
    pass





timer.split('Pre-write: ')
# Write output
colOrder = ['Season','TmName']
otherCols = [x for x in rsg_out.columns if x not in colOrder]
rsg_out = rsg_out[
    colOrder + otherCols
]
createreplacecsv('/Users/Ryan/Google Drive/ncaa-basketball-data/rsg.csv',rsg_out)

createreplacecsv('/Users/Ryan/Google Drive/ncaa-basketball-data/seasonteams.csv',seasonteams_out)

timer.split('Post-write: ')
timer.end()
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

