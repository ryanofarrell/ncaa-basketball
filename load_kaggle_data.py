#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 29 22:06:31 2019

@author: Ryan
"""

from db import get_db
import pandas as pd
from UDFs import createreplacecsv, printtitle, dfcoldiffs, timer

printtitle('Running season calculations')

timer = timer()
timer.start()

# Set the loop of the rsg dataframe to all season
seasonstoloop = list(range(1985, 2019))


def calculate_possessions(reg_season_games):
    '''Add field for number of possessions (BasketballReference Method) \n
    https://www.basketball-reference.com/about/glossary.html
    Returns the dataframe it is provided with two new fields: TmPoss and OppPoss
    '''
    reg_season_games['TmPoss'] = (
            0.5 * ((reg_season_games['TmFGA']
                    + 0.4 * reg_season_games['TmFTA']
                    - 1.07 * (reg_season_games['TmORB'] /
                              (reg_season_games['TmORB']
                               + reg_season_games['OppDRB']))
                    * (reg_season_games['TmFGA']
                       - reg_season_games['TmFGM'])
                    + reg_season_games['TmTO'])
                   + (reg_season_games['OppFGA']
                      + 0.4 * reg_season_games['OppFTA']
                      - 1.07 * (reg_season_games['OppORB'] /
                                (reg_season_games['OppORB']
                                 + reg_season_games['TmDRB']))
                      * (reg_season_games['OppFGA']
                         - reg_season_games['OppFGM'])
                      + reg_season_games['OppTO'])))

    reg_season_games['OppPoss'] = (
            0.5 * ((reg_season_games['OppFGA']
                    + 0.4 * reg_season_games['OppFTA']
                    - 1.07 * (reg_season_games['OppORB'] /
                              (reg_season_games['OppORB']
                               + reg_season_games['TmDRB']))
                    * (reg_season_games['OppFGA']
                       - reg_season_games['OppFGM'])
                    + reg_season_games['OppTO'])
                   + (reg_season_games['TmFGA']
                      + 0.4 * reg_season_games['TmFTA']
                      - 1.07 * (reg_season_games['TmORB'] /
                                (reg_season_games['TmORB']
                                 + reg_season_games['OppDRB']))
                      * (reg_season_games['TmFGA']
                         - reg_season_games['TmFGM'])
                      + reg_season_games['TmTO']))
    )
    return reg_season_games


def opponentadjust(prefix, metric, reg_season_games):
    '''Returns a dataframe with:
    1) a record for each season-team combo in reg_season_games
    2) a field 'OA_' + prefix + metric, which is the opponent-adjusted metric 
    '''

    # Figure out the prefix
    assert prefix in ['Opp', 'Tm'], 'Must be Opp or Tm as the prefix'
    opposite_prefix = 'Opp' if prefix == 'Tm' else 'Tm'

    # aggregate the metric's game data
    season_teams = reg_season_games['Season', 'TmName', prefix + metric, opposite_prefix_metric].groupby(
        ['Season', 'TmName']).agg('sum').reset_index()
    print(season_teams.head())
    # Into temo_st_currseason, get the team names, season, and
    # the opposite side's metric of what is being adjusted
    # For example, when OAing TmPFper40, temp_iteams will contain the
    # team's OppPFper40
    # This is used later for comparing a team's performance to the
    # opponent's average
    temp_st_workingseason = st_workingseason[['TmName', 'Season', opposite_prefix + metric + 'perGame']]

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
    st_workingseason['OA_' + prefix + coremetric + 'perGame'] = \
        st_workingseason['OA_' + prefix + coremetric] / \
        st_workingseason[prefix + 'Game']
    st_workingseason['OA_' + prefix + coremetric + 'per40'] = \
        st_workingseason['OA_' + prefix + coremetric] / \
        st_workingseason[prefix + 'Mins'] * 40
    st_workingseason['OA_' + prefix + coremetric + 'perPoss'] = \
        st_workingseason['OA_' + prefix + coremetric] / \
        st_workingseason[prefix + 'Poss']

    # Delete the useless season aggregate
    del st_workingseason['OA_' + prefix + coremetric]
    del rsg_workingseason['GameOppAdj_' + prefix + coremetric]


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

descendingrankmetrics = ['TmSoS', 'TmTSP', 'TmAstTORatio']
ascendingrankmetrics = ['OppTSP', 'OppAstTORatio']
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
                z = x * y
                if z == 1:
                    descendingrankmetrics.append(oa + prefix + coremetric + suffix)
                else:
                    ascendingrankmetrics.append(oa + prefix + coremetric + suffix)
del oa, prefix, coremetric, suffix, negativecoremetrics, x, y, z

"""
Loads required Kaggle data from 2019 competition
https://www.kaggle.com/c/mens-machine-learning-competition-2019/data
"""
###############################################################################
###############################################################################
# Ingest all data up to end of 2018 season, clean up, and get required fields
###############################################################################
###############################################################################
# Initialize rsg_out
rsg_out = pd.DataFrame()

# Initialize seasonteams_out
seasonteams_out = pd.DataFrame()


def calculate_game_dates(reg_season_games, seasons):
    '''Takes dataframe of regular season games and determines the game date
    Uses day 0 from seasons and adds a timedelta
    '''
    reg_season_games = pd.merge(reg_season_games, seasons[['Season', 'DayZero']], on='Season')
    reg_season_games['DayZero'] = pd.to_datetime(reg_season_games['DayZero'], format='%m/%d/%Y')
    reg_season_games['DayNum'] = pd.to_timedelta(reg_season_games['DayNum'], unit='d')
    reg_season_games['GameDate'] = reg_season_games['DayZero'] + reg_season_games['DayNum']
    del reg_season_games['DayZero'], reg_season_games['DayNum']

    print('Game Dates Calculated Successfully')
    return reg_season_games


def rename_and_duplicate_records(reg_season_games):
    '''Takes dataframe and:
    1) renames the columns to understood naming convention
    2) duplicates the records, flips the winners/losers, and appends
    This results in a dataframe that has a TmGame record for every game a team played,\
    regardless if they won or lost.
    '''
    renamable_columns = {'GameDate': 'GameDate', 'NumOT': 'GameOT', 'WTeamID': 'TmID', 'WScore': 'TmPF',
                         'WFGM': 'TmFGM', 'WFGA': 'TmFGA', 'WFGM2': 'TmFG2M', 'WFGA2': 'TmFG2A', 'WFGM3': 'TmFG3M',
                         'WFGA3': 'TmFG3A', 'WFTM': 'TmFTM', 'WFTA': 'TmFTA', 'WOR': 'TmORB', 'WDR': 'TmDRB',
                         'WTRB': 'TmTRB', 'WAst': 'TmAst', 'WStl': 'TmStl', 'WBlk': 'TmBlk', 'WTO': 'TmTO',
                         'WPF': 'TmFoul', 'WLoc': 'TmLoc', 'LTeamID': 'OppID', 'LScore': 'OppPF', 'LFGM': 'OppFGM',
                         'LFGA': 'OppFGA', 'LFGM2': 'OppFG2M', 'LFGA2': 'OppFG2A', 'LFGM3': 'OppFG3M',
                         'LFGA3': 'OppFG3A', 'LFTM': 'OppFTM', 'LFTA': 'OppFTA', 'LOR': 'OppORB', 'LDR': 'OppDRB',
                         'LTRB': 'OppTRB', 'LAst': 'OppAst', 'LStl': 'OppStl', 'LBlk': 'OppBlk', 'LTO': 'OppTO',
                         'LPF': 'OppFoul', 'LLoc': 'OppLoc'}
    reg_season_games = reg_season_games.rename(columns=renamable_columns)

    # Copy, rename, and append the other half of the games to reg_season_games
    loser_reg_season_games = reg_season_games.copy()
    newnames = pd.DataFrame(list(loser_reg_season_games), columns=['OldName'])
    newnames['NewName'] = newnames['OldName']
    newnames.loc[newnames['OldName'].str[0:3] == 'Opp', 'NewName'] = 'Tm' + newnames['OldName'].str[3:]
    newnames.loc[newnames['OldName'].str[0:2] == 'Tm', 'NewName'] = 'Opp' + newnames['OldName'].str[2:]
    newnames = newnames.set_index('OldName')['NewName']
    loser_reg_season_games = loser_reg_season_games.rename(columns=newnames)
    loser_reg_season_games['TmLoc'] = 'N'
    loser_reg_season_games.loc[loser_reg_season_games['OppLoc'] == 'H', 'TmLoc'] = 'A'
    loser_reg_season_games.loc[loser_reg_season_games['OppLoc'] == 'A', 'TmLoc'] = 'H'
    del loser_reg_season_games['OppLoc']
    reg_season_games = reg_season_games.append(loser_reg_season_games, sort=True)
    print(reg_season_games.columns)

    return reg_season_games


def read_and_clean_source_data():
    reg_season_games_compact = pd.read_csv('data/Stage2DataFiles/RegularSeasonCompactResults.csv')
    reg_season_games_detailed = pd.read_csv('data/Stage2DataFiles/RegularSeasonDetailedResults.csv')
    seasons = pd.read_csv('data/Stage2DataFiles/Seasons.csv')
    teams = pd.read_csv('data/Stage2DataFiles/Teams.csv')

    # Merge compact and detailed results
    reg_season_games_combined = pd.merge(
        left=reg_season_games_compact,
        right=reg_season_games_detailed,
        on=list(reg_season_games_compact),
        how='outer')

    # Get game dates
    reg_season_games_combined = calculate_game_dates(reg_season_games_combined, seasons)

    reg_season_games_combined = rename_and_duplicate_records(reg_season_games_combined)

    return reg_season_games_combined


reg_season_games_combined = read_and_clean_source_data()
print('here')

# Copy, rename, and append the other half of the games to rsg_prev
lrsg_prev = rsg_prev.copy()
newnames = pd.DataFrame(list(lrsg_prev), columns=['OldName'])
newnames['NewName'] = newnames['OldName']
newnames.loc[newnames['OldName'].str[0:3] == 'Opp', 'NewName'] = 'Tm' + newnames['OldName'].str[3:]
newnames.loc[newnames['OldName'].str[0:2] == 'Tm', 'NewName'] = 'Opp' + newnames['OldName'].str[2:]
newnames = newnames.set_index('OldName')['NewName']
lrsg_prev = lrsg_prev.rename(columns=newnames)
lrsg_prev['TmLoc'] = 'N'
lrsg_prev.loc[lrsg_prev['OppLoc'] == 'H', 'TmLoc'] = 'A'
lrsg_prev.loc[lrsg_prev['OppLoc'] == 'A', 'TmLoc'] = 'H'
del lrsg_prev['OppLoc']
rsg_prev = rsg_prev.append(lrsg_prev, sort=True)
del lrsg_prev, newnames

# Handle column differences
rsg_prev['TmFG2A'] = rsg_prev['TmFGA'] - rsg_prev['TmFG3A']
rsg_prev['TmFG2M'] = rsg_prev['TmFGM'] - rsg_prev['TmFG3M']
rsg_prev['OppFG2A'] = rsg_prev['OppFGA'] - rsg_prev['OppFG3A']
rsg_prev['OppFG2M'] = rsg_prev['OppFGM'] - rsg_prev['OppFG3M']
rsg_prev['TmTRB'] = rsg_prev['TmORB'] + rsg_prev['TmDRB']
rsg_prev['OppTRB'] = rsg_prev['OppORB'] + rsg_prev['OppDRB']

# TODO implement assert once previous data being loaded
# assert dfcoldiffs(rsg_prev,rsg_curr,'count') == 0,'Columns different between rsg_out and rsg_tocalc'

# Append current-year data to rsgd
# TODO implement functionality - for now ignoring current year scraping
# rsg_tocalc = rsg_prev.append(rsg_curr)
rsg_tocalc = rsg_prev

del rsg_prev
# , rsg_curr

timer.split('Read everything in: ')

###############################################################################
###############################################################################
# Modifying working rsg dataframe
###############################################################################
###############################################################################

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
rsg_tocalc.loc[rsg_tocalc['TmMargin'] > 0, 'TmWin'] = 1
rsg_tocalc['OppWin'] = 1 - rsg_tocalc['TmWin']

rsg_tocalc = calculate_possessions(rsg_tocalc)

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

# TODO removed this assert when reviving in 2019...figure out what it does and if I need it
'''
if runall == True:
    pass
elif runall == False:
    assert dfcoldiffs(rsg_out,rsg_tocalc,'count') == 0,'Columns different between rsg_out and rsg_tocalc'
'''

rsg_out = rsg_out.append(rsg_tocalc, sort=True)

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
    # 2019-11-04 Don't rank since we don't need to persist this
    '''
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
    '''
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


###############################################################################
###############################################################################
###############################################################################
###############################################################################

# Do tournament information
trd = pd.read_csv(
    filepath_or_buffer='data/Stage2DataFiles/NCAATourneyDetailedResults.csv')
trc = pd.read_csv(
    filepath_or_buffer='data/Stage2DataFiles/NCAATourneyCompactResults.csv')
# TODO get 2018 tourney results file in
# trd18 = pd.read_csv(
#    filepath_or_buffer='/Users/Ryan/Google Drive/ncaa-basketball-data/2018TourneyResults.csv')

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

# TODO get 2018 tourney results in
# trd = trd.append(trd18)

tr = pd.merge(
    left=trc,
    right=trd,
    on=list(trc),
    how='outer')

del trc, trd
# , trd18

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
newnames = pd.DataFrame(list(ltr), columns=['OldName'])
newnames['NewName'] = newnames['OldName']
newnames.loc[newnames['OldName'].str[0:3] == 'Opp', 'NewName'] = 'Tm' + newnames['OldName'].str[3:]
newnames.loc[newnames['OldName'].str[0:2] == 'Tm', 'NewName'] = 'Opp' + newnames['OldName'].str[2:]
newnames = newnames.set_index('OldName')['NewName']
ltr = ltr.rename(columns=newnames)
ltr['TmLoc'] = 'N'
ltr.loc[ltr['OppLoc'] == 'H', 'TmLoc'] = 'A'
ltr.loc[ltr['OppLoc'] == 'A', 'TmLoc'] = 'H'
del ltr['OppLoc']
assert dfcoldiffs(tr, ltr, 'count') == 0, 'Columns different between rsg_out and rsg_tocalc'
tr = tr.append(ltr, sort=True)
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
tr.loc[tr['TmMargin'] > 0, 'TmWin'] = 1
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
    filepath_or_buffer='data/Stage2DataFiles/NCAATourneySeeds.csv'
)
tourneyseeds['TmPlayInTeam'] = 0
tourneyseeds.loc[tourneyseeds['Seed'].str.len() == 4, 'TmPlayInTeam'] = 1
tourneyseeds['Seed'] = tourneyseeds['Seed'].str[1:3].astype('int')
tourneyseeds = tourneyseeds.rename(columns={'TeamID': 'TmID', 'Seed': 'TmTourneySeed'})
tr = pd.merge(tr,
              tourneyseeds,
              how='left',
              on=['Season', 'TmID'])
tourneyseeds = tourneyseeds.rename(columns={
    'TmPlayInTeam': 'OppPlayInTeam',
    'TmID': 'OppID',
    'TmTourneySeed': 'OppTourneySeed'
})
tr = pd.merge(tr,
              tourneyseeds,
              how='left',
              on=['Season', 'OppID'])

# Rename in prep for getting in to seasontourney
tourneyseeds = tourneyseeds.rename(columns={
    'OppPlayInTeam': 'PlayInTeam',
    'OppID': 'ID',
    'OppTourneySeed': 'TourneySeed'
})

# TODO un-limit the TR dataframe
# Drop play-in-games
tr['PlayInGame'] = 0
tr.loc[(tr['TmPlayInTeam'] == 1) & (tr['OppPlayInTeam'] == 1), 'PlayInGame'] = 1
# tr = tr.loc[tr['PlayInGame'] != True ]

# Create seasontourney dataframe and summarize some data
seasontourney = tr.groupby(['TmID', 'Season'])[['TmGame', 'TmWin', 'PlayInGame']].sum().reset_index()
seasontourney = seasontourney.rename(columns={
    'TmWin': 'TourneyWin',
    'TmGame': 'TourneyGame'})

# Get seed information into seasontourney dataframe
seasontourney = pd.merge(
    left=seasontourney,
    right=tourneyseeds,
    how='inner',
    left_on=['Season', 'TmID']
    , right_on=['Season', 'ID'])
del tourneyseeds, seasontourney['ID']

seasontourney['PlayInWin'] = 0
seasontourney.loc[(seasontourney['PlayInTeam'] == 1) &
                  (seasontourney['TourneyGame'] > 1)
, 'PlayInWin'] = 1
seasontourney['TourneyGame'] = seasontourney['TourneyGame'] - seasontourney['PlayInGame']
seasontourney['TourneyWin'] = seasontourney['TourneyWin'] - seasontourney['PlayInWin']

# Get round information into seasontourney
seasontourney['TourneyResultStr'] = '-'
seasontourney.loc[seasontourney['TourneyWin'] == 6, 'TourneyResultStr'] = 'Champion'
seasontourney.loc[seasontourney['TourneyWin'] == 5, 'TourneyResultStr'] = 'Runner Up'
seasontourney.loc[seasontourney['TourneyWin'] == 4, 'TourneyResultStr'] = 'Final 4'
seasontourney.loc[seasontourney['TourneyWin'] == 3, 'TourneyResultStr'] = 'Elite 8'
seasontourney.loc[seasontourney['TourneyWin'] == 2, 'TourneyResultStr'] = 'Sweet 16'
seasontourney.loc[seasontourney['TourneyWin'] == 1, 'TourneyResultStr'] = 'Rnd of 32'
seasontourney.loc[seasontourney['TourneyWin'] == 0, 'TourneyResultStr'] = 'Rnd of 64'

seasonteams_out = pd.merge(
    left=seasonteams_out,
    right=seasontourney,
    how='left',
    on=['Season', 'TmID'])

timer.split('Pre-write: ')

# Initialize DB connection
db = get_db()

# Read in data, convert to dict, insert records into collection
db.games.drop()
'''
data = rsg_out.to_dict('records')
timer.split('Post-dict of rsg dataframe')
db.games.insert_many(data,ordered=False)
timer.split('Post-insert of rsg dataframe')
print('Inserted {} records into database'.format(len(data)))
'''

db.seasonteams.drop()
data = seasonteams_out.to_dict('records')
timer.split('Post-dict of seasonteams dataframe')

db.seasonteams.insert_many(data, ordered=False)
timer.split('Post-insert of seasonteams dataframe')
print('Inserted {} records into database'.format(len(data)))
timer.split('Post-write: ')
timer.end()
