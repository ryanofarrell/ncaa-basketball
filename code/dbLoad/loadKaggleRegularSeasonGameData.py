
#%%
import pandas as pd
import numpy as np
from math import isnan

# Import to add project folder to sys path
import sys
utils_path = '/Users/Ryan/Documents/projects/ncaa-basketball/code/utils'
if utils_path not in sys.path:
    sys.path.append(utils_path)

from db import get_db
from api import getSeasonsList, preAggSeasonGames


def calculatePossessions(reg_season_games):
    """Adds a column for Tm and Opp number of possessions to provided dataframe
    Uses Basketball Reference method:
    https://www.basketball-reference.com/about/glossary.html
    Note TmPoss == OppPoss

    Arguments:
        reg_season_games {DataFrame} -- Games to calculate possessions

    Returns:
        DataFrame -- Input DataFrame + TmPoss and OppPoss
    """
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

    reg_season_games['OppPoss'] = reg_season_games['TmPoss']
    return reg_season_games


def calculateGameDates(reg_season_games, seasons):
    """Adds GameDate column to provided DataFrame by adding DayNum to the
    season's DayZero

    Arguments:
        reg_season_games {DataFrame} -- Games with DayNum as the date
        seasons {DataFrame} -- All seasons with their DayZero

    Returns:
        DataFrame -- Input games + GameDate - DayNum
    """
    reg_season_games = pd.merge(reg_season_games,
                                seasons[['Season', 'DayZero']],
                                on='Season')
    reg_season_games['DayZero'] = pd.to_datetime(
        reg_season_games['DayZero'], format='%m/%d/%Y')
    reg_season_games['DayNum'] = pd.to_timedelta(
        reg_season_games['DayNum'], unit='d')
    reg_season_games['GameDate'] = reg_season_games['DayZero'] + \
        reg_season_games['DayNum']
    del reg_season_games['DayZero'], reg_season_games['DayNum']

    return reg_season_games


def renameKaggleColumns(reg_season_games):
    """Takes dataframe and:
    1) renames the columns to understood naming convention.
    2) duplicates the records, flips the winners/losers, and appends.
    This results in a dataframe that has a TmGame record for every
    game a team played, regardless if they won or lost.

    Arguments:
        reg_season_games {DataFrame} -- Data to manipulate, usually from Kaggle

    Returns:
        DataFrame -- Input data with new names and doubled length for
        W/L perspective
    """
    renamable_columns = {'GameDate': 'GameDate', 'NumOT': 'GameOT',
                         'WTeamID': 'TmID', 'WScore': 'TmPF',
                         'WFGM': 'TmFGM', 'WFGA': 'TmFGA', 'WFGM2': 'TmFG2M',
                         'WFGA2': 'TmFG2A', 'WFGM3': 'TmFG3M',
                         'WFGA3': 'TmFG3A', 'WFTM': 'TmFTM', 'WFTA': 'TmFTA',
                         'WOR': 'TmORB', 'WDR': 'TmDRB',
                         'WTRB': 'TmTRB', 'WAst': 'TmAst', 'WStl': 'TmStl',
                         'WBlk': 'TmBlk', 'WTO': 'TmTO',
                         'WPF': 'TmFoul', 'WLoc': 'TmLoc', 'LTeamID': 'OppID',
                         'LScore': 'OppPF', 'LFGM': 'OppFGM',
                         'LFGA': 'OppFGA', 'LFGM2': 'OppFG2M',
                         'LFGA2': 'OppFG2A', 'LFGM3': 'OppFG3M',
                         'LFGA3': 'OppFG3A', 'LFTM': 'OppFTM',
                         'LFTA': 'OppFTA', 'LOR': 'OppORB', 'LDR': 'OppDRB',
                         'LTRB': 'OppTRB', 'LAst': 'OppAst', 'LStl': 'OppStl',
                         'LBlk': 'OppBlk', 'LTO': 'OppTO',
                         'LPF': 'OppFoul', 'LLoc': 'OppLoc'}
    reg_season_games = reg_season_games.rename(columns=renamable_columns)

    return reg_season_games


def duplicateGameRecords(rsg):
    """Duplicates a dataframe of games and renames all columns appropriately.
    For use when a dataframe only has games from the winner's perspective.
    Renames Tm<> to Opp<> and visa-versa

    Arguments:
        rsg {DataFrame} -- contains the single-perspective records to duplicate

    Returns:
        DataFrame -- Double length of proided data with duplicated records
    """
    # Copy, rename, and append the other half of the games to reg_season_games
    loser_rsg = rsg.copy()
    newnames = pd.DataFrame(list(loser_rsg), columns=['OldName'])
    newnames['NewName'] = newnames['OldName']
    newnames.loc[newnames['OldName'].str[0:3] == 'Opp', 'NewName'] = \
        'Tm' + newnames['OldName'].str[3:]
    newnames.loc[newnames['OldName'].str[0:2] == 'Tm', 'NewName'] = \
        'Opp' + newnames['OldName'].str[2:]
    newnames = newnames.set_index('OldName')['NewName']
    loser_rsg = loser_rsg.rename(columns=newnames)
    loser_rsg['TmLoc'] = 'N'
    loser_rsg.loc[loser_rsg['OppLoc'] == 'H', 'TmLoc'] = 'A'
    loser_rsg.loc[loser_rsg['OppLoc'] == 'A', 'TmLoc'] = 'H'
    del loser_rsg['OppLoc']
    rsg = rsg.append(loser_rsg, sort=True)

    return rsg


def addAdditionalGameColumns(rsg):
    """Adds many detailed columns to build out useful fields

    Arguments:
        rsg {DataFrame} -- Games to add columns to

    Returns:
        DataFrame -- Input data + additional columns
    """
    # Add more field goal fields
    rsg['TmFG2A'] = rsg['TmFGA'] - rsg['TmFG3A']
    rsg['TmFG2M'] = rsg['TmFGM'] - rsg['TmFG3M']
    rsg['OppFG2A'] = rsg['OppFGA'] - rsg['OppFG3A']
    rsg['OppFG2M'] = rsg['OppFGM'] - rsg['OppFG3M']

    # Add more rebounding fields
    rsg['TmTRB'] = rsg['TmORB'] + rsg['TmDRB']
    rsg['OppTRB'] = rsg['OppORB'] + rsg['OppDRB']

    # Add countable field for number of games
    rsg['TmGame'] = 1
    rsg['OppGame'] = 1

    # Add field for number of minutes
    rsg['TmMins'] = 40 + rsg['GameOT'] * 5
    rsg['OppMins'] = rsg['TmMins']

    # Calculate field goal percentages in each game
    rsg['TmFGPct'] = rsg['TmFGM'] / rsg['TmFGA']
    rsg['TmFG3Pct'] = rsg['TmFG3M'] / rsg['TmFG3A']
    rsg['TmFG2Pct'] = rsg['TmFG2M'] / rsg['TmFG2A']
    rsg['TmFTPct'] = rsg['TmFTM'] / rsg['TmFTA']
    rsg['OppFGPct'] = rsg['OppFGM'] / rsg['OppFGA']
    rsg['OppFG3Pct'] = rsg['OppFG3M'] / rsg['OppFG3A']
    rsg['OppFG2Pct'] = rsg['OppFG2M'] / rsg['OppFG2A']
    rsg['OppFTPct'] = rsg['OppFTM'] / rsg['OppFTA']

    # Calculate game margin
    rsg['TmMargin'] = rsg['TmPF'] - rsg['OppPF']
    rsg['OppMargin'] = -rsg['TmMargin']

    # Calculate win columns
    rsg['TmWin'] = 0
    rsg.loc[rsg['TmMargin'] > 0, 'TmWin'] = 1
    rsg['OppWin'] = 1 - rsg['TmWin']

    # Indicate regular season game
    rsg['isRegularSeason'] = True

    rsg = calculatePossessions(rsg)

    return rsg


def addTeamNames(rsg, teams):
    """Adds team names to DataFrame that only has team IDs

    Arguments:
        rsg {DataFrame} -- Thing we want to add team names to
        teams {DataFrame} -- Contains ID-Name key

    Returns:
        DataFrame -- Input data + TmName and OppName
    """
    # Bring in team names for both Tm and Opp
    rsg = pd.merge(
        rsg, teams[['TeamID', 'TeamName']], left_on='TmID', right_on='TeamID')
    del rsg['TeamID']
    rsg = rsg.rename(columns={'TeamName': 'TmName'})
    rsg = pd.merge(
        rsg, teams[['TeamID', 'TeamName']], left_on='OppID', right_on='TeamID')
    del rsg['TeamID']
    rsg = rsg.rename(columns={'TeamName': 'OppName'})

    return rsg


def addVegasLineToKaggleData(rsg):
    # Read in data
    firstRun = True
    for suffix in range(3,20):
        suffStr = str(suffix)
        suffStr = '0'*(2-len(suffStr)) + suffStr
        print(f"Working on {suffStr}")
        csvPath = f'/Users/Ryan/Documents/projects/ncaa-basketball/data/vegasLines/ncaabb{suffStr}.csv'
        
        if firstRun:
            df = pd.read_csv(csvPath)
            firstRun = False
        else:
            df = df.append(pd.read_csv(csvPath))

    df = df[[
        'date',
        'home',
        'road',
        'neutral',
        'line'
    ]]
    df['line'] = pd.to_numeric(df['line'], errors='coerce')
    
    # Check for matching team names, drop those without
    df['home'] = df['home'].str.lower()
    df['home'] = df['home'].str.strip()
    df['road'] = df['road'].str.lower()
    df['road'] = df['road'].str.strip()
    teamSpellings = pd.read_csv(f'/Users/Ryan/Documents/projects/ncaa-basketball/data/TeamSpellings.csv', encoding="ISO-8859-1")
    df = pd.merge(
        left=df,
        right=teamSpellings.rename(columns={'TeamNameSpelling': 'home', 'TeamID': 'homeID'}),
        on=['home'],
        how='left'
    )
    df = pd.merge(
        left=df,
        right=teamSpellings.rename(columns={'TeamNameSpelling': 'road', 'TeamID': 'roadID'}),
        on=['road'],
        how='left'
    )
    dfMissingHome = df.loc[pd.isnull(df['homeID'])]
    dfMissingRoad = df.loc[pd.isnull(df['roadID'])]

    print(f"Dropping {len(df.loc[pd.isnull(df['homeID']) | pd.isnull(df['roadID'])])} records due to no matching team name")

    df = df.loc[~pd.isnull(df['homeID']) & ~pd.isnull(df['roadID'])]

    print(f"Dropping {len(df.loc[pd.isnull(df['line'])])} records due to no line provided")
    df = df.loc[~pd.isnull(df['line'])]

    # Clean up data to only leave fields needed to merge
    for col in ['home', 'road', 'neutral']:
        del df[col]

    df = df.rename(columns={
        'date': 'GameDate',
        'line': 'GameVegasLine',
        'homeID': 'TmID',
        'roadID': 'OppID'
    })

    df['GameDate'] = pd.to_datetime(df['GameDate'])

    # Duplicate records for each team
    df1 = df.copy()
    df1 = df1.rename(columns={
        'TmID': 'OppID',
        'OppID': 'TmID'
    })
    df1['GameVegasLine'] *= -1
    df = df.append(df1)

    df = df.drop_duplicates(subset=['TmID', 'OppID', 'GameDate'])

    # Merge in lines 
    rsg = pd.merge(
        rsg,
        df,
        on=['TmID', 'OppID', 'GameDate'],
        how='left'
    )
    print(f"Added {len(rsg.loc[~pd.isnull(rsg['GameVegasLine'])])} vegas lines")

    # TODO adjust game dates by 1 to add additional ~500 game lines

    return rsg


def readCleanSourceData():
    """Master handler of other functions. Reads in Kaggle data,
    gets game dates, renames columns, adds team names, and  duplicates records.

    Returns:
        DataFrame -- All the fun things described above
    """
    # TODO make sure this has all reg season games until end of 2019 season
    reg_season_games_compact = pd.read_csv(
        'data/MDataFiles_Stage2/MRegularSeasonCompactResults.csv')
    reg_season_games_detailed = pd.read_csv(
        'data/MDataFiles_Stage2/MRegularSeasonDetailedResults.csv')
    seasons = pd.read_csv('data/MDataFiles_Stage2/MSeasons.csv')
    teams = pd.read_csv('data/MDataFiles_Stage2/MTeams.csv')

    # Merge compact and detailed results
    reg_season_games_combined = pd.merge(
        left=reg_season_games_compact,
        right=reg_season_games_detailed,
        on=list(reg_season_games_compact),
        how='outer')

    # Get game dates
    reg_season_games_combined = calculateGameDates(
        reg_season_games_combined, 
        seasons
    )

    # Rename columns
    reg_season_games_combined = renameKaggleColumns(reg_season_games_combined)

    # Add team names
    reg_season_games_combined = addTeamNames(
        reg_season_games_combined, 
        teams
    )

    # Duplicate records and rename losing-perspective side, then append
    reg_season_games_combined = duplicateGameRecords(reg_season_games_combined)

    return reg_season_games_combined


#%% Main
if __name__ == '__main__':
    # Get all Kaggle data
    reg_season_games_combined = readCleanSourceData()

    # Add some details
    reg_season_games_combined = addAdditionalGameColumns(
        reg_season_games_combined)
    reg_season_games_combined = addVegasLineToKaggleData(
        reg_season_games_combined)

    # Split into detailed and not for memory saving in database
    # Find rows with >5 null values (prev was 1; some games have 0FTA
    # which screws up FTPct)
    null_rows = reg_season_games_combined.isnull().sum(axis=1) > 5
    null_data = reg_season_games_combined[null_rows]
    # drop columns with <500 populated values ()
    null_data = null_data.dropna(axis=1, thresh=len(null_data)-500)
    detailed_data = reg_season_games_combined[-null_rows]
    assert len(null_data) + len(detailed_data) == \
        len(reg_season_games_combined)

    # Read in data, convert to dict, insert records into collection
    db = get_db()
    # TODO look in to batch delete previous seasons vs entire drop
    print(f"Dropping games from MongoDB")
    db.games.drop()

    # TODO convert to function from current season scraper
    print(f"Converting {len(null_data)} non-detailed records to dict")
    null_data_dict = null_data.to_dict('records')
    print(f"Inserting {len(null_data_dict)} records to database")
    db.games.insert_many(null_data_dict, ordered=False)
    print(f"Inserted {len(null_data_dict)} records.")

    print(f"Converting {len(detailed_data)} detailed records to dict")
    detailed_data_dict = detailed_data.to_dict('records')
    print(f"Inserting {len(detailed_data_dict)} records to database")
    db.games.insert_many(detailed_data_dict, ordered=False)
    print(f"Inserted {len(detailed_data_dict)} records.")


    # Pre-aggregate all season data 
    db.seasonteams.drop()
    SEASONS = getSeasonsList(_db=db)
    for season in SEASONS:
        print(f"Working on the {season} season.")
        data = preAggSeasonGames(
            pd.DataFrame(list(db.games.find({'Season': season})))
        )
        print(f"Converting {len(data)} new season-team records to dict")
        data_dict = data.to_dict('records')
        print(f"Inserting {len(data_dict)} records to database")
        db.seasonteams.insert_many(data_dict, ordered=False)
        print(f"Inserted {len(data_dict)} records.")
