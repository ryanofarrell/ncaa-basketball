
from db import get_db
import pandas as pd
from math import isnan

def calculate_possessions(reg_season_games):
    """Add field for number of possessions (BasketballReference Method) \n
    https://www.basketball-reference.com/about/glossary.html
    Returns the dataframe it is provided with two new fields: TmPoss and OppPoss
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


def calculate_game_dates(reg_season_games, seasons):
    """Takes dataframe of regular season games and determines the game date
    Uses day 0 from seasons and adds a timedelta
    """
    reg_season_games = pd.merge(reg_season_games, seasons[['Season', 'DayZero']], on='Season')
    reg_season_games['DayZero'] = pd.to_datetime(reg_season_games['DayZero'], format='%m/%d/%Y')
    reg_season_games['DayNum'] = pd.to_timedelta(reg_season_games['DayNum'], unit='d')
    reg_season_games['GameDate'] = reg_season_games['DayZero'] + reg_season_games['DayNum']
    del reg_season_games['DayZero'], reg_season_games['DayNum']

    return reg_season_games


def renameKaggleColumns(reg_season_games):
    """Takes dataframe and:
    1) renames the columns to understood naming convention
    2) duplicates the records, flips the winners/losers, and appends
    This results in a dataframe that has a TmGame record for every game a team played,\
    regardless if they won or lost.
    """
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

    return reg_season_games


def duplicateKaggleRecords(rsg):
    # Copy, rename, and append the other half of the games to reg_season_games
    loser_rsg = rsg.copy()
    newnames = pd.DataFrame(list(loser_rsg), columns=['OldName'])
    newnames['NewName'] = newnames['OldName']
    newnames.loc[newnames['OldName'].str[0:3] == 'Opp', 'NewName'] = 'Tm' + newnames['OldName'].str[3:]
    newnames.loc[newnames['OldName'].str[0:2] == 'Tm', 'NewName'] = 'Opp' + newnames['OldName'].str[2:]
    newnames = newnames.set_index('OldName')['NewName']
    loser_rsg = loser_rsg.rename(columns=newnames)
    loser_rsg['TmLoc'] = 'N'
    loser_rsg.loc[loser_rsg['OppLoc'] == 'H', 'TmLoc'] = 'A'
    loser_rsg.loc[loser_rsg['OppLoc'] == 'A', 'TmLoc'] = 'H'
    del loser_rsg['OppLoc']
    rsg = rsg.append(loser_rsg, sort=True)

    return rsg


def addAdditionalGameColumns(rsg):
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

    rsg = calculate_possessions(rsg)

    return rsg


def addTeamNamesToKaggleData(rsg, teams):
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

    # Rename columns
    reg_season_games_combined = renameKaggleColumns(reg_season_games_combined)

    # Add team names
    reg_season_games_combined = addTeamNamesToKaggleData(reg_season_games_combined, teams)

    # Duplicate records and rename losing-perspective side, then append
    reg_season_games_combined = duplicateKaggleRecords(reg_season_games_combined)

    return reg_season_games_combined


reg_season_games_combined = read_and_clean_source_data()

reg_season_games_combined = addAdditionalGameColumns(reg_season_games_combined)

db = get_db()

# Read in data, convert to dict, insert records into collection
print(f"Dropping games from MongoDB")
db.games.drop()

# Split into detailed and not for memory saving in database
null_data = reg_season_games_combined[reg_season_games_combined.isnull().any(axis=1)]
null_data = null_data.dropna(axis=1)
print(f"Converting {len(null_data)} non-detailed records to dict")
null_data_dict = null_data.to_dict('records')
print(f"Inserting {len(null_data_dict)} records to database")
db.games.insert_many(null_data_dict, ordered=False)
print(f"Inserted {len(null_data_dict)} records.")

detailed_data = reg_season_games_combined[-reg_season_games_combined.isnull().any(axis=1)]
print(f"Converting {len(detailed_data)} detailed records to dict")
detailed_data_dict = detailed_data.to_dict('records')
print(f"Inserting {len(detailed_data_dict)} records to database")
db.games.insert_many(detailed_data_dict, ordered=False)
print(f"Inserted {len(detailed_data_dict)} records.")


