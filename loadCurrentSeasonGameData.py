from bs4 import BeautifulSoup
import pandas as pd
from urllib.request import urlopen
import numpy as np
from requests import get
import re
from loadRegularSeasonGameData import *
from db import get_db
import math

# TODO get current season game lines
# TODO optimize database connections


def scrapeCurrSeasonCompactResults(webpage, CURRENTSEASON):
    """Scrapes compact results from Massey Ratings website

    Arguments:
        webpage {str} -- The link to this year's Massey ratings site
        Should be D1-D1 games ONLY
        CURRENTSEASON {int} -- The current season (e.g. 2020)

    Returns:
        DataFrame -- The compact results of the current season
        - GameDate
        - GameOT
        - TmID
        - TmPF
        - OppID
        - OppPF
        - Season
        - TmLoc
    """
    page = urlopen(webpage)
    soup = BeautifulSoup(page, 'lxml')
    compactResultsString = str(soup.pre)
    del webpage, page, soup

    # Import Data into DF
    x = compactResultsString.split('\n')
    compactResultsDf = pd.DataFrame(x, columns=['RawStr'])
    del compactResultsString, x

    # Remove last 4 rows
    compactResultsDf = compactResultsDf[:-4]

    # Remove/replace strings
    compactResultsDf['RawStr'].replace(
        regex=True, inplace=True, to_replace=r'&amp;', value=r'&')
    compactResultsDf['RawStr'].replace(
        regex=True, inplace=True, to_replace=r'<pre>', value=r'')

    # Split string into columns
    compactResultsDf['GameDate'] = compactResultsDf['RawStr'].str[:10]
    compactResultsDf['TmName'] = compactResultsDf['RawStr'].str[10:36]
    compactResultsDf['TmPF'] = compactResultsDf['RawStr'].str[36:39]
    compactResultsDf['OppName'] = compactResultsDf['RawStr'].str[39:65]
    compactResultsDf['OppPF'] = compactResultsDf['RawStr'].str[65:68]
    compactResultsDf['GameOT'] = pd.to_numeric(
        compactResultsDf['RawStr'].str[70:71], errors='coerce')
    compactResultsDf['GameOT'] = np.nan_to_num(compactResultsDf['GameOT'])
    del compactResultsDf['RawStr']

    # Strip Whitespaces
    compactResultsDf['GameDate'] = compactResultsDf['GameDate'].str.strip()
    compactResultsDf['TmPF'] = compactResultsDf['TmPF'].str.strip()
    compactResultsDf['TmName'] = compactResultsDf['TmName'].str.strip()
    compactResultsDf['OppName'] = compactResultsDf['OppName'].str.strip()
    compactResultsDf['OppPF'] = compactResultsDf['OppPF'].str.strip()

    compactResultsDf[['TmPF', 'OppPF']] = compactResultsDf[
        ['TmPF', 'OppPF']].apply(pd.to_numeric)

    # Calculate Margin and team locations
    compactResultsDf['TmLoc'] = ''
    compactResultsDf['Season'] = CURRENTSEASON
    compactResultsDf.loc[(compactResultsDf['TmName'].str[:1] == '@'),
                         'TmLoc'] = 'H'
    compactResultsDf.loc[(compactResultsDf['OppName'].str[:1] == '@'),
                         'TmLoc'] = 'A'
    compactResultsDf.loc[(compactResultsDf['OppName'].str[:1] != '@') &
                         (compactResultsDf['TmName'].str[:1] != '@'),
                         'TmLoc'] = 'N'

    # Remove @
    compactResultsDf['TmName'].replace(regex=True, inplace=True,
                                       to_replace=r'@', value=r'')
    compactResultsDf['OppName'].replace(regex=True, inplace=True,
                                        to_replace=r'@', value=r'')

    compactResultsDf = addTmIDAndOppID(compactResultsDf)

    # Duplicate records
    compactResultsDf = duplicateGameRecords(compactResultsDf)

    return compactResultsDf


def getNewGames(compactResults, currseason):
    """Queries the database and returns the games that are not yet in it

    Arguments:
        compactResults {DataFrame} -- The results to check against the database
        currseason {int} -- The current season (e.g. 2020)

    Returns:
        DataFrame -- All of the games that are not yet in the DataBase
    """
    # Compare all compact results to database
    db = get_db()
    query = {'Season': currseason}
    dbResults = pd.DataFrame(list(db.games.find(query)))

    if len(dbResults) > 0:
        # Convert datetime objects to strings for parsing into URL
        dbResults['GameDate'] = dbResults[
            'GameDate'].dt.strftime('%Y-%m-%d')

        comparedResults = pd.merge(compactResults, dbResults,
                                   how='left',
                                   on=list(compactResults.columns))

        rowsMissingData = comparedResults[comparedResults.isnull().any(axis=1)]
        # TODO add functionality for when there already is a record that is
        # missing data
        # For now, break here if there is anything in the _id column
        for idx, row in rowsMissingData.iterrows():
            assert isinstance(row['_id'], float), \
                'Need to delete the old record for ' + str(row['_id'])

        # Only return the columns in the initial dataframe
        rowsMissingData = rowsMissingData[list(compactResults.columns)]
        return rowsMissingData

    else:  # When there are no records for the season yet in the DB
        return compactResults


def scrapeCurrSeasonDetailedResults(webpage, datesToScrape):
    """Scrapes every game for the provided days from SportsReference.
    Gets detailed stats provided in Kaggle data.

    Arguments:
        webpage {str} -- The SportsReference base URL
        datesToScrape {DataFrame} -- Only one column 'Date', YYYY-MM-DD
            of the dates to scrape detailed data

    Returns:
        DataFrame -- Detailed results of EVERY game on the dates provided
    """
    # Get distinct dates from scraped
    datesToScrape['Year'] = datesToScrape['Date'].str[0:4].astype(int)
    datesToScrape['Month'] = datesToScrape['Date'].str[5:7].astype(int)
    datesToScrape['Day'] = datesToScrape['Date'].str[8:].astype(int)
    # print(datesToScrape)

    # Create list of webpages to scrape from later
    webpages = []
    for row in datesToScrape.iterrows():
        webpages.append(webpage +
                        "month=" + str(row[1]['Month']) +
                        "&day=" + str(row[1]['Day']) +
                        "&year=" + str(row[1]['Year']))
    del row
    # print(webpages)

    # On each of those day webpages, scrape the link to the game's results
    links = []
    for link in webpages:
        print('Working on ' + link)
        try:
            response = get(link)
            html_soup = BeautifulSoup(response.text, 'html.parser')
            games = html_soup.find_all('div', class_='game_summary nohover')
            for gamenum in range(0, len(games)):
                currgame = games[gamenum]
                currgamelink = currgame.find('td', class_='right gamelink')
                currgamelink = currgamelink.a
                links.append(str(currgamelink))
        except:  # TODO remove bare except
            print('Skipped ' + link)
            pass
    links = pd.DataFrame(links, columns=['RawStr'])
    links['GameLink'] = 'https://www.sports-reference.com' + (
                        links['RawStr'].str.extract(r'"(.*)"', expand=True))
    del links['RawStr'], webpage, gamenum, games

    # Init dataframes
    detailedResultsDf = pd.DataFrame()

    # Init variables
    GameDate = []
    TmName = []
    TmPF = []
    TmFGM = []
    TmFGA = []
    TmFG2M = []
    TmFG2A = []
    TmFG3M = []
    TmFG3A = []
    TmFTM = []
    TmFTA = []
    TmORB = []
    TmDRB = []
    TmTRB = []
    TmAst = []
    TmStl = []
    TmBlk = []
    TmTO = []
    TmFoul = []
    OppName = []
    OppPF = []
    OppFGM = []
    OppFGA = []
    OppFG2M = []
    OppFG2A = []
    OppFG3M = []
    OppFG3A = []
    OppFTM = []
    OppFTA = []
    OppORB = []
    OppDRB = []
    OppTRB = []
    OppAst = []
    OppStl = []
    OppBlk = []
    OppTO = []
    OppFoul = []

    n = 1

    # Loop through all game links
    for currgamelink in links['GameLink']:
        try:
            GameDate = currgamelink[47:57]
            print('Working on ', str(n), ' of ', str(len(links)),
                  ' - ', currgamelink[47:])
            response = get(currgamelink)
            html_soup = BeautifulSoup(response.text, 'html.parser')
            boxes = html_soup.find_all('div',
                                       id=re.compile(r'all_box-score-basic'))
            assert len(boxes) == 2, 'Can not find two boxes at' + currgamelink

            # Get Tm records into variables
            Tmfooter = boxes[0].tfoot
            TmName = str(boxes[0].h2.text)
            TmPF = int(Tmfooter.find('td', attrs={'data-stat': "pts"}).text)
            TmFGM = int(Tmfooter.find('td', attrs={'data-stat': "fg"}).text)
            TmFGA = int(Tmfooter.find('td', attrs={'data-stat': "fga"}).text)
            TmFG2M = int(Tmfooter.find('td', attrs={'data-stat': "fg2"}).text)
            TmFG2A = int(Tmfooter.find('td', attrs={'data-stat': "fg2a"}).text)
            TmFG3M = int(Tmfooter.find('td', attrs={'data-stat': "fg3"}).text)
            TmFG3A = int(Tmfooter.find('td', attrs={'data-stat': "fg3a"}).text)
            TmFTM = int(Tmfooter.find('td', attrs={'data-stat': "ft"}).text)
            TmFTA = int(Tmfooter.find('td', attrs={'data-stat': "fta"}).text)
            TmORB = int(Tmfooter.find('td', attrs={'data-stat': "orb"}).text)
            TmDRB = int(Tmfooter.find('td', attrs={'data-stat': "drb"}).text)
            TmTRB = int(Tmfooter.find('td', attrs={'data-stat': "trb"}).text)
            TmAst = int(Tmfooter.find('td', attrs={'data-stat': "ast"}).text)
            TmStl = int(Tmfooter.find('td', attrs={'data-stat': "stl"}).text)
            TmBlk = int(Tmfooter.find('td', attrs={'data-stat': "blk"}).text)
            TmTO = int(Tmfooter.find('td', attrs={'data-stat': "tov"}).text)
            TmFoul = int(Tmfooter.find('td', attrs={'data-stat': "pf"}).text)

            # Get Opp records into variables
            Oppfooter = boxes[1].tfoot
            OppName = str(boxes[1].h2.text)
            OppPF = int(Oppfooter.find('td', attrs={'data-stat': "pts"}).text)
            OppFGM = int(Oppfooter.find('td', attrs={'data-stat': "fg"}).text)
            OppFGA = int(Oppfooter.find('td', attrs={'data-stat': "fga"}).text)
            OppFG2M = int(Oppfooter.find('td', attrs={'data-stat': "fg2"}).text)
            OppFG2A = int(Oppfooter.find('td', attrs={'data-stat': "fg2a"}).text)
            OppFG3M = int(Oppfooter.find('td', attrs={'data-stat': "fg3"}).text)
            OppFG3A = int(Oppfooter.find('td', attrs={'data-stat': "fg3a"}).text)
            OppFTM = int(Oppfooter.find('td', attrs={'data-stat': "ft"}).text)
            OppFTA = int(Oppfooter.find('td', attrs={'data-stat': "fta"}).text)
            OppORB = int(Oppfooter.find('td', attrs={'data-stat': "orb"}).text)
            OppDRB = int(Oppfooter.find('td', attrs={'data-stat': "drb"}).text)
            OppTRB = int(Oppfooter.find('td', attrs={'data-stat': "trb"}).text)
            OppAst = int(Oppfooter.find('td', attrs={'data-stat': "ast"}).text)
            OppStl = int(Oppfooter.find('td', attrs={'data-stat': "stl"}).text)
            OppBlk = int(Oppfooter.find('td', attrs={'data-stat': "blk"}).text)
            OppTO = int(Oppfooter.find('td', attrs={'data-stat': "tov"}).text)
            OppFoul = int(Oppfooter.find('td', attrs={'data-stat': "pf"}).text)

            detailedResultsDf = detailedResultsDf.append(
                    {'GameDate': GameDate,
                     'TmName': TmName,
                     'TmPF': TmPF,
                     'TmFGM': TmFGM,
                     'TmFGA': TmFGA,
                     'TmFG2M': TmFG2M,
                     'TmFG2A': TmFG2A,
                     'TmFG3M': TmFG3M,
                     'TmFG3A': TmFG3A,
                     'TmFTM': TmFTM,
                     'TmFTA': TmFTA,
                     'TmORB': TmORB,
                     'TmDRB': TmDRB,
                     'TmTRB': TmTRB,
                     'TmAst': TmAst,
                     'TmStl': TmStl,
                     'TmBlk': TmBlk,
                     'TmTO': TmTO,
                     'TmFoul': TmFoul,
                     'OppName': OppName,
                     'OppPF': OppPF,
                     'OppFGM': OppFGM,
                     'OppFGA': OppFGA,
                     'OppFG2M': OppFG2M,
                     'OppFG2A': OppFG2A,
                     'OppFG3M': OppFG3M,
                     'OppFG3A': OppFG3A,
                     'OppFTM': OppFTM,
                     'OppFTA': OppFTA,
                     'OppORB': OppORB,
                     'OppDRB': OppDRB,
                     'OppTRB': OppTRB,
                     'OppAst': OppAst,
                     'OppStl': OppStl,
                     'OppBlk': OppBlk,
                     'OppTO': OppTO,
                     'OppFoul': OppFoul
                     }, ignore_index=True)

            detailedResultsDf = detailedResultsDf.append(
                    {'GameDate': GameDate,
                     'TmName': OppName,
                     'TmPF': OppPF,
                     'TmFGM': OppFGM,
                     'TmFGA': OppFGA,
                     'TmFG2M': OppFG2M,
                     'TmFG2A': OppFG2A,
                     'TmFG3M': OppFG3M,
                     'TmFG3A': OppFG3A,
                     'TmFTM': OppFTM,
                     'TmFTA': OppFTA,
                     'TmORB': OppORB,
                     'TmDRB': OppDRB,
                     'TmTRB': OppTRB,
                     'TmAst': OppAst,
                     'TmStl': OppStl,
                     'TmBlk': OppBlk,
                     'TmTO': OppTO,
                     'TmFoul': OppFoul,
                     'OppName': TmName,
                     'OppPF': TmPF,
                     'OppFGM': TmFGM,
                     'OppFGA': TmFGA,
                     'OppFG2M': TmFG2M,
                     'OppFG2A': TmFG2A,
                     'OppFG3M': TmFG3M,
                     'OppFG3A': TmFG3A,
                     'OppFTM': TmFTM,
                     'OppFTA': TmFTA,
                     'OppORB': TmORB,
                     'OppDRB': TmDRB,
                     'OppTRB': TmTRB,
                     'OppAst': TmAst,
                     'OppStl': TmStl,
                     'OppBlk': TmBlk,
                     'OppTO': TmTO,
                     'OppFoul': TmFoul
                     }, ignore_index=True)
        except TypeError:
            print('Skipping number ' + str(n))
        n = n + 1

    del TmName, TmPF, TmFGM, TmFGA, TmFG2M, TmFG2A, TmFG3M, TmFG3A, TmFTM
    del TmFTA, TmORB, TmDRB, TmTRB, TmAst, TmStl, TmBlk, TmFoul, TmTO, OppTO
    del OppName, OppPF, OppFGM, OppFGA, OppFG2M, OppFG2A, OppFG3M, OppFG3A
    del OppFTM, OppFTA, OppORB, OppDRB, OppTRB, OppAst, OppStl, OppBlk, OppFoul
    del GameDate, n
    del boxes, currgamelink

    # Remove records from team names
    detailedResultsDf['TmName'].replace(regex=True, inplace=True,
                                        to_replace=r'( \([0-9]+-[0-9]+\))',
                                        value=r'')
    detailedResultsDf['OppName'].replace(regex=True, inplace=True,
                                         to_replace=r'( \([0-9]+-[0-9]+\))',
                                         value=r'')

    # Merge in team IDs
    detailedResultsDf = addTmIDAndOppID(detailedResultsDf)

    return detailedResultsDf


def addTmIDAndOppID(data):
    """For a variety of team name spellings, gets the correct TmID and OppID,
    and drops the potentially misspelled TmName and OppName

    Arguments:
        data {DataFrame} -- containing columns of TmName and OppName for which
        we want to get the correct TmID and OppID

    Returns:
        DataFrame -- Original DataFrame + TmID and OppID - TmName and OppName
    """
    assert 'TmName' in data.columns, 'Missing TmName col'
    assert 'OppName' in data.columns, 'Missing OppName col'
    # Get the team IDs from file that has many different
    # ways to spell the team name
    fp = 'data/TeamSpellings.csv'
    teamspellings = pd.read_csv(filepath_or_buffer=fp, encoding="ISO-8859-1")
    del fp

    preMergeLen = len(data)
    teamspellings = teamspellings.rename(columns={'TeamID': 'TmID'})
    data['TeamNameSpelling'] = data['TmName'].str.lower()
    data = pd.merge(data, teamspellings,
                    how='left', on=['TeamNameSpelling'])
    del data['TeamNameSpelling']
    teamspellings = teamspellings.rename(columns={'TmID': 'OppID'})
    data['TeamNameSpelling'] = data['OppName'].str.lower()
    data = pd.merge(data, teamspellings,
                    how='left', on=['TeamNameSpelling'])
    del data['TeamNameSpelling'], data['TmName'], data['OppName']

    # Sanity check results, make sure there are no lost
    # or missing values
    assert len(data) == preMergeLen, 'Some records lost at team name spellings'

    return data


def insertNewGameRecords(df):
    """Adds new game records to database

    Arguments:
        df {DataFrame} -- Records to be added

    Prints when results are added.
    """
    print(f"Converting {len(df)} new records to dict")
    data_dict = df.to_dict('records')
    db = get_db()
    print(f"Inserting {len(data_dict)} records to database")
    db.games.insert_many(data_dict, ordered=False)
    print(f"Inserted {len(data_dict)} records.")


def dataQualityAdjustments(compactData, detailedData):
    # Adjust Hawaii-San Francisco from 11/29/19 to 11/30/19
    # Hawaii = 1218, San Fran = 1362
    detailedData.loc[
        (detailedData['TmID'] == 1218) &
        (detailedData['OppID'] == 1362) &
        (detailedData['GameDate'] == '2019-11-29'), 'GameDate'] = '2019-11-30'
    detailedData.loc[
        (detailedData['TmID'] == 1362) &
        (detailedData['OppID'] == 1218) &
        (detailedData['GameDate'] == '2019-11-29'), 'GameDate'] = '2019-11-30'

    return compactData, detailedData


if __name__ == '__main__':
    # Massey URL only takes D1-D1 games
    MASSEYURL = 'https://www.masseyratings.com/scores.php?s=309912&sub=11590&all=1&mode=2&format=0'
    SPORTSREFURL = 'https://www.sports-reference.com/cbb/boxscores/index.cgi?'
    CURRENTSEASON = 2020  # spring season convention

    # Get list of all the games played this year
    df = scrapeCurrSeasonCompactResults(MASSEYURL, CURRENTSEASON)

    # Whittle that list down to games without details in DB
    df = getNewGames(df, CURRENTSEASON)

    if len(df) > 0:
        # Get the dates missing details
        compactResultsUniqueDates = pd.DataFrame(df['GameDate'].unique(),
                                                 columns=['Date'])

        # Scrape ALL the games for the dates where games are missing details
        df2 = scrapeCurrSeasonDetailedResults(SPORTSREFURL,
                                              compactResultsUniqueDates)

        columnsToMergeOn = ['TmID', 'OppID', 'GameDate']

        # Handle date mismatches (usually for games in Hawaii)
        df, df2 = dataQualityAdjustments(df, df2)

        # Merge left to use Massey as the ultimate source for data, with
        # SportsRef to augment with details.
        # Note that df only contains the games missing details
        # This limits the records to insert down to ONLY the new ones on
        # the dates given.
        regSeasonGamesDetailed = pd.merge(df, df2,
                                          on=columnsToMergeOn,
                                          how='left')

        # Format gamedate as datetime
        regSeasonGamesDetailed['GameDate'] = pd.to_datetime(
            regSeasonGamesDetailed['GameDate'], format='%Y/%m/%d')

        # Note there are some discrepancies between Massey & SR
        # We will use Massey as the source or record but track diffs
        diffPointsAcrossSources = regSeasonGamesDetailed.loc[
            (regSeasonGamesDetailed['TmPF_x'] != regSeasonGamesDetailed[
                'TmPF_y']) |
            (regSeasonGamesDetailed['OppPF_x'] != regSeasonGamesDetailed[
                'OppPF_y'])]
        del regSeasonGamesDetailed['TmPF_y'], regSeasonGamesDetailed['OppPF_y']
        regSeasonGamesDetailed = regSeasonGamesDetailed.rename(columns={
            'TmPF_x': 'TmPF',
            'OppPF_x': 'OppPF'})

        regSeasonGamesDetailed = addAdditionalGameColumns(
            regSeasonGamesDetailed)
        teams = pd.read_csv('data/Stage2DataFiles/Teams.csv')
        regSeasonGamesDetailed = addTeamNames(regSeasonGamesDetailed, teams)

        gamesMissingDetails = regSeasonGamesDetailed.loc[
            regSeasonGamesDetailed['TmAst'].isnull()]
        countMissingDetails = len(gamesMissingDetails)
        assert countMissingDetails == 0, 'Some new games missing details'

        insertNewGameRecords(regSeasonGamesDetailed)
    else:  # If there are no new games since this was last run
        print("No new records!")