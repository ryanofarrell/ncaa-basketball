#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 16 20:18:29 2018

@author: Ryan
"""


from bs4 import BeautifulSoup
from requests import get
import pandas as pd
import re
import numpy as np
from UDFs import createreplacecsv, printtitle
from datetime import timedelta

printtitle('Running AP poll scraper')


# TODO assertion with crazy table strutures
# TODO handle diffs between NR/RV in Coach Poll Results (1171 is  example)
# TODO automate ingestion, checking for new poll, appending, and rewriting

try:
    rankhistory = pd.read_csv(
            '/Users/Ryan/Google Drive/ncaa-basketball-data/appollhistory.csv')
    firstpoll = rankhistory['PollNum'].max() + 1
    importedlen = len(rankhistory)
    print('Imported appollhistory.csv, beginning scrape at poll #',
          str(firstpoll))
except:
    rankhistory = pd.DataFrame()
    firstpoll = 539  # 539
    print('No appollhistory.csv found, beginning scrape at poll #',
          str(firstpoll))


lastpoll = 1180  # 1171
pollnumrange = range(firstpoll, lastpoll)

# Loop through each poll
for pollnum in pollnumrange:
    # Try to do the scrape, if successful, append to rankhistory
    try:
        print('Trying scrape of poll #' + str(pollnum))

        # Pull Web Data HTML
        webpage = "http://www.collegepollarchive.com/mbasketball/ap/seasons.cfm?appollid=" + str(pollnum)
        response = get(webpage)
        html_soup = BeautifulSoup(response.text, 'html.parser')

        # Get poll name from header
        header = html_soup.find_all('h2')
        pollname = str(header)
        pollname = re.search('<h2>(.+)<.h2>', pollname).group(1)
        del header

        # Initailize dfs for later merging
        dfs = []
        currpoll = pd.DataFrame()

        # Get rank output from HTML
        appollrankoutput = html_soup.find_all('td',
                                              class_='td-row td-poll-rank')
        appollrank = []
        for x in appollrankoutput:
            appollrank.append(str(x))
        del appollrankoutput

        currpoll['APPollRank'] = appollrank

        del appollrank

        # Get prevrank, team, and conference output from HTML
        prevrank_team_conf_output = html_soup.find_all('td',
                                                       class_='td-row td-left')
        prevrank_team_conf = []
        for x in prevrank_team_conf_output:
            prevrank_team_conf.append(str(x))
        del prevrank_team_conf_output

        currpoll['PrevAPPollRank'] = prevrank_team_conf[0::3]
        currpoll['Team'] = prevrank_team_conf[1::3]
        currpoll['Conf'] = prevrank_team_conf[2::3]

        del prevrank_team_conf

        # Get record, points from HTML
        record_points_output = html_soup.find_all('td',
                                                  class_='td-row td-right')
        record_points = []
        for x in record_points_output:
            record_points.append(str(x))
        del record_points_output

        # Check if output has 3 data points or two, adjust accordingly
        if len(record_points) / len(currpoll) == 3:
            currpoll['Record'] = record_points[0::3]
            currpoll['Points'] = record_points[1::3]
            currpoll['CoachPollRank'] = record_points[2::3]

        elif len(record_points) / len(currpoll) == 2:
            currpoll['Record'] = record_points[0::2]
            currpoll['Points'] = record_points[1::2]
            currpoll['CoachPollRank'] = np.nan

        del record_points

        # Split columns on regex to get useful data
        currpoll['APPollRank'] = (
                currpoll['APPollRank']
                .str
                .extract(r'<strong>(.+)<.strong>', expand=True))
        currpoll['Team'] = (
                currpoll['Team']
                .str
                .extract(r'td-left"><a href.*">(.*)<.a>', expand=True))
        currpoll['PrevAPPollRank'] = (
                currpoll['PrevAPPollRank']
                .str
                .extract(r'>&lt;.(.*)<.td>', expand=True))
        currpoll['Conf'] = (
                currpoll['Conf']
                .str
                .extract(r'td-left">(?:<abbr tit.*">)?(.*)(?:<\/abbr>)?<\/td>',
                         expand=True))
        currpoll['Conf'].replace(regex=True, inplace=True,
                                 to_replace=r'</abbr>', value=r'')
        currpoll['Record'] = (
                currpoll['Record']
                .str
                .extract(r'td-right">(.*)<.td>', expand=True))
        currpoll['Points'] = (
                currpoll['Points']
                .str
                .extract(r'td-right">(.*)<.td>', expand=True))
        try:
            REGEX = re.compile(r'td-right">(?:\n)'
                               r'?(?:<abbr tit.*">*)'
                               r'?(?:\s*)(.*)(?:\s*)'
                               r'(?:\n)(?:\s*)<\/td>'
                               )
            currpoll['CoachPollRank'] = (
                    currpoll['CoachPollRank']
                    .str
                    .extract(REGEX, expand=True))
            currpoll['CoachPollRank'].replace(regex=True, inplace=True,
                                              to_replace=r'[^0-9]',
                                              value=r'')
            del REGEX
        except AttributeError:
            pass
        currpoll['PollName'] = pollname
        currpoll['PollNum'] = pollnum

        rankhistory = rankhistory.append(currpoll)
        print('Successful scrape...')
    # If scrape is unsuccessful, change while loop condition to exit
    except:
        print('Scrape failed at poll #' + str(pollnum))
        break

# Modify rankhistory output
rankhistory = rankhistory.reset_index(drop=True)

rankhistory['APPollRank'] = pd.to_numeric(rankhistory['APPollRank'])
rankhistory['PrevAPPollRank'] = pd.to_numeric(rankhistory['PrevAPPollRank'],
                                              errors='coerce')

rankhistory['CoachPollRank'] = pd.to_numeric(rankhistory['CoachPollRank'])
rankhistory['Points'] = pd.to_numeric(rankhistory['Points'])

# Split poll name
rankhistory['PollDate'] = (
        rankhistory['PollName']
        .str
        .extract(r'(.*).AP Men.*', expand=True))
# TODO get dates of preseason and final polls into PollDate

# Extract season from preseason/final
rankhistory.loc[
        (rankhistory['PollDate'].str.contains('Preseason') |
         rankhistory['PollDate'].str.contains('Final')),
        'Season'] = rankhistory['PollDate'].str[0:4]

# Extract season from in-season polls
rankhistory['temp'] = (
        pd.to_datetime(rankhistory['PollDate'], errors='coerce')
        + timedelta(days=180))
rankhistory.loc[rankhistory['temp'].notnull(), 'Season'] = (
        rankhistory['temp'].astype(str).str[0:4])

del rankhistory['temp']

rankhistory['Season'] = pd.to_numeric(rankhistory['Season'])
if len(rankhistory) == importedlen:
    print('Not writing new file, no records added to dataframe...')
else:
    createreplacecsv('/Users/Ryan/Google Drive/ncaa-basketball-data/appollhistory.csv',rankhistory)

