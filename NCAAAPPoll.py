#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 16 20:18:29 2018

@author: Ryan
"""


from bs4 import BeautifulSoup
from urllib.request import urlopen
from requests import get
import pandas as pd
from functools import reduce
import re
import numpy as np
from UDFs import createreplacecsv


# TODO assertion with crazy table strutures
# TODO why are there no records in any of the polls?
# TODO export data to CSV
# TODO handle differences between NR/RV in Coach Poll Results (1171 is an example)


rankhistory = pd.DataFrame()
firstpoll = 1150 #539
lastpoll = 1172 #1171
pollnumrange = range(firstpoll,lastpoll)

# 
for pollnum in pollnumrange:
    
    # Pull Web Data HTML
    webpage = "http://www.collegepollarchive.com/mbasketball/ap/seasons.cfm?appollid=" + str(pollnum)
    response = get(webpage)
    html_soup = BeautifulSoup(response.text, 'html.parser')
       
    # Get poll name from header
    header = html_soup.find_all('h2')
    pollname = str(header)
    pollname = re.search('<h2>(.+)<.h2>',pollname).group(1)
    print("Poll: " + pollname)
    del header

    # Initailize dfs for later merging
    dfs = []
    currpoll = pd.DataFrame()

    # Get rank output from HTML
    appollrankoutput = html_soup.find_all('td',class_ = 'td-row td-poll-rank')
    appollrank = []
    for x in appollrankoutput:
        appollrank.append(str(x))
    del appollrankoutput
    
    currpoll['APPollRank'] = appollrank
    
    del appollrank
    
    # Get prevrank, team, and conference output from HTML
    prevrank_team_conf_output = html_soup.find_all('td',class_ = 'td-row td-left')
    prevrank_team_conf = []
    for x in prevrank_team_conf_output:
        prevrank_team_conf.append(str(x))
    del prevrank_team_conf_output
  
    currpoll['PrevAPPollRank'] = prevrank_team_conf[0::3]   
    currpoll['Team'] = prevrank_team_conf[1::3]
    currpoll['Conf'] = prevrank_team_conf[2::3]
    
    del prevrank_team_conf
    
    # Get record, points from HTML
    record_points_output = html_soup.find_all('td',class_ = 'td-row td-right')
    record_points = []
    for x in record_points_output:
        record_points.append(str(x))
    del record_points_output
    
    # Check if remaining output has 3 data points or two, and adjust accordingly
    if len(record_points) / len(currpoll) == 3:
        currpoll['Record'] = record_points[0::3]
        currpoll['Points'] = record_points[1::3]
        currpoll['CoachPollRank'] = record_points[1::3]
        
    elif len(record_points) / len(currpoll) == 2:
    
        currpoll['Record'] = record_points[0::2]
        currpoll['Points'] = record_points[1::2]
        currpoll['CoachPollRank'] = np.nan
    del record_points
    
    # Split columns on regex to get useful data
    currpoll['APPollRank'] = currpoll['APPollRank'].str.extract(r'<strong>(.+)<.strong>'
                                                           ,expand=True)
    currpoll['Team'] = currpoll['Team'].str.extract(r'td-left"><a href.*">(.*)<.a>'
                                                            ,expand=True)
    currpoll['PrevAPPollRank'] = currpoll['PrevAPPollRank'].str.extract(r'>&lt;.(.*)<.td>'
                                                            ,expand=True)
    currpoll['Conf'] = currpoll['Conf'].str.extract(r'td-left">(?:<abbr tit.*">)?(.*)(?:<\/abbr>)?<\/td>'
                                                            ,expand=True)
    currpoll['Conf'].replace(regex=True, inplace=True, to_replace=r'</abbr>', value=r'')
    currpoll['Record'] = currpoll['Record'].str.extract(r'td-right">(.*)<.td>'
                                                            ,expand=True)
    currpoll['Points'] = currpoll['Points'].str.extract(r'td-right">(.*)<.td>'
                                                            ,expand=True)
    try:
        currpoll['CoachPollRank'] = currpoll['CoachPollRank'].str.extract(
                                                            r'td-right">(?:\n)?(?:<abbr tit.*">*)?(?:\s*)(.*)(?:\s*)(?:\n)(?:\s*)<\/td>'
                                                                ,expand=True)
        currpoll['CoachPollRank'].replace(regex=True, inplace=True, to_replace=r'</abbr>', value=r'')
        currpoll['CoachPollRank'].replace(regex=True, inplace=True, to_replace=r'[^0-9]', value=r'')
    except AttributeError:
        pass
    currpoll['PollName'] = pollname
    currpoll['PollNum'] = pollnum
    
    rankhistory = rankhistory.append(currpoll)

# modify rankhistory output
rankhistory['APPollRank'] = pd.to_numeric(rankhistory['APPollRank'])
rankhistory['PrevAPPollRank'] = pd.to_numeric(rankhistory['PrevAPPollRank'],
                                               errors = 'coerce')

rankhistory['CoachPollRank'] = pd.to_numeric(rankhistory['CoachPollRank'])
rankhistory['Points'] = pd.to_numeric(rankhistory['Points'])

createreplacecsv('/Users/Ryan/Google Drive/ncaa-basketball-data/test.csv',rankhistory)
