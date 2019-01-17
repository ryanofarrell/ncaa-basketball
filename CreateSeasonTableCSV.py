#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 21 11:03:20 2018

@author: Ryan
"""


import pandas as pd
from UDFs import createreplacecsv
# TODO morgan state vs howard wtf?

seasonteams = pd.read_csv('/Users/Ryan/Google Drive/ncaa-basketball-data/seasonteams.csv')

inclcolumns = ['Season'
               ,'TmName'
               ,'TmGame'
               ,'TmWin'
               ,'TmLoss'
               ,'TmSoS'
#               ,'Rank_TmSoS'
               ,'TourneyWin'
               ,'TourneyGame'
               ,'PlayInWin'
               ,'PlayInGame'
               ,'TourneySeed'
               ,'TourneyResultStr'
               ]


metrics = ['Margin'
           ,'PF'
           ]

suffixes = [
            'perGame'
            ,'per40'
            ,'perPoss'
            ]

inclmetrics = []

for prefix in ['Tm','Opp']:
    for metric in metrics:
        for suffix in suffixes:
            inclmetrics.append(prefix + metric + suffix)
            inclmetrics.append('OA_' + prefix + metric + suffix)
#            inclmetrics.append('Rank_' + prefix + metric + suffix)
#            inclmetrics.append('Rank_OA_' + prefix + metric + suffix)

for metric in inclmetrics:
    inclcolumns.append(metric)
    
del metric, prefix, suffix

# Initial trim of crazy columns
seasontable = seasonteams[inclcolumns]



testst = seasontable.loc[seasontable['TmName'] == 'Florida']
testst = seasontable.loc[seasontable['Season'] == 2019]

createreplacecsv('/Users/Ryan/Google Drive/ncaa-basketball-data/seasontable.csv',seasontable)
