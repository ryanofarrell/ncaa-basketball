"""
Created on Mon Jul 30 20:39:24 2018

@author: Ryan O'Farrell
"""

import pandas as pd
import time
#import datetime as datetime
#from sklearn.model_selection import train_test_split
#from sklearn.ensemble import RandomForestRegressor
#import numpy as np

begin = time.time()

###############################################################################
###############################################################################
# Ingest all data up to end of 2018 season
###############################################################################
###############################################################################

rsgc = pd.read_csv('/Users/Ryan/Google Drive/ncaa-basketball-data/2018-kaggle-update/RegularSeasonCompactResults.csv')
rsgd = pd.read_csv('/Users/Ryan/Google Drive/ncaa-basketball-data/2018-kaggle-update/RegularSeasonDetailedResults.csv')








# Benchmark time
totaltime = time.time()-begin
if totaltime < 60:
    print('Total Process Time: ' + str(round((totaltime),2)) + ' sec')
else:
    print('Total Process Time: ' + str(round((totaltime)/60,2)) + ' min')

del begin, totaltime