#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec  8 12:44:33 2018

@author: Ryan
"""

###############################################################################
# Create "Print Time" UDF
# Inputs: prefix (string to print); timetoprint (numeric time to print)
# Outputs: Prints the prefix plus the timetoprint, in seconds or minutes
###############################################################################


def printtime(prefix, timetoprint):
    if timetoprint < 60:
        print(prefix, str(round((timetoprint), 3)), ' sec')
    else:
        print(prefix, str(round((timetoprint)/60, 3)), ' min')
###############################################################################
###############################################################################


###############################################################################
# Create "Replace CSV" UDF
# Inputs: csvtoreplace; dataframetowrite
# Outputs: replaces the file at the location, or creates it
###############################################################################


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
###############################################################################
###############################################################################

###############################################################################
# "CreateReplace xlsx" UDF
# Inputs: exceltoreplace (the file to replace); replacementdataframe (numeric time to print)
# Outputs: replaces the file at the location
###############################################################################
def createreplacexlsx(xlsxtoreplace,dataframetowrite):
    # Assert cavtoreplace is a string and ends in xlsx
    assert type(xlsxtoreplace) == str, 'First argument is not a string'
    assert xlsxtoreplace[-5:] == '.xlsx', 'First argument is not a xlsx file'

    # Create new file string
    import re
    regex = '^(/(?:.+/)*)(.+)(.xlsx)'
    currxlsxpath = re.match(regex,xlsxtoreplace).group(1)
    currxlsxname = re.match(regex,xlsxtoreplace).group(2)

    from datetime import datetime
    now = str(datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
    newxlsxpath = currxlsxpath + 'archive/'
    newxlsxname = currxlsxname + '_ReplacedOn_' + now + '.xlsx'

    # Assert the archive path exists
    import os
    assert os.path.exists(newxlsxpath), "No 'archive' folder in the provided path"

    # Assert dataframetowrite is a dataframe
    import pandas as pd
    assert type(dataframetowrite) == pd.core.frame.DataFrame, 'Second argument is not a DataFrame'


    print('\nWriting: ' + currxlsxname + '.xlsx...')

    # Move current file to archive folder, rename file
    try:
        os.rename(xlsxtoreplace, newxlsxpath + newxlsxname)
        print('Archived current ' + currxlsxname + '.xlsx into archive folder...')
    except FileNotFoundError:
        print('No file archived; no file named ' + currxlsxname + '.xlsx at given path')

    dataframetowrite.to_excel(currxlsxpath + currxlsxname + '.xlsx', index=False)
    print('Success! Wrote dataframe to ' + currxlsxname + '.xlsx')
###############################################################################
###############################################################################


###############################################################################
# "Title" UDF
# Inputs: String to title
# Outputs: printed title
###############################################################################
def printtitle(stringtoprint):
    assert type(stringtoprint) == str, 'Please input a string'
    print('\n##################################')
    print('# ' + stringtoprint)
    print('##################################\n')
###############################################################################
###############################################################################

###############################################################################
# "dfcoldiffs" UDF
# Inputs: two dataframes, specifier of list or count
# Outputs: the column differences
###############################################################################
def dfcoldiffs(df1, df2, spec = 'list'):
    import pandas as pd
    assert type(df1) == pd.core.frame.DataFrame, 'First input is not a DataFrame'
    assert type(df2) == pd.core.frame.DataFrame, 'Second input is not a DataFrame'
    assert type(spec) == str, 'Third argument is not a string'
    assert spec in ['count','list'], 'Must request either a count or a list'

    df1_cols = sorted(list(df1))
    df2_cols = sorted(list(df2))

    coldiffs = []

    coldiffs = [['In first, not in second',sorted(list(set(df1_cols) - set(df2_cols)))]]
    coldiffs.append(['In second, not in first',sorted(list(set(df2_cols) - set(df1_cols)))])

    if spec == 'list':
        return coldiffs
    elif spec == 'count':
        return len(coldiffs[0][1]) + len(coldiffs[1][1])

###############################################################################
###############################################################################

###############################################################################
# "timer" class
###############################################################################
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
###############################################################################
###############################################################################
