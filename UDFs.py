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
def printtime(prefix,timetoprint):
    if timetoprint < 60:
        print(prefix + str(round((timetoprint),3)) + ' sec')
    else:
        print(prefix + str(round((timetoprint)/60,3)) + ' min')
###############################################################################
###############################################################################


###############################################################################
# Create "Replace CSV" UDF
# Inputs: csvtoreplace (the file to replace); replacementdataframe (numeric time to print)
# Outputs: replaces the file at the location
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


    print('Writing: ' + csvtoreplace)

    # Move current file to archive folder, rename file
    try:
        os.rename(csvtoreplace, newcsvpath + newcsvname)
        print('Archived current ' + currcsvname + '.csv into archive/ folder as ' + newcsvname)
    except FileNotFoundError:
        print('No file archived; no file named ' + currcsvname + '.csv at given path')
    
    dataframetowrite.to_csv(currcsvpath + currcsvname + '.csv', index=False)
    print('Wrote dataframe to ' + currcsvname + '.csv in ' + currcsvpath)
###############################################################################
###############################################################################
