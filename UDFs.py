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
