#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 30 20:39:24 2018

@author: Ryan
"""

import pandas as pd
import time
import datetime as datetime
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import numpy as np

begin = time.time()

# Create list of matrics for further use
metrics = ['PF','Margin','FGM','FGA',
            'FGM3','FGA3','FGM2','FGA2','Ast','OR','DR','TR',
            'FTA','FTM','Blk','Foul']

###############################################################################
###############################################################################
# Define opponentadjust UDF (for ease of opponent-adjusting metrics)
###############################################################################
###############################################################################
def opponentadjust(OAmetric):
    global irsg, iteams
    
    # Figure out the prefix, core metric, for use later
    if OAmetric[:2] == 'Tm':
        prefix = OAmetric[:2]
        otherprefix = 'Opp'
        coremetric = OAmetric[2:]    
    if OAmetric[:3] == 'Opp':
        prefix = OAmetric[:3]
        otherprefix = 'Tm'
        coremetric = OAmetric[3:]
    # print (coremetric + prefix)
    
    # From iteams put average PF into opponent side of irsg
    # Example, Opp_AvgPF_Against, Opp_AvgPA_Against
    # If I am OAing TmPFper40 (my offense proficiency), I want to get OppPFper40 for my opponent
    # So, for a TmName, get their OppPFper40, aka their PAper40
    tempiteams = iteams[['TmName',otherprefix+coremetric]]
    
    # Rename my opponent's metric to say it's *their* average <insert metric>
    # Rename to OppAvg_OppScoreper40 (it's my opponent's average opponents (me) score per 40)
    tempiteams = tempiteams.rename(columns = {otherprefix+coremetric:'OppAvg_'+otherprefix+coremetric})

    # Merge in this info into irsg, for the opponent in irsg
    irsg = pd.merge(irsg,tempiteams,left_on='OppName',right_on='TmName',how='left',suffixes=('','_y'))
    del irsg['TmName_y']
    
    # In irsg, determine for that game how the Tm did vs Opp_Avg's
    # Example, GameOppAdj_TmPFper40 = TmPFper40 - OppAvg_OppPFper40
    irsg['GameOppAdj_'+OAmetric] = irsg[OAmetric] - irsg['OppAvg_'+otherprefix+coremetric]

    # switch it for when you start with an opponent
    if prefix == 'Opp':
        irsg['GameOppAdj_'+OAmetric] = irsg['GameOppAdj_'+OAmetric] * -1
    
    # In iteamstemp, sum the opponent-adjusted metric and get a new average
    # Example, sum(GameOppAdj_TmPFper40) gets you the TOTAL OA_PFper40
    iteamstemp = irsg.groupby(['TmName'])['GameOppAdj_'+OAmetric].sum().reset_index()

    # bring that value back into iteams
    iteams = pd.merge(iteams,iteamstemp,left_on='TmName',right_on='TmName',how='left',suffixes=('','_y'))
    iteams = iteams.rename(columns = {'GameOppAdj_'+OAmetric:'OA_'+OAmetric})
    iteams['OA_'+OAmetric] = iteams['OA_'+OAmetric] / iteams['GameMins'] * 40
#    del iteams['TmName_y']


###############################################################################
###############################################################################
# Read in raw game data, clean up, manipulate for further analysis
###############################################################################
###############################################################################
def creatersg():
    # Set start time
    rsgbegin = time.time()
    
    global metrics
    
    # Read in regular season games (rsg), seasons, teams
    rsg = pd.read_csv(filepath_or_buffer = '/Users/Ryan/Desktop/DataFiles/PrelimData2018/RegularSeasonDetailedResults_Prelim2018.csv')
    seasons = pd.read_csv(filepath_or_buffer = '/Users/Ryan/Desktop/DataFiles/Seasons.csv')
    teams = pd.read_csv(filepath_or_buffer = '/Users/Ryan/Desktop/DataFiles/Teams.csv')
    
    # Merge in day 0 to rsg, add days to day 0 to get date of game
    rsg = pd.merge(rsg,seasons[['Season','DayZero']],on='Season')
    rsg['DayZero'] = pd.to_datetime(rsg['DayZero'],format='%m/%d/%Y')
    rsg['DayNum'] = pd.to_timedelta(rsg['DayNum'],unit='d')
    rsg['GameDate'] = rsg['DayZero'] + rsg['DayNum']
    del rsg['DayNum'], rsg['DayZero']
    
    # Duplicate rsg for renaming
    lrsg = rsg.copy()
    
    # Rename columns for rsg
    rsg = rsg.rename(columns = {'WTeamID':'TmID','WScore':'TmPF','LTeamID':'OppID','LScore':'OppPF','WLoc':'TmLoc'})
    rsg = rsg.rename(columns = {'WFGM':'TmFGM','WFGA':'TmFGA','WFGM3':'TmFGM3','WFGA3':'TmFGA3','WFTM':'TmFTM','WFTA':'TmFTA'})
    rsg = rsg.rename(columns = {'WOR':'TmOR','WDR':'TmDR','WAst':'TmAst','WTo':'TmTO','WFTM':'TmFTM','WFTA':'TmFTA'})
    rsg = rsg.rename(columns = {'WTO':'TmTO','WStl':'TmStl','WBlk':'TmBlk','WPF':'TmFoul'})
    rsg = rsg.rename(columns = {'LFGM':'OppFGM','LFGA':'OppFGA','LFGM3':'OppFGM3','LFGA3':'OppFGA3','LFTM':'OppFTM','LFTA':'OppFTA'})
    rsg = rsg.rename(columns = {'LOR':'OppOR','LDR':'OppDR','LAst':'OppAst','LTo':'OppTO','LFTM':'OppFTM','LFTA':'OppFTA'})
    rsg = rsg.rename(columns = {'LTO':'OppTO','LStl':'OppStl','LBlk':'OppBlk','LPF':'OppFoul'})
    rsg['TmWin'] = 1
    
    
    # Rename columns for lrsg
    lrsg = lrsg.rename(columns = {'WTeamID':'OppID','WScore':'OppPF','LTeamID':'TmID','LScore':'TmPF'})
    lrsg = lrsg.rename(columns = {'WFGM':'OppFGM','WFGA':'OppFGA','WFGM3':'OppFGM3','WFGA3':'OppFGA3','WFTM':'OppFTM','WFTA':'OppFTA'})
    lrsg = lrsg.rename(columns = {'WOR':'OppOR','WDR':'OppDR','WAst':'OppAst','WTo':'OppTO','WFTM':'OppFTM','WFTA':'OppFTA'})
    lrsg = lrsg.rename(columns = {'WTO':'OppTO','WStl':'OppStl','WBlk':'OppBlk','WPF':'OppFoul'})
    lrsg = lrsg.rename(columns = {'LFGM':'TmFGM','LFGA':'TmFGA','LFGM3':'TmFGM3','LFGA3':'TmFGA3','LFTM':'TmFTM','LFTA':'TmFTA'})
    lrsg = lrsg.rename(columns = {'LOR':'TmOR','LDR':'TmDR','LAst':'TmAst','LTo':'TmTO','LFTM':'TmFTM','LFTA':'TmFTA'})
    lrsg = lrsg.rename(columns = {'LTO':'TmTO','LStl':'TmStl','LBlk':'TmBlk','LPF':'TmFoul'})
    lrsg['TmWin'] = 0
    
    
    # Put in loser locations
    lrsg.loc[(lrsg['WLoc'] == 'H'),'TmLoc'] = 'A'
    lrsg.loc[(lrsg['WLoc'] == 'A'),'TmLoc'] = 'H'
    lrsg.loc[(lrsg['WLoc'] == 'N'),'TmLoc'] = 'N'
    del lrsg['WLoc']
    
    # Append lrsg to rsg, delete lrsg,
    rsg = rsg.append(lrsg)
    del lrsg
    
    # Bring in team names for both Tm and Opp
    rsg = pd.merge(rsg,teams[['TeamID','TeamName']],left_on='TmID',right_on='TeamID')
    del rsg['TeamID']
    rsg = rsg.rename(columns = {'TeamName':'TmName'})
    rsg = pd.merge(rsg,teams[['TeamID','TeamName']],left_on='OppID',right_on='TeamID')
    del rsg['TeamID']
    rsg = rsg.rename(columns = {'TeamName':'OppName'})
    
    # Add countable field for number of games
    rsg['TmGame'] = 1
    
    # Add field for number of minutes
    rsg['GameMins'] = 40 + rsg['NumOT']*5
    
    # Add field for Total Rebounds
    rsg['TmTR'] = rsg['TmOR'] + rsg['TmDR']
    rsg['OppTR'] = rsg['OppOR'] + rsg['OppDR']
    
    # Count number of FGA2/FGM2
    rsg['TmFGM2'] = rsg['TmFGM'] - rsg['TmFGM3']
    rsg['TmFGA2'] = rsg['TmFGA'] - rsg['TmFGA3']
    rsg['OppFGM2'] = rsg['OppFGM'] - rsg['OppFGM3']
    rsg['OppFGA2'] = rsg['OppFGA'] - rsg['OppFGA3']
    
    # Calculate game margin
    rsg['TmMargin'] = rsg['TmPF'] - rsg['OppPF']
    rsg['OppMargin'] = -rsg['TmMargin']
    
    # Add in per-40 stats to rsg
    for x in {'Opp','Tm'}:
        for column in metrics:
            rsg[x + column + 'per40'] = rsg[x + column] / rsg['GameMins'] * 40
    del column, x
    
    # Total Possessions (tbd)
    # rsg['TmPoss'] = rsg['TmFGA'] + rsg['TmFGA3'] TBD
    
    # Benchmark time
    rsgtime = time.time()-rsgbegin
    if rsgtime < 60:
        print('Create RSG Time: ' + str(round((rsgtime),2)) + ' sec')
    else:
        print('Create RSG Time: ' + str(round((rsgtime)/60,2)) + ' min')
    
    # return the rsg dataframe as the output
    return rsg

###############################################################################
###############################################################################
# Read in raw vegas data, clean up, manipulate for further analysis
###############################################################################
###############################################################################
def createvegas(rsg):
    # Set start time
    vegasbegin = time.time()
    
    # Read in raw vegas analysis data
    vegas = pd.read_csv(filepath_or_buffer = '/Users/Ryan/Google Drive/HistoricalNCAAData/VegasAnalysisFull.csv')
    
    # De-dupe vegas data (raw data has a row for both sides of the line)
    vegasfaves = vegas.loc[(vegas['TeamLineVegas'] < 0)]
    vegaspushes = vegas.loc[(vegas['TeamLineVegas'] == 0) & (vegas['Team'] < vegas['Opponent'])]
    vegas = vegasfaves.append(vegaspushes)
    del vegasfaves, vegaspushes
    
    # Only pull necessary columns
    vegas = vegas[['Date','Team','Opponent','TeamLineVegas']]
    
    # Rename columns
    vegas = vegas.rename(columns = {'Date':'GameDate','Team':'Tm1','Opponent':'Tm2','TeamLineVegas':'Tm1LineVegas'})
    
    # Change GameDate column to Datetime type
    vegas['GameDate'] = pd.to_datetime(vegas['GameDate'],format='%Y/%m/%d')
    
    # Get season by adding 24 weeks to game date and pulling year
    vegas['Season'] = vegas['GameDate'] + datetime.timedelta(weeks=24)
    vegas['Season'] = vegas['Season'].dt.year
    
    # Get game results into vegas
    vegas = pd.merge(vegas,rsg[['GameDate','TmName','OppName','TmWin','TmMargin']],left_on=['GameDate','Tm1','Tm2'],
                     right_on=['GameDate','TmName','OppName'])
    
    # Delete merged-in names
    del vegas['TmName'], vegas['OppName']
    
    # Rename columns
    vegas = vegas.rename(columns = {'TmMargin':'Tm1Margin','TmWin':'Tm1Win'})
    
    # Check margin vs vegas to create TmWinVegas
    vegas['Tm1WinVegas'] = ""
    vegas.loc[(vegas['Tm1LineVegas'] > -1 * vegas['Tm1Margin']),'Tm1WinVegas'] = 1
    vegas.loc[(vegas['Tm1LineVegas'] < -1 * vegas['Tm1Margin']),'Tm1WinVegas'] = -1
    vegas.loc[(vegas['Tm1LineVegas'] == -1 * vegas['Tm1Margin']),'Tm1WinVegas'] = 0
    vegas['Tm1WinVegas'] = pd.to_numeric(vegas['Tm1WinVegas'])
    
    # Benchmark time
    vegastime = time.time()-vegasbegin
    if vegastime < 60:
        print('Create Vegas Time: ' + str(round((vegastime),2)) + ' sec')
    else:
        print('Create Vegas Time: ' + str(round((vegastime)/60,2)) + ' min')
    
    # Output vegas DF
    return vegas
    
###############################################################################
###############################################################################
# Create vegasdates dataframe
###############################################################################
###############################################################################
def createvegasdates(vegas,rsg,size):
    # Set start time
    vegasdatesbegin = time.time()
    
    # Handle logic based on size input
    if size == 'full':
    
        # Pull out each unique game date/Season combo, count # of games
        vegasdates = vegas[['Season','GameDate','Tm1']].groupby(['Season','GameDate']).agg('count').reset_index()
        vegasdates = vegasdates.rename(columns = {'Tm1':'GameCount'})
       
        # Trim vegasdates to not include games where there is no game data to use in calculations
        vegasdates = vegasdates.loc[(vegasdates['GameDate'] <= max(rsg['GameDate']))]
      
    if size == 'small':
        # Create small vegasdates for testing
        vegasdates = vegasdates.loc[:7]
    

    
    # Benchmark time
    vegasdatestime = time.time()-vegasdatesbegin
    if vegasdatestime < 60:
        print('Create Vegasdates Time: ' + str(round((vegasdatestime),2)) + ' sec')
    else:
        print('Create Vegasdates Time: ' + str(round((vegasdatestime)/60,2)) + ' min')
    
    
    # return vegasdated dataframe
    return vegasdates


###############################################################################
###############################################################################
# Calculate cumulative stats for a season up to a date
###############################################################################
###############################################################################
def createvegasout(how):
    # Set start time
    vegasoutbegin = time.time()

    if how == 'read-in':
        # Read CSV
        vegasout = pd.read_csv(filepath_or_buffer = '/Users/Ryan/Desktop/DataFiles/PrelimData2018/vegasout.csv')
    
    if how == 'new':
        # Create rsg dataframe
        rsg = creatersg()
        # Create vegas dataframe
        vegas = createvegas(rsg)
        # Create vegasdates dataframe
        vegasdates = createvegasdates(vegas,rsg,size='small')
        global metrics
        
        # Create vegasout dataframe for the output of the for loop
        vegasout = pd.DataFrame()
    
        # Create summable fields
        summables = ['GameMins','TmWin','TmGame']
        for x in {'Opp','Tm'}:
            for column in metrics:
                summables.append(x + column)
        del column, x
        
        ###############################
        ## Begin For Loop
        ###############################
        # Loop through the data we have vegas info for
        for row in vegasdates.itertuples():
        
            # Set included season & the day of the game
            inclseason = row[1]
            dayofgame = row[2]
            print ('Season: ' + str(inclseason) + '; Date: ' + str(dayofgame))
        
            # Limit the game data to that season, & before the game date (i.e., all games before the current game)
            irsg = rsg.loc[(rsg['Season'] == inclseason) & (rsg['GameDate'] < dayofgame)]
            
            # Sum the summable fields
            iteams = irsg.groupby(['TmID','TmName'])[summables].sum().reset_index()
            
            for x in {'Opp','Tm'}:
                for column in metrics:
                    iteams[x + column + 'per40'] = iteams[x + column] / iteams['GameMins'] * 40
            del column, x
            
            # put Season & GameDate into iteams so we know what to merge back to the vegas data on
            iteams['Season'] = inclseason
            iteams['GameDate'] = dayofgame
            
            # Calculate Win Pct
            iteams['TmWin%'] = iteams['TmWin'] / iteams['TmGame']
            
            # Calculate Shooting Percentages
            iteams['TmFGPct'] = iteams['TmFGMper40'] / iteams['TmFGAper40']
            iteams['TmFG3Pct'] = iteams['TmFGM3per40'] / iteams['TmFGA3per40']
            iteams['TmFG2Pct'] = iteams['TmFGM2per40'] / iteams['TmFGA2per40']
            iteams['TmFTPct'] = iteams['TmFTMper40'] / iteams['TmFTAper40']
            
            # Calculate Ast per FG
            iteams['TmAstRate'] = iteams['TmAstper40'] / iteams['TmFGMper40']
            
            # Opponent-adjust a bunch of things
            for x in {'Opp','Tm'}:
                for column in metrics:
                    opponentadjust(x + column + 'per40')
            del column, x
            
            # Merge into vegasTms the Tm1 data from the vegas dataframe
            iteamsTm1 = iteams.add_prefix('Tm1_')
            vegasTms = pd.merge(vegas,iteamsTm1,left_on=['Season','GameDate','Tm1'],right_on=['Tm1_Season','Tm1_GameDate','Tm1_TmName'],how='left')
            # Merge
            iteamsTm2 = iteams.add_prefix('Tm2_')
            vegasTms = pd.merge(vegasTms,iteamsTm2,left_on=['Season','GameDate','Tm2'],right_on=['Tm2_Season','Tm2_GameDate','Tm2_TmName'],how='left')
        
            vegasTms = vegasTms.dropna()
            
            vegasout = vegasout.append(vegasTms)    
        ###############################
        ## End For Loop
        ###############################
        
        # clean up after for loop (not necessary since function)
        #del dayofgame, inclseason, row, vegasTms, iteamsTm1, iteamsTm2, summables
        
        # Check Output against input dates and game counts, produce vegasdatesmissing
        vegasdatesout = vegasout[['Season','GameDate','Tm1']].groupby(['Season','GameDate']).agg('count').reset_index()
        vegasdatesout = vegasdatesout.rename(columns = {'Tm1':'GameCountOut'})
        vegasdatesmissing = pd.merge(vegasdates,vegasdatesout,left_on=['Season','GameDate'],right_on=['Season','GameDate'],how='outer')
        vegasdatesmissing.fillna(0,inplace=True)
        vegasdatesmissing['Missing'] = vegasdatesmissing['GameCount'] - vegasdatesmissing['GameCountOut']
        x = vegasdatesmissing['Missing'].sum()
        y = vegasdatesmissing['GameCount'].sum()
        print('Missing ' + str(int(x)) + ' games; ' + str(round((1-(x/y))*100,1)) + '% used' )
        del vegasdatesout, vegasdatesmissing, x, y
        
        # Test DFs
        # testteamseason = vegasout.loc[(vegasout['Tm1'] == 'Florida')&(vegasout['Season'] == 2018)]
        #testrsg = rsg.loc[(rsg['TmName'] == 'Florida')&(rsg['Season'] == 2018)][['GameDate','TmName','OppName']]
        
        # Remove duplicate columns from vegasout dataframe that got merged in
        del vegasout['Tm1_Season'], vegasout['Tm1_GameDate'], vegasout['Tm1_TmName']
        del vegasout['Tm2_Season'], vegasout['Tm2_GameDate'], vegasout['Tm2_TmName']
    
        # Write CSV
        #vegasout.to_csv('/Users/Ryan/Desktop/DataFiles/PrelimData2018/vegasout.csv', index=False)
    
    # Benchmark time
    vegasouttime = time.time()-vegasoutbegin
    if vegasouttime < 60:
        print('Create Vegasout Time: ' + str(round((vegasouttime),2)) + ' sec')
    else:
        print('Create Vegasout Time: ' + str(round((vegasouttime)/60,2)) + ' min')
        
    return vegasout


###############################################################################
###############################################################################
# Prep for Machine Learning
###############################################################################
###############################################################################
def mlprep(mldf):

    # Set start time
    mlprepbegin = time.time()
    
    rawmldf = mldf.copy()
    
    # Create array of results
    results = np.array(mldf['Tm1Margin'])
    
    # Only include those that are needed
    mldf = mldf[['Tm1LineVegas']]

    metricstocomp = ['TmMarginper40','OA_TmMarginper40',
                     'TmPFper40','OA_TmPFper40','OppPFper40','OA_OppPFper40',
                     'TmFGA2per40','OA_TmFGA2per40','OppFGA2per40','OA_OppFGA2per40',
                     'TmFGM2per40','OA_TmFGM2per40','OppFGM2per40','OA_OppFGM2per40',
                     'TmFG2Pct',
                     'TmFGA3per40','OA_TmFGA3per40','OppFGA3per40','OA_OppFGA3per40',
                     'TmFGM3per40','OA_TmFGM3per40','OppFGM3per40','OA_OppFGM3per40',
                     'TmFG3Pct',
                     'TmFGAper40','OA_TmFGAper40','OppFGAper40','OA_OppFGAper40',
                     'TmFGMper40','OA_TmFGMper40','OppFGMper40','OA_OppFGMper40',
                     'TmFGPct',
                     'TmAstper40','OA_TmAstper40','OppAstper40','OA_OppAstper40',
                     'TmAstRate',
                     'TmORper40','OA_TmORper40','OppORper40','OA_OppORper40',
                     'TmDRper40','OA_TmDRper40','OppDRper40','OA_OppDRper40',
                     'TmTRper40','OA_TmTRper40','OppTRper40','OA_OppTRper40',
                     'TmFTAper40','OA_TmFTAper40','OppFTAper40','OA_OppFTAper40',
                     'TmFTMper40','OA_TmFTMper40','OppFTMper40','OA_OppFTMper40',
                     'TmFTPct',
                     'TmBlkper40','OA_TmBlkper40','OppBlkper40','OA_OppBlkper40',
                     'TmFoulper40','OA_TmFoulper40','OppFoulper40','OA_OppFoulper40',
                     ]

    
    for metric in metricstocomp:
        tempvegasout = rawmldf[['Tm1_'+metric,'Tm2_'+metric]]
        tempvegasout2 = pd.DataFrame()
        tempvegasout2['Tm1_' + metric + 'Comp'] = tempvegasout['Tm1_'+metric] - tempvegasout['Tm2_'+metric]
        mldf = pd.merge(mldf,tempvegasout2,how='left',left_index = True,right_index = True)
#        del mldf['Tm1_'+metric], mldf['Tm2_'+metric]

    
    del mldf['Tm1LineVegas']
    
    # Benchmark time
    mlpreptime = time.time()-mlprepbegin
    if mlpreptime < 60:
        print('Create MLDF Time: ' + str(round((mlpreptime),2)) + ' sec')
    else:
        print('Create MLDF Time: ' + str(round((mlpreptime)/60,2)) + ' min')
    
    return mldf, results
    
    
###############################################################################
###############################################################################
# Run Machine Learning
###############################################################################
###############################################################################
def mlmodel(mldf):

    global results
    
    # Set start time
    mlmodelbegin = time.time()
    
    # List the factors
    factors_list = list(mldf.columns)
    
    # Turn DF into np array for use in ML
    mldf = np.array(mldf)
    
    # Split the data into training and testing
    train_features, test_features, train_results, test_results = train_test_split(mldf, results, test_size = 0.25, random_state = 42)
    
    # Set up and train ML model
    rf = RandomForestRegressor(n_estimators = 1000)
    rf.fit(train_features, train_results)
    
    # use model test data to predict outcomes
    predictions = rf.predict(test_features)
    #errors = abs(predictions - test_results)
    #print('Mean Absolute Error:', round(np.mean(errors), 6))
    
    # List factors in irder of importance
    importances = list(rf.feature_importances_)
    feature_importances = pd.DataFrame()
    feature_importances['feature'] = factors_list
    feature_importances['importance'] = importances
#    feature_importances = [(feature, round(importance, 2)) for feature, importance in zip(factors_list, importances)]
#    feature_importances = sorted(feature_importances, key = lambda x: x[1], reverse = True)
    del importances
    
    # Get all output into the same DF, predicteddf
    predicteddf = pd.DataFrame(test_features,columns=factors_list)
    predicteddf['predictedmargin'] = predictions
    predicteddf['margin'] = test_results
    predicteddf.loc[(predicteddf['predictedmargin'] > 0),'predictedwin'] = 1
    predicteddf.loc[(predicteddf['predictedmargin'] < 0),'predictedwin'] = -1
    #predicteddf.loc[(predicteddf['prediction'] == 0),'predictedresult'] = 0
    predicteddf['win'] = predicteddf['margin'] / predicteddf['margin'].abs()
    
    predicteddf['correct'] = 0
    predicteddf.loc[(predicteddf['win'] == predicteddf['predictedwin']),'correct'] = 1
        
    #predicteddf['confidence'] = predicteddf['prediction'].abs()
    
    print("Correct Percent: " + str(round(predicteddf['correct'].mean(),4)*100) +'%')
    
#    confidence_results = pd.DataFrame(columns=['predictedmarginmin','result','records'])
#    confidences = [0,.05,.1,.15,.2,.25,.3,.35,.4,.45,.5,.55,.6,.65,.7,.75,.8,.85,.9,.95]
#    resultslist = []
#    records = []
#    for confidencemin in confidences:
#        resultslist.append(predicteddf['correct'].loc[predicteddf['confidence'] >= confidencemin].mean())
#        records.append(predicteddf['correct'].loc[predicteddf['confidence'] >= confidencemin].size)
#    confidence_results['confidencemin'] = confidences
#    confidence_results['result'] = resultslist
#    confidence_results['records'] = records
    
    
    # Write out ML results
    now = str(datetime.datetime.now().strftime('%Y-%m-%d %H-%M-%S'))
    predicteddf.to_csv('/Users/Ryan/Google Drive/HistoricalNCAAData/GoodNCAAGambling/MLDataOutput/predicteddf ' + now + '.csv', index=False)
    feature_importances.to_csv('/Users/Ryan/Google Drive/HistoricalNCAAData/GoodNCAAGambling/MLDataOutput/feature_importances ' + now + '.csv', index=False)
    #confidence_results.to_csv('/Users/Ryan/Google Drive/HistoricalNCAAData/GoodNCAAGambling/MLDataOutput/confidence_results ' + now + '.csv', index=False)
    # Benchmark time
    mlmodeltime = time.time()-mlmodelbegin
    if mlmodeltime < 60:
        print('Create MLDF Time: ' + str(round((mlmodeltime),2)) + ' sec')
    else:
        print('Create MLDF Time: ' + str(round((mlmodeltime)/60,2)) + ' min')
    

    return feature_importances, confidence_results

###############################################################################
###############################################################################
# Master Handler
###############################################################################
###############################################################################

# Set up vegasout dataframe
vegasout = createvegasout(how='read-in')

# Set up mldf for use in machine learning
mldf, results = mlprep(vegasout)

# Run machine learning
feature_importances, confidence_results = mlmodel(mldf)

# Benchmark time
totaltime = time.time()-begin
if totaltime < 60:
    print('Total Process Time: ' + str(round((totaltime),2)) + ' sec')
else:
    print('Total Process Time: ' + str(round((totaltime)/60,2)) + ' min')

del begin, totaltime
