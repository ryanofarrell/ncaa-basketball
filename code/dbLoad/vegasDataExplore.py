
#%%
import pandas as pd
import numpy as np
from datetime import timedelta


# Import to add project folder to sys path
import sys
utils_path = '/Users/Ryan/Documents/projects/ncaa-basketball/code/utils'
if utils_path not in sys.path:
    sys.path.append(utils_path)

from db import get_db
from api import getSeasonsList


#%% Read in data
firstRun=True
for suffix in range(3,20):
    suffStr = str(suffix)
    suffStr = '0'*(2-len(suffStr)) + suffStr
    print(f"Working on {suffStr}")
    csvPath = f'/Users/Ryan/Documents/projects/ncaa-basketball/data/vegasLines/ncaabb{suffStr}.csv'
    
    if firstRun:
        df = pd.read_csv(csvPath)
        firstRun = False
    else:
        df = df.append(pd.read_csv(csvPath))

df = df[[
    'date',
    'home',
    'road',
    'neutral',
    'line'
]]
df['line'] = pd.to_numeric(df['line'], errors='coerce')


# %% Try to add team IDs
# Lowercase team names
df['home'] = df['home'].str.lower()
df['home'] = df['home'].str.strip()
df['road'] = df['road'].str.lower()
df['road'] = df['road'].str.strip()
teamSpellings = pd.read_csv(f'/Users/Ryan/Documents/projects/ncaa-basketball/data/TeamSpellings.csv', encoding="ISO-8859-1")
df = pd.merge(
    left=df,
    right=teamSpellings.rename(columns={'TeamNameSpelling': 'home', 'TeamID': 'homeID'}),
    on=['home'],
    how='left'
)
df = pd.merge(
    left=df,
    right=teamSpellings.rename(columns={'TeamNameSpelling': 'road', 'TeamID': 'roadID'}),
    on=['road'],
    how='left'
)
dfMissingHome = df.loc[pd.isnull(df['homeID'])]
dfMissingRoad = df.loc[pd.isnull(df['roadID'])]

print(f"Dropping {len(df.loc[pd.isnull(df['homeID']) | pd.isnull(df['roadID'])])} records due to no matching team name")

df = df.loc[~pd.isnull(df['homeID']) & ~pd.isnull(df['roadID'])]

print(f"Dropping {len(df.loc[pd.isnull(df['line'])])} records due to no line provided")
df = df.loc[~pd.isnull(df['line'])]


# %% Duplicate records to insert into database

for col in ['home', 'road', 'neutral']:
    del df[col]

df = df.rename(columns={
    'date': 'GameDate',
    'line': 'GameVegasLine',
    'homeID': 'TmID',
    'roadID': 'OppID'
})

df['GameDate'] = pd.to_datetime(df['GameDate'])#.dt.strftime('%Y-%m-%d')

df1 = df.copy()
df1 = df1.rename(columns={
    'TmID': 'OppID',
    'OppID': 'TmID'
})
df = df.append(df1)


# %% Example data for inserts
db = get_db()

games = pd.DataFrame(list(db.games.find(
    filter={'Season':{'$gte': 2004}},
    projection=['TmID', 'OppID', 'GameDate']
)))

dfWithLines = pd.merge(
    df,
    games,
    on=['GameDate', 'TmID', 'OppID'],
    how='left'
)

print(f"Found {len(df.loc[~pd.isnull(df['_id'])])} matches in database of {len(df)} possible lines")

dfNoMatch = df.loc[pd.isnull(df['_id'])]
dfNoMatch['GameDate'] += timedelta(days=1)
del dfNoMatch['_id']
dfNewMatch = pd.merge(
    dfNoMatch,
    games,
    on=['GameDate', 'TmID', 'OppID'],
    how='left'
)
print(f"Found add'l {len(dfNewMatch.loc[~pd.isnull(dfNewMatch['_id'])])} matches in database of {len(dfNewMatch)} possible lines when adding one day to original game date")
df = df.append(dfNewMatch.loc[~pd.isnull(dfNewMatch['_id'])])

dfNoMatch = dfNewMatch.loc[pd.isnull(dfNewMatch['_id'])]
dfNoMatch['GameDate'] -= timedelta(days=2)
del dfNoMatch['_id']
dfNewMatch = pd.merge(
    dfNoMatch,
    games,
    on=['GameDate', 'TmID', 'OppID'],
    how='left'
)
print(f"Found add'l {len(dfNewMatch.loc[~pd.isnull(dfNewMatch['_id'])])} matches in database of {len(dfNewMatch)} possible lines when subtracting one day from original game date")
df = df.append(dfNewMatch.loc[~pd.isnull(dfNewMatch['_id'])])

# %%
