#%%
from sklearn.cluster import SpectralClustering
import pandas as pd
import networkx as nx

#%%
df = pd.read_csv('../../data/Stage2DataFiles/RegularSeasonCompactResults.csv')
teams = pd.read_csv('../../data/Stage2DataFiles/Teams.csv')
df = pd.merge(
    df, 
    teams[['TeamID', 'TeamName']], 
    left_on='WTeamID', 
    right_on='TeamID'
)
del df['TeamID']
df = df.rename(columns={'TeamName': 'TmName'})
df = pd.merge(
    df, 
    teams[['TeamID', 'TeamName']], 
    left_on='LTeamID', 
    right_on='TeamID'
)
del df['TeamID']
df = df.rename(columns={'TeamName': 'OppName'})
df = df.loc[df['Season'] == 2018]

# %%
g = nx.Graph()
edges = [tuple(x) for x in df[['TmName', 'OppName']].to_numpy()]
g.add_edges_from(edges)
A = nx.to_numpy_matrix(g)
teamList = g.nodes()


# %%
clustering = SpectralClustering(
    affinity='precomputed'
)
labels = clustering.fit_predict(A)
results = pd.DataFrame({'TeamName': teamList, 'cluster': labels})

# %%
