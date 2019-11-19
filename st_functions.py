
import streamlit as st
from db import get_db

def return_teams_list(db = get_db()):
    '''
    Returns a list of the teams in seasonteams
    '''
    pipeline = [{'$group':{'_id':{'Team':'$TmName'}}},
                {'$sort':{'_id':1}}
                ]
    results = db.seasonteams.aggregate(pipeline)
    teams_list = []
    for x in results:
        teams_list.append(x['_id']['Team'])
    return teams_list

def return_seasons_list(db = get_db()):
    '''
    Returns a list of the seasons in seasonteams
    '''
    pipeline = [{'$group':{'_id':{'Season':'$Season'}}},
                {'$sort':{'_id':-1}}
                ]
    results = db.seasonteams.aggregate(pipeline)
    seasons_list = []
    for x in results:
        seasons_list.append(x['_id']['Season'])
    return seasons_list


def selectbox_seasons(db = get_db()):
    '''
    Prompt user to select a season, setting variable 'season'
    '''
    seasons_list = return_seasons_list(db)
    season = st.selectbox('Select a season',seasons_list)
    return season

def slider_seasons(db = get_db()):
    '''
    Prompt user to select a season, setting variable 'season'
    '''
    pipeline = [{'$group':{'_id':{'Season':'$Season'}}},
                {'$sort':{'_id':-1}}
                ]
    results = db.seasonteams.aggregate(pipeline)
    seasons_list = []
    for x in results:
        seasons_list.append(x['_id']['Season'])
    season = st.slider('Select a season',min_value = min(seasons_list),max_value = max(seasons_list))
    return season


def selectbox_team(db = get_db()):
    '''
    Prompt user to select a single team
    '''
    teams_list= return_teams_list(db)
    team = st.selectbox('Select a team',teams_list)
    return team

def sidebar_multiselect_team(db = get_db()):
    '''
    Add a multiselect team to the sidebar
    '''
    teams_list = return_teams_list()
    teams = st.sidebar.multiselect('Select a team or teams',tuple(teams_list))
    return teams


#selectbox_seasons()
#selectbox_teams()

