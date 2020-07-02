"""Streamlit application to display a college basketball team's history
from 1985-2020 seasons.
"""
import streamlit as st
import pandas as pd
import altair as alt
from db import get_db
from st_functions import get_teams_list, is_ascending_rank

# TODO cache data
# TODO round fields

if __name__ == "__main__":
    st.title("Explore a team's history in a variety of metrics!")
    DB = get_db(config_file_name='docker_database.ini')

    # SEASONS = get_seasons_list(_db=DB)
    TEAMS = get_teams_list(_db=DB)
    TEAM = st.sidebar.selectbox("Select a team", TEAMS)
    PREFIX = st.sidebar.selectbox("Select a prefix", ['Tm', 'Opp'])
    PREFIX_LONGSTRING = 'Team' if PREFIX == 'Tm' else 'Opponent'
    OTHER_PREFIX = 'Opp' if PREFIX == 'Tm' else 'Tm'
    METRICS = ['PF', 'Margin',
               'FGM', 'FGA',
               'FG3M', 'FG3A',
               'FG2M', 'FG2A',
               'FTA', 'FTM',
               'Ast', 'ORB',
               'DRB', 'TRB',
               'TO', 'Stl',
               'Blk', 'Foul']
    METRIC = st.sidebar.selectbox("Select a metric", METRICS)
    DENOM = st.sidebar.selectbox(
        'Select a normalization', ['per40', 'perGame', 'perPoss']
    )
    DENOM_FIELD = 'Mins' if DENOM == 'per40' else DENOM[-4:]
    if DENOM == 'perPoss':
        DENOM_LONGSTRING = 'per possession'
    elif DENOM == 'per40':
        DENOM_LONGSTRING = 'per 40 mins'
    elif DENOM == 'perGame':
        DENOM_LONGSTRING = 'per Game'

    OA_BOOL = st.sidebar.checkbox("Opponent Adjust?")
    OA_PREF = 'OA_' if OA_BOOL else ''
    OA_LONGSTRING = ' (opponent-adjusted)' if OA_BOOL else ''

    NORMALIZE_CONST = 40 if DENOM == 'per40' else 1

    # Get results from DB
    season_team_cursor = DB.seasonteams.find(
        {},
        {
            # Get OA metric fields
            PREFIX+METRIC: 1,
            PREFIX+DENOM_FIELD: 1,
            'OppSum_'+OTHER_PREFIX+METRIC: 1,
            'OppSum_'+OTHER_PREFIX+DENOM_FIELD: 1,
            # Get W/L record fields
            'TmWin': 1,
            'TmGame': 1,
            # Get required aggregate fields
            'TmName': 1,
            'Season': 1,
            '_id': 0
        }
    )
    season_team = pd.DataFrame(list(season_team_cursor))
    season_team['Season'] = season_team.Season.astype(str)

    # Opponent-adjust selected metric
    season_team[PREFIX+METRIC+DENOM] = season_team[PREFIX+METRIC] / season_team[PREFIX+DENOM_FIELD] * NORMALIZE_CONST
    season_team['OA_'+PREFIX+METRIC+DENOM] = \
        (season_team[PREFIX+METRIC+DENOM]) - \
        (
            (season_team['OppSum_'+OTHER_PREFIX+METRIC] - season_team[PREFIX+METRIC]) /
            (season_team['OppSum_'+OTHER_PREFIX+DENOM_FIELD] - season_team[PREFIX+DENOM_FIELD])
        ) * NORMALIZE_CONST

    # Determine team's regular-season record
    # TODO update with postseason games
    season_team['TmLoss'] = season_team['TmGame'] - season_team['TmWin']
    season_team['Record'] = season_team['TmWin'].map(str) + '-' + season_team['TmLoss'].map(str)

    # Rank each team's values within the season
    season_team['Rnk_'+OA_PREF+PREFIX+METRIC+DENOM] = season_team.groupby(
        'Season'
    )[OA_PREF+PREFIX+METRIC+DENOM].rank(
        'min', ascending=is_ascending_rank(PREFIX, METRIC)
    )

    # Create chart
    season_team_chart = season_team.loc[season_team.TmName == TEAM]
    TITLE_STRING = f"{TEAM}: Rank of {PREFIX_LONGSTRING}'s {METRIC} {DENOM_LONGSTRING+OA_LONGSTRING}  [1 is best]"
    chart = alt.Chart(
        data=season_team_chart, 
        title=TITLE_STRING
    ).mark_line(
        point=True
    ).encode(
        alt.X('Season'),
        alt.Y('Rnk_'+OA_PREF+PREFIX+METRIC+DENOM,
              scale=alt.Scale(domain=(353,1)),
              axis=alt.Axis(title='Rank')),
        tooltip=['Season',
                 'Record', 
                 OA_PREF+PREFIX+METRIC+DENOM, 
                 'Rnk_'+OA_PREF+PREFIX+METRIC+DENOM]
    ).interactive()

    st.altair_chart(chart)
