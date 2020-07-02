import streamlit as st
from db import get_db
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.figure_factory as ff
import plotly.express as px
import st_functions as stfn
import altair as alt
db = get_db()


@st.cache
def get_results_df(query,fields):
    df = pd.DataFrame(list(db.seasonteams.find(query,fields)))
    return df


def tidy_results_df(df):
    '''
    1) Rounds all "perX" fields to 2 decimals
    2) Change season from numeric to text
    '''

    # 1) Round all 'per40' fields to 2 decimals
    for col in df.columns:
        if col[-5:] == 'per40' or col[-7:] == 'perGame':
            df = df.round({col:2})
        if col[-7:] == 'perPoss':
            df = df.round({col:4})

    # 2) Change seasons from numeric to text
    try:
        df['Season'] = df['Season'].apply(str)
    except KeyError:
        pass
    return df

selected_teams = stfn.sidebar_multiselect_team(db)
if len(selected_teams) == 0:
    st.error('Please select a team')

metrics_list = ['PF', 'Margin',
                'FGM', 'FGA',
                'FG3M', 'FG3A',
                'FG2M', 'FG2A',
                'FTA', 'FTM',
                'Ast', 'ORB',
                'DRB', 'TRB',
                'TO', 'Stl',
                'Blk', 'Foul']

prefix = st.sidebar.selectbox('Tm or Opp',['Tm','Opp'])

metric = st.sidebar.selectbox('Select a metric',metrics_list)

metric_appendix = st.sidebar.selectbox('Select a normalization',['per40','perPoss'])

is_opponent_adjusted = st.sidebar.checkbox('Opponent-adjusted?')
adjustment = 'OA_' if is_opponent_adjusted else ''

fields = {'_id':0,
        'TmGame':1,
        'TmWin':1,
        'Season':1,
        'TmName':1}

display_attr = adjustment + prefix + metric + metric_appendix
fields[display_attr] = 1

#st.write(fields)

query = {'TmName':{'$in':selected_teams}}



df = get_results_df(query,fields)
df = tidy_results_df(df)
df['TmWinPct'] = df['TmWin'] / df['TmGame']
#st.write(df)

# Create a selection that chooses the nearest point & selects based on x-value
nearest = alt.selection(type='single', nearest=True, on='mouseover',
                        fields=['Season'], empty='none')

line = alt.Chart(df).mark_line().encode(
    x='Season',
    y=display_attr,
    color='TmName'
)

# Transparent selectors across the chart. This is what tells us
# the x-value of the cursor
selectors = alt.Chart(df).mark_point().encode(
    x='Season',
    opacity=alt.value(0),
).add_selection(
    nearest
)

# Draw points on the line, and highlight based on selection
points = line.mark_point().encode(
    opacity=alt.condition(nearest, alt.value(1), alt.value(0))
)

# Draw text labels near the points, and highlight based on selection
text = line.mark_text(align='left', dx=5, dy=-5).encode(
    text=alt.condition(nearest,display_attr, alt.value(' '))
)

# Draw a rule at the location of the selection
rules = alt.Chart(df).mark_rule(color='gray').encode(
    x='Season',
).transform_filter(
    nearest
)

# Put the five layers into a chart and bind the data
chart = alt.layer(
    line, selectors, points, rules, text
).properties(
    height=800
)

# Try/except for scenarios when there is no team selected
try:
    st.altair_chart(chart,width=0)
except ValueError:
    pass


# Dist plot the teams' results
'''
chart2 = alt.Chart(df[['TmName','TmMarginper40']]).mark_area(
    opacity=0.4,
    interpolate='step'
).encode(
    alt.X('TmMarginper40:Q', bin=alt.Bin(step=3)),
    alt.Y('count()', stack=None),
    alt.Color('TmName')
)


#st.altair_chart(chart2)

chart3 = alt.Chart(df).mark_line(opacity=0.7).encode(
    x='Season',
    y=alt.Y('TmWinPct'),
    color='TmName',
).add_selection(nearest)
st.altair_chart(chart3)

'''