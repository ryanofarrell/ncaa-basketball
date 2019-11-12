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


selected_teams = stfn.sidebar_multiselect_team(db)

query = {'OppAst':{'$ne':np.nan},
        'TmName':{'$in':selected_teams}
        }
fields = {'_id':0,
        'TmMarginper40':1,
        'OppPFper40':1,
        'TmPFper40':1,
        'TmGame':1,
        'TmWin':1,
        'Season':1,
        'TmName':1}

df = pd.DataFrame(list(db.seasonteams.find(query,fields)))
st.write(df)

# Create a selection that chooses the nearest point & selects based on x-value
nearest = alt.selection(type='single', nearest=True, on='mouseover',
                        fields=['Season'], empty='none')

line = alt.Chart(df).mark_line().encode(
    x='Season',
    y='TmMarginper40',
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
    text=alt.condition(nearest, 'TmMarginper40', alt.value(' '))
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
    height=600
)

st.altair_chart(chart)

