
import streamlit as st
import pandas as pd
from db import get_db
import st_functions as stfn
import plotly.graph_objects as go
import plotly.express as px
import plotly.figure_factory as ff
import altair as alt

db = get_db()

season = stfn.slider_seasons(db)

query = {'Season':season}
fields = {'_id':0,
        'TmName':1,
        'OA_TmMarginper40':1,
        'OA_TmPFper40':1,
        'OA_OppPFper40':1,
        'OA_TmORBperPoss':1,
        'TmTSP':1
        }

results = db.seasonteams.find(query)
df = pd.DataFrame(list(results))
df = df.sort_values(by=['OA_TmMarginper40'])
'''
fig = go.Figure()
fig.add_trace(go.Scatter(
    x=df['OA_TmMarginper40'],
    y=df['TmName'],
    marker=dict(color="crimson", size=4),
    mode="markers",
    name="Test",
))
st.plotly_chart(fig)

fig2 = ff.create_distplot([df['OA_TmMarginper40'],df['OA_TmPFper40'],df['OA_OppPFper40']],group_labels = ['Opponent-Adjusted Margin','Offense','Defense'])
st.plotly_chart(fig2)
'''

metrics = ['PF', 'Margin',
           'FGM', 'FGA',
           'FG3M', 'FG3A',
           'FG2M', 'FG2A',
           'FTA', 'FTM',
           'Ast', 'ORB',
           'DRB', 'TRB',
           'TO', 'Stl',
           'Blk', 'Foul']

metrics = ['ORB',
           'DRB', 'TRB']
scattermatrix_fields = []
for x in {'Opp', 'Tm'}:
    for column in metrics:
        scattermatrix_fields.append(x + column + 'perPoss')
del column, x

brush = alt.selection_single()
'''
c = alt.Chart(df).mark_circle().encode(
    alt.X(alt.repeat("column"), type='quantitative'),
    alt.Y(alt.repeat("row"), type='quantitative'),
    tooltip=['TmName:N', 'TmWin:N'],
    color=alt.condition(selector,'TmWin',alt.value('lightgray'))
).properties(
    width=150,
    height=150
).repeat(
    row=scattermatrix_fields[0],
    column=scattermatrix_fields[1]
).add_selection(
    brush
).interactive()
'''

c = alt.Chart(df).mark_point().encode(
    x='TmORBperPoss:Q',
    y='TmDRBperPoss:Q',
    color=alt.condition(brush,'TmWin',alt.value('lightgray'))
).add_selection(
    brush
)

st.altair_chart(c)