
import streamlit as st

import numpy as np
import pandas as pd
import altair as alt
from db import get_db

from st_functions import get_teams_list, is_ascending_rank, get_seasons_list, opponentAdjust


def createSeasonSummaryStripPlot(data, team):
    # TODO make chart interactive where I can click on a dot and see that team

    # flip defense score, rename columns, melt for visualizing
    data['OA_OppPFper40'] = -data['OA_OppPFper40']
    data = data.rename(
        columns={
            'OA_OppPFper40': 'Defense Score',
            'OA_TmPFper40': 'Offense Score',
            'OA_TmMarginper40': 'Overall Score',
            'TmName': 'Team'
        }
    )
    data = pd.melt(
        data,
        id_vars=['Team'],
        value_vars=['Offense Score', 'Defense Score', 'Overall Score'],
        var_name='Metric',
        value_name='Value'
    )

    # Create chart
    base = alt.Chart(
        data[['Metric', 'Value', 'Team']],
        title=f"{team}'s defense, offense, and overall score"
    )

    # TODO figure out how to sort category axis
    teamMark = base.mark_tick(
        thickness=2,
        size=25
    ).encode(
        x=alt.X(
            'Value', 
            axis=alt.Axis(title='Value (better -------->)')
        ),
        y=alt.Y(
            'Metric', 
            axis=alt.Axis(title=None)
        ),
        color=alt.value('orange')
    ).transform_filter(
        alt.datum.Team == team
    )

    teamValue = base.mark_text(
        dy=18,
        align='center'
    ).encode(
        x='Value',
        y='Metric',
        text='Value',
        color=alt.value('black')
    ).transform_filter(
        alt.datum.Team == team
    )

    nonTeamMarks = base.mark_circle().encode(
        x='Value',
        y='Metric'
    ).transform_filter(
        alt.datum.Team != team
    )

    return teamMark + nonTeamMarks + teamValue


def tidy_results_df(df):
    '''
    1) Rounds all non-rank "perX" fields to 2 or 4 decimals
    2) Change season from numeric to text
    3) Makes ranks integers
    4) Makes GameOT an integer
    '''

    for col in df.columns:
        # 3) Make ranks integers, also include a padded rank column (always 3 digits)
        if col[0:3] == 'Rnk':
            df[col] = df[col].astype(int)
            df[col+'_Padded'] = df[col].apply('{:0>3}'.format)
        # 1) Round all non-rank 'per40' fields to 2 decimals
        elif col[-5:] == 'per40' or col[-7:] == 'perGame':
            df = df.round({col:2})
        elif col[-7:] == 'perPoss':
            df = df.round({col:4})

    # 2) Change seasons from numeric to text
    try:
        df['Season'] = df['Season'].apply(str)
    except KeyError:
        pass

    # 4) Make GameOT an integer
    try:
        df['GameOT'] = df['GameOT'].apply(int)
    except KeyError:
        pass
    
    
    return df


def color_negative_red(val):
    """
    Takes a scalar and returns a string with
    the css property `'color: red'` for negative
    strings, black otherwise.
    """
    try:
        color = 'red' if val < 0 else 'black'
    except TypeError:
        color = 'black'
    return 'color: %s' % color


if __name__ == "__main__":
    # Request inputs from user
    DB = get_db(config_file_name='docker_database.ini')
    TEAMS = get_teams_list(_db=DB)
    TEAM = st.sidebar.selectbox("Select a team", TEAMS)
    SEASONS = get_seasons_list(_db=DB)
    SEASON = st.sidebar.slider(
        "Select a season",
        min_value=min(SEASONS),
        max_value=max(SEASONS),
        value=max(SEASONS)
    )

    # Write title
    st.title(
        f"{TEAM}'s {SEASON} Season"
    )

    # Get team's game results from the database
    season_games_cursor = DB.games.find(
        {'TmName': TEAM,
        'Season': SEASON},
        {'_id': 0}
    )
    season_games = pd.DataFrame(list(season_games_cursor))
    season_games.sort_values(by=['GameDate'], inplace=True)
    season_games = tidy_results_df(season_games)
    #st.write(season_games)
    
    # Get aggregated season results from DB
    season_team_cursor = DB.seasonteams.find(
        {'Season': SEASON},
        {
            '_id': 0
        }
    )
    season_team = pd.DataFrame(list(season_team_cursor))

    # Opponent-adjust fields from pre-aggregated data
    season_team = opponentAdjust(
        season_team,
        PREFIXES=['Tm', 'Opp'],
        includeOARankFields=True,
        includeNormFields=True
    )
    season_team = tidy_results_df(season_team)
    
    # Write Ranks
    st.subheader(f"Team Rank: {season_team.loc[season_team['TmName'] == TEAM, 'Rnk_OA_TmMarginper40'].values[0]}")
    st.subheader(f"Offense Rank: {season_team.loc[season_team['TmName'] == TEAM, 'Rnk_OA_TmPFper40'].values[0]}")
    st.subheader(f"Defense Rank: {season_team.loc[season_team['TmName'] == TEAM, 'Rnk_OA_OppPFper40'].values[0]}")


    # Display visuals
    st.write(createSeasonSummaryStripPlot(season_team, TEAM))

    # Merge in opponent's rank to season games, rename column
    season_games = pd.merge(
        left=season_games,
        right=season_team[['TmName', 'Season', 'Rnk_OA_TmMarginper40_Padded', 'TmMarginper40']],
        left_on=['OppName', 'Season'],
        right_on=['TmName', 'Season'],
        how='left'
    )
    season_games = season_games.rename(
        columns={
            #'Rnk_OA_TmMarginper40': 'OppRank'
            'TmMarginper40': 'Opp_TmMarginper40'
        }
    )

    # Create and rename columns for dataframe
    season_games['Opponent'] = season_games.agg(lambda x: f"({x['Rnk_OA_TmMarginper40_Padded']}) {x['OppName']}", axis=1)
    season_games['Result'] = np.where(season_games['TmWin'] == 1, 'W', 'L')
    season_games['GameOT_String'] = season_games.agg(lambda x: f" {x['GameOT']}OT", axis=1)
    season_games.loc[season_games['GameOT_String'] == ' 0OT', 'GameOT_String'] = ''
    season_games['Result'] = season_games.agg(lambda x: f"{x['Result']} by {abs(x['TmMargin'])} ({x['TmPF']}-{x['OppPF']}){x['GameOT_String']}", axis=1)
    season_games['OA Margin'] = (season_games['TmMargin'] / season_games['TmMins'] * 40 + season_games['Opp_TmMarginper40']).astype(float)
    # Limit columns in results dataframe
    season_games = season_games[
        [
            'GameDate',
            'Opponent',
            'Result',
            'OA Margin'
        ]
    ]
    # Display team's results
    st.dataframe(
        season_games.style.applymap(
            color_negative_red
        ).format(
            {
                'OA Margin': "{:7,.2f}"
            }
        )
    )
    """
    # Determine percent of points fields
    season_team['TmPF_FT'] = season_team['TmFTM']
    season_team['TmPF_FG2'] = 2*season_team['TmFG2M']
    season_team['TmPF_FG3'] = 3*season_team['TmFG3M']

    season_sum = season_team[['TmPF_FT', 'TmPF_FG2', 'TmPF_FG3']].sum(
        axis=0
    ).reset_index(
    ).rename(
        columns={
            'index': 'Method',
            0: 'Points'
        }
    )
    season_sum['TmName'] = 'Average'

    pop_df = pd.melt(
        season_team.loc[season_team['TmName'] == TEAM],
        id_vars=['TmName'],
        value_vars=['TmPF_FT','TmPF_FG2','TmPF_FG3']
    ).rename(
        columns={
            'variable': 'Method',
            'value': 'Points'
        }
    ).append(season_sum, sort=True)
    st.write(pop_df)


    st.write(season_team)

    pop_chart = alt.Chart(pop_df).mark_bar().encode(
        x=alt.X('Points', stack='normalize'),
        y='TmName',
        color='Method'
    )
    st.write(pop_chart)

    # Method 2
    # Determine percent of points fields
    season_team['TmPoP_FT'] = season_team['TmFTM'] / season_team['TmPF']
    season_team['TmPoP_FG2'] = 2*season_team['TmFG2M'] / season_team['TmPF']
    season_team['TmPoP_FG3'] = 3*season_team['TmFG3M'] / season_team['TmPF']
    season_team['OppPoP_FT'] = season_team['OppFTM'] / season_team['OppPF']
    season_team['OppPoP_FG2'] = 2*season_team['OppFG2M'] / season_team['OppPF']
    season_team['OppPoP_FG3'] = 3*season_team['OppFG3M'] / season_team['OppPF']

    pop_df = pd.melt(
        season_team.loc[season_team['TmName'] == TEAM],
        id_vars=['TmName'],
        value_vars=[
            'TmPoP_FT',
            'TmPoP_FG2',
            'TmPoP_FG3',
            'OppPoP_FT',
            'OppPoP_FG2',
            'OppPoP_FG3'
        ]
    )

    pop_df.loc[pop_df['variable'].str[0:6] == 'OppPoP', 'TmName'] = f"{TEAM}'s Opponents"
    pop_df['variable'] = pop_df['variable'].str.replace(r"(Tm|Opp)", '')
    
    
    '''
    pop_df = pop_df.pivot(
        index='TmName',
        columns='variable',
        values='value'
    ).reset_index()
    '''
    
    
    season_sum = season_team[['Season','TmPF_FT', 'TmPF_FG2', 'TmPF_FG3', 'TmPF']].groupby('Season').sum().reset_index()
    season_sum['PoP_FT'] = season_sum['TmPF_FT'] / season_sum['TmPF']
    season_sum['PoP_FG2'] = season_sum['TmPF_FG2'] / season_sum['TmPF']
    season_sum['PoP_FG3'] = season_sum['TmPF_FG3'] / season_sum['TmPF']
    for field in ['TmPF_FT', 'TmPF_FG2', 'TmPF_FG3', 'TmPF']:
        del season_sum[field] 

    season_sum['Season'] = 'Season Average'
    season_sum = season_sum.rename(
        columns={
            'Season': 'TmName'
        }
    )
    season_sum = pd.melt(
        season_sum,
        id_vars=['TmName'],
        value_vars=[
            'PoP_FT',
            'PoP_FG2',
            'PoP_FG3'
        ]
    )
    #pop_df = pop_df.append(season_sum, sort=True).reset_index()
    st.write(pop_df,'Above is pop_df here')


    pop_chart_2 = alt.Chart(pop_df).mark_bar().encode(
        x=alt.X('value'),
        y='TmName',
        color='TmName',
        row='variable'
    )

    st.write(pop_chart_2)


    """
