
import pandas as pd


# TODO switch back the allow_output_mutation=True once bug 
#@st.cache
def cache_game_data(q, f, _db):
    returned_games = pd.DataFrame(list(_db.games.find(q, f)))
    return returned_games


def get_teams_list(_db):
    """Get a list of all the teams with at least one game ever

    Keyword Arguments:
        _db {database connection} -- Connection to MongoDB

    Returns:
        List -- Every team with at least 1 game played ever
    """
    pipeline = [{'$group': {'_id': {'Team': '$TmName'}}},
                {'$sort': {'_id': 1}}
                ]
    results = _db.games.aggregate(pipeline)
    teams_list = []
    for itm in results:
        teams_list.append(itm['_id']['Team'])
    return teams_list


def get_seasons_list(_db):
    '''
    Returns a list of the seasons in seasonteams
    '''
    pipeline = [{'$group': {'_id': {'Season': '$Season'}}},
                {'$sort': {'_id': -1}}
                ]
    results = _db.games.aggregate(pipeline)
    seasons_list = []
    for itm in results:
        seasons_list.append(itm['_id']['Season'])
    return seasons_list


def returnGamesForSeasonTeam(_db, teams=['Florida'], season=2020):
    """Gets all the games for provided teams for a season

    Keyword Arguments:
        teams {list} -- List of team names to return games
                        (default: {['Florida']})
        season {int} -- Season in question (default: {2020})
        db {database connection} -- Connection to MongoDB (default: {get_db()})

    Returns:
        DataFrame -- The games for the team in the given season,
            sorted oldest --> newest
    """
    query = {'TmName': {'$in': teams},
             'Season': season}
    games_df = pd.DataFrame(list(db.games.find(query)))
    games_df.sort_values(by=['GameDate'], inplace=True)
    return games_df


def opponentAdjustMetric(_db, team='Florida', prefix='Tm', metric='PF',
                         suffix='perPoss',
                         season=2020):

    # Figure out the prefix
    assert prefix in ['Opp', 'Tm'], 'Must be Opp or Tm as the prefix'
    otherPrefix = 'Opp' if prefix == 'Tm' else 'Tm'

    # Figure out the field to sum for denominator of team and opps
    assert suffix in ['per40', 'perPoss', 'perGame'], 'Bad suffix'
    denomField = 'Mins' if suffix == 'per40' else suffix[-4:]
    normalizeConst = 40 if suffix == 'per40' else 1

    # Get all of team's games into a list
    tmGames = returnGamesForSeasonTeam([team],
                                       season,
                                       _db)[['OppName',
                                            prefix+metric,
                                            prefix+denomField]]
    # st.write(tmGames)

    # Aggregate the team's metric and denominator fields
    tmAggMetric = tmGames[[prefix+metric, prefix+denomField]].sum()
    # st.write(tmAggMetric)

    st.write(f"{team}'s total {prefix+metric} in \
        {tmAggMetric[prefix+denomField]} \
        {denomField}: {tmAggMetric[prefix+metric]}")

    tmAvgMetric = tmAggMetric[prefix+metric] / tmAggMetric[prefix+denomField] * normalizeConst
    st.write(f"Thats {tmAvgMetric} {suffix}")

    # Get all the opponents names into a list
    oppNames = list(tmGames['OppName'])

    # Get all the games the opponents played
    oppGames = returnGamesForSeasonTeam(oppNames,
                                        season,
                                        _db)[[otherPrefix+metric,
                                             'OppName',
                                             otherPrefix+denomField,
                                             'TmName']]
    # st.write(oppGames)

    # Drop the games that were against the team in question
    oppGames = oppGames.loc[oppGames['OppName'] != team]

    # For every opponent team plays more than 1 game against,
    # append that opponent's games to oppGames prior to getting the average
    # Example: Tm A plays Tm B twice
    # Tm B's games should have 2 records each in OppGames
    oppCnt = tmGames['OppName'].value_counts()
    multOpponents = oppCnt.loc[oppCnt > 1]
    # st.write(multOpponents)
    for idx, val in multOpponents.iteritems():
        gamesToAppend = oppGames.loc[oppGames['TmName'] == idx]
        for x in range(1, val):
            oppGames = oppGames.append(gamesToAppend)
    # st.write(oppGames)

    # Aggregate the opponents' metric and denominator fields
    oppAggMetric = oppGames[[otherPrefix+metric, otherPrefix+denomField]].sum()
    # st.write(oppAggMetric)

    st.write(f"{team}'s opponents' total {otherPrefix+metric} in \
        {oppAggMetric[otherPrefix+denomField]} non-{team} {denomField}: \
        {oppAggMetric[otherPrefix+metric]}")

    oppAvgMetric = oppAggMetric[otherPrefix+metric] / oppAggMetric[otherPrefix+denomField] * normalizeConst
    st.write(f"Thats {oppAvgMetric} {suffix}")

    oaMetric = (tmAvgMetric - oppAvgMetric)

    return oaMetric



def opponentAdjustMetricAllSeasons(_db, prefix='Tm', metric='Margin', suffix='per40', season = 2020):
    # Figure out the prefix
    assert prefix in ['Opp', 'Tm'], 'Must be Opp or Tm as the prefix'
    otherPrefix = 'Opp' if prefix == 'Tm' else 'Tm'

    # Figure out the field to sum for denominator of team and opps
    assert suffix in ['per40', 'perPoss', 'perGame'], 'Bad suffix'
    denomField = 'Mins' if suffix == 'per40' else suffix[-4:]
    normalizeConst = 40 if suffix == 'per40' else 1

    # Return all games for all seasons
    print('here')
    # TODO figure out how to make all seasons query faster
    query = {'Season': season}
    fields = {'TmName': 1,
              'Season': 1,
              'OppName': 1,
              prefix+metric: 1,
              prefix+denomField: 1,
              otherPrefix+metric: 1,
              otherPrefix+denomField: 1}
    allGames = db.games.find(query, fields)
    allGames = pd.DataFrame(list(allGames))
    print('here2')
    #st.write(allGames)
    print('here3')
    # Aggregate the season's values for each team, for both metric and denominator
    seasonAggMetric = allGames.groupby(['TmName', 'Season'])[prefix+metric,prefix+denomField,otherPrefix+metric,otherPrefix+denomField].sum()
    seasonAggMetric[prefix+metric+suffix] = seasonAggMetric[prefix+metric] / seasonAggMetric[prefix+denomField] * normalizeConst
    #st.write(seasonAggMetric, 'Above is each teams aggregated stats for the season')
    
    # Rename for merging, only bring un-aggregated fields
    #mergeToGames = allGames.groupby(['TmName', 'Season'])[otherPrefix+metric, otherPrefix+denomField].sum()
    mergeToGames = seasonAggMetric[[otherPrefix+metric,otherPrefix+denomField]].rename(columns={otherPrefix+metric: 'OppTotal_'+otherPrefix+metric, otherPrefix+denomField: 'OppTotal_'+otherPrefix+denomField})
    #mergeToGames = mergeToGames[['OppTotal_'+prefix+metric, 'OppTotal_'+prefix+denomField]]
    #st.write(mergeToGames,"above is the aggregated metrics, renamed for merging back into games")

    allGames = pd.merge(allGames, mergeToGames, left_on=['OppName', 'Season'], right_on=['TmName', 'Season'])
    # st.write(allGames)

    # Remove the individual game's values from the opponent's aggregated metrics
    # TODO what happens if I have 2 games I only remove one game's from the unaggregated version
    # Could look into a groupby for tm and opp and subtract those via merge
    # Other side: excluding this game, I am considered an 'average opponent'?

    allGames['OppTotal_'+otherPrefix+metric+'_Adjusted'] = allGames['OppTotal_'+otherPrefix+metric] - allGames[prefix+metric]
    allGames['OppTotal_'+otherPrefix+denomField+'_Adjusted'] = allGames['OppTotal_'+otherPrefix+denomField] - allGames[prefix+denomField]
    #st.write(allGames)

    seasonAggOppMetric = allGames.groupby(['TmName', 'Season'])['OppTotal_'+otherPrefix+metric+'_Adjusted', 'OppTotal_'+otherPrefix+denomField+'_Adjusted'].sum()
    seasonAggOppMetric['Opp_'+otherPrefix+metric+suffix] = seasonAggOppMetric['OppTotal_'+otherPrefix+metric+'_Adjusted']/seasonAggOppMetric['OppTotal_'+otherPrefix+denomField+'_Adjusted']*normalizeConst
    #st.write(seasonAggOppMetric,'Above is season aggregated otherprefix + metric')

    seasonAggMetric = pd.merge(seasonAggMetric,seasonAggOppMetric,on=['TmName','Season']).reset_index()
    #st.write(seasonAggMetric)

    # TODO determine to add or subtract
    seasonAggMetric['OA_'+prefix+metric+suffix] = seasonAggMetric[prefix+metric+suffix] - seasonAggMetric['Opp_'+otherPrefix+metric+suffix]
    #st.write(seasonAggMetric)
    return seasonAggMetric

if __name__ == "__main__":
    pass
