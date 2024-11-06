import streamlit as st
import pickle
import pandas as pd
import altair as alt

# CSS styling
cricket_css = """
<style>
body {
    background-color: #1e1e1e; /* Dark background */
    color: #ffffff; /* White text */
    font-family: Arial, sans-serif;
}

.stButton button {
    background-color: #008cba; /* Cricket blue button background */
    color: white; /* Text color */
    font-weight: bold;
    border-radius: 5px;
    padding: 10px 20px;
    transition: background-color 0.3s;
}

.stButton button:hover {
    background-color: #005f7a; /* Darker blue on hover */
}

.stTitle {
    color: #ffffff; /* White title */
}

.stHeader {
    color: #ffffff; /* White header */
}

.stSelectbox, .stNumberInput {
    background-color: #333333; /* Dark gray input fields */
    color: #ffffff; /* White text */
}

.stDataFrame {
    border-collapse: collapse;
    width: 100%;
}

.stDataFrame th, .stDataFrame td {
    border: 1px solid #dddddd;
    text-align: left;
    padding: 8px;
}

.stDataFrame th {
    background-color: #005f7a; /* Dark blue header */
    color: white; /* Text color */
}

.stDataFrame tr:nth-child(even) {
    background-color: #444444; /* Darker gray alternate row background color */
}
</style>
"""

# List of teams and cities
teams = ['Sunrisers Hyderabad', 'Mumbai Indians', 'Royal Challengers Bangalore',
         'Kolkata Knight Riders', 'Kings XI Punjab', 'Chennai Super Kings',
         'Rajasthan Royals', 'Delhi Capitals']

cities = ['Hyderabad', 'Bangalore', 'Mumbai', 'Indore', 'Kolkata', 'Delhi',
          'Chandigarh', 'Jaipur', 'Chennai', 'Cape Town', 'Port Elizabeth',
          'Durban', 'Centurion', 'East London', 'Johannesburg', 'Kimberley',
          'Bloemfontein', 'Ahmedabad', 'Cuttack', 'Nagpur', 'Dharamsala',
          'Visakhapatnam', 'Pune', 'Raipur', 'Ranchi', 'Abu Dhabi',
          'Sharjah', 'Mohali', 'Bengaluru']

# Load machine learning model and player statistics data
pipe = pickle.load(open('pipe.pkl', 'rb'))
player_stats_df = pd.read_csv('IPL Player stat.csv')

# Streamlit app title
st.title('IPL Win Predictor')

# Select teams, city, target, score, overs, wickets
with st.form(key='input_form'):
    col1, col2 = st.columns(2)
    with col1:
        batting_team = st.selectbox('Select the batting team', sorted(teams))
    with col2:
        bowling_team = st.selectbox('Select the bowling team', sorted(teams))
    selected_city = st.selectbox('Select host city', sorted(cities))
    target = st.number_input('Target')
    col3, col4, col5 = st.columns(3)
    with col3:
        score = st.number_input('Score')
    with col4:
        overs = st.number_input('Overs completed')
    with col5:
        wickets = st.number_input('Wickets out')
    submit_button = st.form_submit_button(label='Predict Probability')

# Display player statistics for selected teams
if submit_button:
    runs_left = target - score
    balls_left = 120 - (overs * 6)
    wickets = 10 - wickets
    crr = score / overs
    rrr = (runs_left * 6) / balls_left

    # Display prediction results
    st.subheader('Prediction Result')

    # Use input data for prediction
    input_df = pd.DataFrame({
        'batting_team': [batting_team],
        'bowling_team': [bowling_team],
        'city': [selected_city],
        'runs_left': [runs_left],
        'balls_left': [balls_left],
        'wickets': [wickets],
        'total_runs_x': [target],
        'crr': [crr],
        'rrr': [rrr]
    })

    # Predict win probability
    result = pipe.predict_proba(input_df)
    loss = result[0][0]
    win = result[0][1]

    # Display win probability
    st.write(f"{batting_team} Win Probability: {round(win * 100, 2)}%")
    st.write(f"{bowling_team} Win Probability: {round(loss * 100, 2)}%")

    # Plot win probability
    win_prob_df = pd.DataFrame({'Team': [batting_team, bowling_team], 'Win Probability': [win, loss]})
    chart = alt.Chart(win_prob_df).mark_bar().encode(
        x='Team',
        y='Win Probability',
        color='Team'
    ).properties(
        title='Win Probability'
    )
    st.altair_chart(chart, use_container_width=True)

    # Display additional information
    st.subheader('Additional Information')

    player_stats_df = pd.read_csv('IPL Player stat.csv')

matches_df = pd.read_csv('matches.csv')

# Load player statistics dataset
player_stats_df = pd.read_csv('IPL Player stat.csv')

# Load unsold players dataset
unsold_players_df = pd.read_csv('Unsold_players.csv')

# Load top buys dataset
top_buys_df = pd.read_csv('Top_buys.csv')

# Load IPL players dataset
ipl_players_df = pd.read_csv('IPL_players.csv')

# Add functionality
st.sidebar.title('Player Analysis')

# Sidebar options
analysis_option = st.sidebar.radio('Select Analysis', ['Top Scorers', 'Top Wicket-takers', 'Unsold Players', 'Top Buys', 'IPL Players', 'Toss Winning Percentage'])

# Perform analysis based on selected option
if analysis_option == 'Top Scorers':
    top_scorers = player_stats_df.sort_values(by='runs', ascending=False).head(10)
    st.subheader('Top 10 Run Scorers in IPL')
    st.write(top_scorers[['player', 'runs']])

elif analysis_option == 'Top Wicket-takers':
    top_wicket_takers = player_stats_df.sort_values(by='wickets', ascending=False).head(10)
    st.subheader('Top 10 Wicket-takers in IPL')
    st.write(top_wicket_takers[['player', 'wickets']])

elif analysis_option == 'Unsold Players':
    st.subheader('Unsold Players')
    st.write(unsold_players_df)

elif analysis_option == 'Top Buys':
    st.subheader('Top Buys')
    st.write(top_buys_df)

elif analysis_option == 'IPL Players':
    st.subheader('IPL Players')
    st.write(ipl_players_df)

elif analysis_option == 'Toss Winning Percentage':
    # Calculate toss winning percentage for each team
    toss_counts = matches_df['toss_winner'].value_counts()
    total_matches_team1 = matches_df['team1'].value_counts()
    total_matches_team2 = matches_df['team2'].value_counts()
    total_matches = total_matches_team1.add(total_matches_team2, fill_value=0)
    toss_win_percentage = (toss_counts / total_matches) * 100

    # Display toss winning percentage for each team
    # Display toss winning percentage for each team
    st.subheader('Toss Winning Percentage for Each Team')
    st.dataframe(toss_win_percentage.rename('Toss Winning Percentage').to_frame().style.format("{:.2f}%"), width=800)