import os
import ast

import streamlit as st
import pandas as pd
import plotly.graph_objects as go

DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'final', 'temp.csv')


# Load the data
@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH)
    df['floor_player_rating'] = df['floor_player_rating'].apply(ast.literal_eval)
    df['ceiling_player_rating'] = df['ceiling_player_rating'].apply(ast.literal_eval)
    return df


# Function to get color
def get_color(rating):
    r = max(0, min(255, int(255 * (1 - rating / 10))))
    g = max(0, min(255, int(255 * rating / 10)))
    return f'rgb({r}, {g}, 0)'


def rating_block(label, rating):
    st.markdown(f"""
        <p>{label}:</p>
        <p style='font-size: 24px; font-weight: bold;'>
            <span style='color:{get_color(rating)};'>{rating:.2f}</span>
        </p>
    """, unsafe_allow_html=True)


def comps_chart(player):
    floor_data = player['floor_player_rating']
    ceiling_data = player['ceiling_player_rating']

    years = sorted(set(floor_data.keys()) | set(ceiling_data.keys()))

    floor_values = [floor_data.get(year) for year in years]
    ceiling_values = [ceiling_data.get(year) for year in years]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=years, y=ceiling_values, mode='lines+markers', name=f"{player['ceiling_player']} (ceiling)"))
    fig.add_trace(go.Scatter(x=years, y=floor_values, mode='lines+markers', name=f"{player['floor_player']} (floor)"))

    fig.update_layout(
        title="Player Comps",
        xaxis_title="Year",
        yaxis_title="Rating",
        legend_title="Comparison Players",
        height=250
    )
    return fig


df = load_data()

# Sidebar for sorting and filtering options
st.sidebar.title("Options")

sort_column = st.sidebar.selectbox(
    "Sort players by:",
    ["pick", "Peak Impact Rating", "Immediate Impact Rating", "name"]
)

sort_order = st.sidebar.radio("Sort order:", ["Ascending", "Descending"])

year_filter = st.sidebar.selectbox("Filter by Year:", [2024, 2023, 'All'])

# Search feature
search_query = st.sidebar.text_input("Search player:")

# Apply sorting, filtering, and search
df = df.sort_values(by=sort_column, ascending=(sort_order == "Ascending"))

if year_filter != 'All':
    df = df[df['year'] == year_filter]

if search_query:
    df = df[df['name'].str.contains(search_query, case=False, regex=False)]

# Main content
st.title("NBA Draft Prospect Dashboard")

for i, player in df.iterrows():
    with st.container():
        col1, col2, col3 = st.columns([2, 1, 3])

        with col1:
            st.subheader(player['name'])
            st.write(f"Pick: {player['pick']}")

        with col2:
            rating_block("Peak Impact Rating", player['Peak Impact Rating'])
            rating_block("Immediate Impact Rating", player['Immediate Impact Rating'])

        with col3:
            st.plotly_chart(comps_chart(player), use_container_width=True, key=f"comps_{i}")

        st.divider()
