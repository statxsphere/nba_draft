import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import ast

# Load the data
@st.cache_data
def load_data():
    df = pd.read_csv('data/final/temp.csv')
    df['floor_player_rating'] = df['floor_player_rating'].apply(ast.literal_eval)
    df['ceiling_player_rating'] = df['ceiling_player_rating'].apply(ast.literal_eval)
    return df

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
if sort_order == "Descending":
    df = df.sort_values(by=sort_column, ascending=False)
else:
    df = df.sort_values(by=sort_column, ascending=True)

if not year_filter == 'All':
    df = df[df['year'] == year_filter]

if search_query:
    df = df[df['name'].str.contains(search_query, case=False)]

# Main content
st.title("NBA Draft Prospect Dashboard")

# Function to get color
def get_color(rating):
    r = max(0, min(255, int(255 * (10 - rating) / 9)))
    g = max(0, min(255, int(255 * rating / 9)))
    b = 0
    return f'rgb({r}, {g}, {b})'

for _, player in df.iterrows():
    with st.container():
        col1, col2, col3 = st.columns([2, 1, 3])
        
        with col1:
            st.subheader(player['name'])
            st.write(f"Pick: {player['pick']}")
        
        with col2:
            peak_color = f"rgb({int(255 * (1 - player['Peak Impact Rating'] / 10))}, {int(255 * (player['Peak Impact Rating'] / 10))}, 0)"
            immediate_color = f"rgb({int(255 * (1 - player['Immediate Impact Rating'] / 10))}, {int(255 * (player['Immediate Impact Rating'] / 10))}, 0)"
           
            st.markdown(f"""
                <p>Peak Impact Rating:</p>
                <p style='font-size: 24px; font-weight: bold;'>
                    <span style='color:{peak_color};'>{player['Peak Impact Rating']:.2f}</span>
                </p>
            """, unsafe_allow_html=True)
           
            st.markdown(f"""
                <p>Immediate Impact Rating:</p>
                <p style='font-size: 24px; font-weight: bold;'>
                    <span style='color:{immediate_color};'>{player['Immediate Impact Rating']:.2f}</span>
                </p>
            """, unsafe_allow_html=True)

        with col3:
            # Prepare data for the chart
            floor_data = player['floor_player_rating']
            ceiling_data = player['ceiling_player_rating']
           
            years = list(set(floor_data.keys()) | set(ceiling_data.keys()))
            years.sort()
           
            floor_values = [floor_data.get(year, None) for year in years]
            ceiling_values = [ceiling_data.get(year, None) for year in years]
           
            # Create the chart
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=years, y=ceiling_values, mode='lines+markers', name=f"{player['ceiling_player']} (ceiling)"))
            fig.add_trace(go.Scatter(x=years, y=floor_values, mode='lines+markers', name=f"{player['floor_player']} (floor)"))
           
            fig.update_layout(
                title=f"Player Comps",
                xaxis_title="Year",
                yaxis_title="Rating",
                legend_title="Comparison Players",
                height=250
            )
           
            st.plotly_chart(fig, use_container_width=True)
       
        
        st.divider()