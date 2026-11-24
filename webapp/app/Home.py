import streamlit as st

st.set_page_config(page_title="F1 Race Predictions", page_icon="🏁", layout="wide")
race = st.Page("pages/1_Prediction.py", title="Race prediction", icon=":material/sports_motorsports:")
history = st.Page("pages/2_Past_Predictions.py", title="Prediction history", icon=":material/history:")

# Keep the original prediction and history URLs available to bookmarked sessions.
def landing():
    st.switch_page(race)


home = st.Page(landing, title="Home", default=True, visibility="hidden")
st.navigation([home, race, history], position="top").run()
