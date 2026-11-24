import pandas as pd
import streamlit as st

from interface import masthead, results_table
from services.api import get_options, predict

REQUIRED = ["Season", "Round", "CircuitID", "DriverID", "ConstructorName", "Grid"]


@st.cache_data(ttl=60)
def load_options():
    return get_options()


def category(label, key, default):
    values = options.get(key, [])
    if not values:
        return st.text_input(label, value=default)
    return st.selectbox(label, values, index=values.index(default) if default in values else 0,
                        accept_new_options=True)


masthead("Race prediction", "Estimate a driver’s chance of finishing in the top ten.")
single, batch = st.tabs(["Single driver", "Batch upload"])

with single:
    options = load_options()
    if "error" in options:
        st.warning("Driver and circuit lists are unavailable. Enter the values manually.")
    setup, output = st.columns([1.45, 1], gap="medium")
    with setup.container(border=True):
        st.subheader("Race setup")
        with st.form("race_setup"):
            left, right = st.columns(2)
            with left:
                season = st.number_input("Season", 1950, 2100, 2022)
                circuit = category("Circuit", "CircuitID", "bahrain")
                driver = category("Driver", "DriverID", "leclerc")
            with right:
                round_num = st.number_input("Round", 1, 30, 1)
                grid = st.number_input("Starting grid", 0, 40, 1)
                team = category("Constructor", "ConstructorName", "Ferrari")
            submitted = st.form_submit_button("Run prediction", type="primary", width="stretch")
    with output.container(border=True):
        st.subheader("Predicted finish")
        if submitted:
            if not all([circuit, driver, team]):
                st.error("Enter a circuit, driver, and constructor.")
            else:
                row = dict(zip(REQUIRED, [season, round_num, circuit, driver, team, grid]))
                with st.spinner("Running the model…"):
                    result = predict([row])
                if isinstance(result, dict):
                    st.error(result.get("error", "Prediction unavailable."))
                elif result:
                    st.session_state.last_prediction = result[0]
        last = st.session_state.get("last_prediction")
        if last:
            st.caption(f"{last['driver_id']} · {last['constructor_name']} · Grid {last['grid']}")
            st.markdown("### " + ("Inside the top 10" if last["prediction"] else "Outside the top 10"))
            chance = last.get("probability")
            if chance is not None:
                st.metric("Top-10 probability", f"{chance:.1%}")
                st.progress(float(chance))
            st.caption(f"{last['circuit_id']} · {last['season']} · Round {last['round']}")
            with st.expander("Prediction details"):
                results_table([last])
        else:
            st.metric("Top-10 probability", "—")
            st.caption("Run a race setup to see the model’s forecast here.")

with batch:
    st.subheader("Predict a starting grid")
    st.caption("Upload one row per driver with these columns: " + ", ".join(REQUIRED))
    upload = st.file_uploader("Race entries", type="csv")
    if upload:
        try:
            frame = pd.read_csv(upload)
        except (ValueError, UnicodeError) as error:
            st.error(f"Could not read this CSV: {error}")
        else:
            st.dataframe(frame, hide_index=True, width="stretch")
            missing = [key for key in REQUIRED if key not in frame]
            if missing:
                st.error("Missing columns: " + ", ".join(missing))
            elif frame.empty or frame[REQUIRED].isna().any().any():
                st.warning("Include at least one driver and fill in all required values.")
            elif st.button("Run batch prediction", type="primary"):
                with st.spinner("Predicting finishes…"):
                    result = predict(frame[REQUIRED].to_dict(orient="records"))
                if isinstance(result, dict):
                    st.error(result.get("error", "Batch prediction unavailable."))
                else:
                    st.subheader("Batch results")
                    results_table(result)
