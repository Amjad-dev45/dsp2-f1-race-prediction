import streamlit as st

from interface import masthead, results_table
from services.api import get_past_predictions

masthead("Prediction history", "Review saved forecasts by date and prediction source.")
with st.container(border=True):
    start, end, channel = st.columns(3)
    first = start.date_input("From", value=None)
    last = end.date_input("Through", value=None)
    source = channel.selectbox("Source", ["All", "Webapp", "Scheduled"])

if first and last and first > last:
    st.warning("Choose an end date on or after the start date.")
else:
    with st.spinner("Loading predictions…"):
        records = get_past_predictions(first and first.isoformat(), last and last.isoformat(),
                                       source.lower())
    if isinstance(records, dict):
        st.error(records.get("error", "Could not load history."))
    elif records:
        total, top, drivers = st.columns(3)
        total.metric("Predictions", len(records), border=True)
        top.metric("Predicted top 10", sum(bool(row["prediction"]) for row in records), border=True)
        drivers.metric("Drivers", len({row["driver_id"] for row in records}), border=True)
        results_table(records)
    else:
        st.info("No predictions match these filters. Run a prediction or change the date range.")
