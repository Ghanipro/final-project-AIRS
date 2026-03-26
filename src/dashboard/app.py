import pandas as pd
import streamlit as st
import plotly.express as px
import os

st.set_page_config(page_title="AIRS Model Comparison", layout="wide")

st.title("AIRS RL Model Comparison Dashboard")

leaderboard_path = st.sidebar.text_input("Leaderboard path", "data/results/leaderboard.csv")

if not os.path.exists(leaderboard_path):
    st.error(f"Leaderboard not found at: {leaderboard_path}. Run: python scripts/evaluate_all.py")
    st.stop()

df = pd.read_csv(leaderboard_path)

st.subheader("Leaderboard (per method x seed)")
st.dataframe(df.sort_values(["breach_rate", "avg_return"], ascending=[True, False]), use_container_width=True)

st.subheader("Aggregated by method (mean across seeds)")
agg = df.groupby("method").agg(
    breach_rate=("breach_rate", "mean"),
    contain_rate=("contain_rate", "mean"),
    avg_return=("avg_return", "mean"),
    avg_ep_len=("avg_ep_len", "mean"),
    avg_action_cost=("avg_action_cost", "mean"),
    avg_availability_loss=("avg_availability_loss", "mean"),
    avg_security_loss=("avg_security_loss", "mean"),
    avg_max_stage=("avg_max_stage", "mean"),
).reset_index()

st.dataframe(agg.sort_values(["breach_rate", "avg_return"], ascending=[True, False]), use_container_width=True)

c1, c2 = st.columns(2)
with c1:
    fig = px.box(df, x="method", y="breach_rate", points="all", title="Breach rate by method")
    st.plotly_chart(fig, use_container_width=True)

with c2:
    fig = px.box(df, x="method", y="avg_return", points="all", title="Average return by method")
    st.plotly_chart(fig, use_container_width=True)

c3, c4 = st.columns(2)
with c3:
    fig = px.scatter(
        df,
        x="avg_action_cost",
        y="breach_rate",
        color="method",
        hover_data=["seed", "avg_return"],
        title="Cost vs Breach Rate (per seed)",
    )
    st.plotly_chart(fig, use_container_width=True)

with c4:
    fig = px.box(df, x="method", y="avg_availability_loss", points="all", title="Availability loss by method")
    st.plotly_chart(fig, use_container_width=True)
