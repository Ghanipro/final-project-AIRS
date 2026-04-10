import json
import os
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(page_title="AIRS Model Comparison", layout="wide")

st.title("AIRS RL Model Comparison Dashboard")


def _latest_file(patterns: list[str], fallback: str) -> str:
    candidates = []
    for pattern in patterns:
        for path in Path("data").glob(pattern):
            if path.is_file():
                candidates.append(path)

    if not candidates:
        return fallback

    latest = max(candidates, key=lambda p: p.stat().st_mtime)
    return latest.as_posix()


def find_latest_marl_summary() -> str:
    return _latest_file(
        ["marl_eval*/summary_all.csv", "marl_eval*/summary.json", "marl_eval*/summary_all.json"],
        "data/marl_eval_all/summary_all.csv",
    )


def find_latest_marl_comparison() -> str:
    return _latest_file(
        ["marl_eval*/comparison_vs_rulebased.csv"],
        "data/marl_eval_all/comparison_vs_rulebased.csv",
    )


def load_marl_payload(summary_path: str) -> pd.DataFrame | dict | None:
    path = Path(summary_path)
    if not path.exists():
        return None
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def normalize_marl_payload(payload):
    """Normalize JSON payloads to DataFrame when possible."""
    if payload is None:
        return None
    if isinstance(payload, pd.DataFrame):
        return payload
    if isinstance(payload, list):
        if len(payload) == 0:
            return pd.DataFrame()
        if isinstance(payload[0], dict):
            return pd.DataFrame(payload)
    return payload


def _format_float(value, digits=3):
    if value is None:
        return "n/a"
    try:
        return f"{float(value):.{digits}f}"
    except Exception:
        return "n/a"


def _pick_best_marl_row(marl_df: pd.DataFrame) -> dict:
    """Pick a representative best blue model row for top metrics cards."""
    cands = marl_df[marl_df.get("blue_algo", "") != "RuleBasedBlue"].copy()
    if cands.empty:
        return marl_df.iloc[0].to_dict()

    cols = ["breach_rate", "contain_rate", "avg_return", "avg_security_loss", "avg_availability_loss", "avg_blue_cost"]
    for c in cols:
        if c not in cands.columns:
            cands[c] = 0.0

    cands["_rank_score"] = (
        -4.0 * cands["breach_rate"].astype(float)
        + 3.0 * cands["contain_rate"].astype(float)
        + 1.5 * cands["avg_return"].astype(float)
        - 1.0 * cands["avg_security_loss"].astype(float)
        - 0.8 * cands["avg_availability_loss"].astype(float)
        - 0.5 * cands["avg_blue_cost"].astype(float)
    )
    return cands.sort_values("_rank_score", ascending=False).iloc[0].to_dict()

leaderboard_path = st.sidebar.text_input("Leaderboard path", "data/results/leaderboard.csv")
marl_summary_default = find_latest_marl_summary()
marl_summary_path = st.sidebar.text_input("MARL summary path", marl_summary_default)
comparison_default = find_latest_marl_comparison()
comparison_path = st.sidebar.text_input("MARL comparison path", comparison_default)

st.caption(f"Active MARL summary: {marl_summary_path}")

st.subheader("Single-Agent Baseline Leaderboard")

if not os.path.exists(leaderboard_path):
    st.info(f"Single-agent leaderboard not found at: {leaderboard_path}. Run: python scripts/evaluate_all.py")
else:
    df = pd.read_csv(leaderboard_path)

    target_methods = ["PPO", "DQN", "A2C", "RuleBased"]
    df = df[df["method"].isin(target_methods)].copy()

    if df.empty:
        st.info("No rows for PPO/DQN/A2C/RuleBased found in leaderboard.")
    else:
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

st.subheader("Red-vs-Blue MARL Evaluation")

summary_payload = load_marl_payload(marl_summary_path)
comparison_payload = load_marl_payload(comparison_path)

summary_payload = normalize_marl_payload(summary_payload)
comparison_payload = normalize_marl_payload(comparison_payload)

if summary_payload is not None:
    if isinstance(summary_payload, pd.DataFrame):
        marl_df = summary_payload.copy()
        if "blue_algo" in marl_df.columns:
            summary_row = _pick_best_marl_row(marl_df)
        else:
            summary_row = marl_df.iloc[0].to_dict()

        if comparison_payload is not None and isinstance(comparison_payload, pd.DataFrame) and not comparison_payload.empty:
            best_row = comparison_payload.sort_values(
                ["beats_core_security_metrics", "beats_avg_return_per_step", "beats_avg_ep_len_taskaware"],
                ascending=[False, False, False],
            ).iloc[0]
            st.success(
                f"Best current MARL candidate: {best_row['blue_algo']} | core-security={bool(best_row['beats_core_security_metrics'])} | "
                f"return/step={bool(best_row['beats_avg_return_per_step'])} | task-aware ep-len={bool(best_row['beats_avg_ep_len_taskaware'])}"
            )

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("MARL Breach Rate", _format_float(summary_row.get("breach_rate")))
        m2.metric("MARL Contain Rate", _format_float(summary_row.get("contain_rate")))
        m3.metric("MARL Avg Return", _format_float(summary_row.get("avg_return"), 2))
        m4.metric("MARL Avg Ep Len", _format_float(summary_row.get("avg_ep_len"), 2))

        if "blue_algo" in marl_df.columns:
            rb_row = marl_df[marl_df["blue_algo"] == "RuleBasedBlue"]
            model_rows = marl_df[marl_df["blue_algo"] != "RuleBasedBlue"].copy()

            if not model_rows.empty:
                st.subheader("Blue vs Red Agent Performance")
                st.caption("Blue success is containment rate. Red success is breach rate (higher means red is winning more).")

                model_rows["blue_success_rate"] = model_rows["contain_rate"].astype(float)
                model_rows["red_success_rate"] = model_rows["breach_rate"].astype(float)
                model_rows["net_blue_advantage"] = model_rows["blue_success_rate"] - model_rows["red_success_rate"]

                k1, k2, k3 = st.columns(3)
                best_blue = model_rows.sort_values("blue_success_rate", ascending=False).iloc[0]
                best_red = model_rows.sort_values("red_success_rate", ascending=False).iloc[0]
                best_net = model_rows.sort_values("net_blue_advantage", ascending=False).iloc[0]
                k1.metric("Best Blue Success", f"{best_blue['blue_algo']}: {float(best_blue['blue_success_rate']):.3f}")
                k2.metric("Strongest Red Pressure", f"vs {best_red['blue_algo']}: {float(best_red['red_success_rate']):.3f}")
                k3.metric("Best Net Blue Advantage", f"{best_net['blue_algo']}: {float(best_net['net_blue_advantage']):.3f}")

                c1, c2 = st.columns(2)
                with c1:
                    perf_long = model_rows[["blue_algo", "blue_success_rate", "red_success_rate"]].melt(
                        id_vars=["blue_algo"],
                        var_name="side",
                        value_name="rate",
                    )
                    fig = px.bar(
                        perf_long,
                        x="blue_algo",
                        y="rate",
                        color="side",
                        barmode="group",
                        title="Blue Success (Contain) vs Red Success (Breach)",
                    )
                    st.plotly_chart(fig, use_container_width=True)

                with c2:
                    fig = px.bar(
                        model_rows,
                        x="blue_algo",
                        y="net_blue_advantage",
                        color="blue_algo",
                        title="Net Blue Advantage (Contain - Breach)",
                    )
                    st.plotly_chart(fig, use_container_width=True)

                c3, c4 = st.columns(2)
                with c3:
                    fig = px.scatter(
                        model_rows,
                        x="avg_blue_cost",
                        y="avg_return",
                        color="blue_algo",
                        size="contain_rate",
                        hover_data=["breach_rate", "avg_security_loss", "avg_availability_loss"],
                        title="Blue Efficiency: Return vs Cost (bubble=size contain rate)",
                    )
                    st.plotly_chart(fig, use_container_width=True)

                with c4:
                    fig = px.scatter(
                        model_rows,
                        x="breach_rate",
                        y="contain_rate",
                        color="blue_algo",
                        size="avg_return",
                        hover_data=["avg_ep_len", "avg_security_loss", "avg_availability_loss", "avg_blue_cost"],
                        title="Security Tradeoff: Breach vs Contain",
                    )
                    st.plotly_chart(fig, use_container_width=True)

                if not rb_row.empty:
                    rb = rb_row.iloc[0]
                    st.caption(
                        f"RuleBased reference: breach={float(rb['breach_rate']):.3f}, contain={float(rb['contain_rate']):.3f}, "
                        f"avg_return={float(rb['avg_return']):.2f}, avg_ep_len={float(rb['avg_ep_len']):.2f}"
                    )

        if comparison_payload is not None and isinstance(comparison_payload, pd.DataFrame):
            st.subheader("MARL vs RuleBased")
            st.dataframe(comparison_payload, use_container_width=True)

        st.subheader("MARL Summary")
        st.dataframe(marl_df, use_container_width=True)

    else:
        st.json(summary_payload, expanded=False)

    marl_episodes_path = os.path.join(os.path.dirname(marl_summary_path), "episodes.csv")
    if os.path.exists(marl_episodes_path):
        marl_df = pd.read_csv(marl_episodes_path)
        d1, d2 = st.columns(2)

        with d1:
            fig = px.histogram(marl_df, x="ep_len", nbins=20, title="MARL Episode Length Distribution")
            st.plotly_chart(fig, use_container_width=True)

        with d2:
            fig = px.histogram(marl_df, x="return", nbins=20, title="MARL Return Distribution")
            st.plotly_chart(fig, use_container_width=True)

        d3, d4 = st.columns(2)
        with d3:
            fig = px.line(marl_df, x="episode", y="avg_security_loss", title="MARL Security Loss Over Episodes")
            st.plotly_chart(fig, use_container_width=True)

        with d4:
            fig = px.line(marl_df, x="episode", y="avg_availability_loss", title="MARL Availability Loss Over Episodes")
            st.plotly_chart(fig, use_container_width=True)
    else:
        pass
else:
    st.info(f"MARL summary not found at: {marl_summary_path}. Run: python scripts/evaluate_marl.py --blue-algo PPO --episodes 200")
