import os
import numpy as np
import streamlit as st
import networkx as nx
import plotly.graph_objects as go

from src.environment.airs_env import AIRSEnv, AIRSConfig
from src.environment.enterprise_topology import EnterpriseTopology
from src.demo.demo_runner import load_model, step_demo


def build_graph(topology: EnterpriseTopology):
    G = nx.Graph()
    for n in topology.nodes:
        G.add_node(n["id"], **n)
    for u, v in topology.edges:
        G.add_edge(u, v)
    return G


def graph_figure(G, env: AIRSEnv):
    pos = nx.spring_layout(G, seed=42)

    edge_x, edge_y = [], []
    for u, v in G.edges():
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        edge_x += [x0, x1, None]
        edge_y += [y0, y1, None]

    edge_trace = go.Scatter(x=edge_x, y=edge_y, line=dict(width=1, color="#888"),
                            hoverinfo="none", mode="lines")

    node_x, node_y, node_text, node_color = [], [], [], []
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)

        stage = int(env.c[node])
        intensity = float(env.b[node])
        node_info = G.nodes[node]
        text = f"{node_info['name']}<br>zone={node_info['zone']} type={node_info['type']}<br>stage={stage} intensity={intensity:.2f}"
        node_text.append(text)

        # color by stage
        node_color.append(stage)

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode="markers+text",
        text=[str(n) for n in G.nodes()],
        textposition="bottom center",
        hovertext=node_text,
        hoverinfo="text",
        marker=dict(
            showscale=True,
            colorscale="YlOrRd",
            color=node_color,
            size=18,
            colorbar=dict(title="Stage"),
            line_width=1,
        )
    )

    fig = go.Figure(data=[edge_trace, node_trace])
    fig.update_layout(
        showlegend=False,
        margin=dict(l=10, r=10, t=10, b=10),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        height=500
    )
    return fig


st.set_page_config(page_title="AIRS Enterprise Demo", layout="wide")
st.title("AIRS Enterprise Demo (10-node network)")

topology = EnterpriseTopology()
G = build_graph(topology)

algo = st.sidebar.selectbox("Policy", ["PPO", "DQN", "A2C", "RecurrentPPO", "QRDQN", "RuleBased", "Manual"])
seed = st.sidebar.number_input("Seed", min_value=0, max_value=999, value=0)

if "env" not in st.session_state:
    cfg = AIRSConfig(n_nodes=10, max_steps=150)
    env = AIRSEnv(cfg)
    obs, info = env.reset(seed=seed)
    st.session_state.env = env
    st.session_state.obs = obs
    st.session_state.model = None

env = st.session_state.env
obs = st.session_state.obs

model_path = st.sidebar.text_input(
    "Model path",
    value="data/models/PPO/seed_0/model.zip",
)
if algo not in ("Manual", "RuleBased"):
    if st.sidebar.button("Load model"):
        st.session_state.model = load_model(algo, model_path, env)
        st.sidebar.success("Model loaded")

st.subheader("Network State")
st.plotly_chart(graph_figure(G, env), use_container_width=True)

st.subheader("Controls")
c1, c2 = st.columns(2)
with c1:
    step_btn = st.button("Step")
with c2:
    reset_btn = st.button("Reset")

manual_action = None
if algo == "Manual":
    manual_action = st.number_input("Manual action (int)", min_value=0, max_value=1+6*env.n-1, value=0)

if reset_btn:
    obs, info = env.reset(seed=seed)
    st.session_state.obs = obs
    st.experimental_rerun()

if step_btn:
    model = st.session_state.model
    obs, reward, terminated, truncated, info, action = step_demo(env, obs, algo, model, manual_action)
    st.session_state.obs = obs

    st.write("Action:", action)
    st.write("Reward:", reward)
    st.write("Info:", info)

    if terminated:
        st.success("Episode terminated (breach or contained)")
    if truncated:
        st.warning("Episode truncated (max steps)")

st.write("Current t:", env._t)