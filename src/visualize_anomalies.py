from collections import Counter
import math
import pandas as pd
import networkx as nx
from pyvis.network import Network

from build_graph import G

user_profile_df = pd.read_csv("data/user_profile.csv")
user_data_df = pd.read_csv("data/user_data_real.csv")

user_to_type = dict(zip(user_profile_df["user"], user_profile_df["type"]))
type_counts = Counter(user_to_type.values())

type_data_pairs = []
for _, row in user_data_df.iterrows():
    user = row["user"]
    data_name = row["data"]
    type_data_pairs.append((user_to_type[user], data_name))

pair_counter = Counter(type_data_pairs)
degree_centrality = nx.degree_centrality(G)

data_centrality = {
    node: score
    for node, score in degree_centrality.items()
    if G.nodes[node].get("type") == "data"
}

results = []
for (user_type, data_name), freq in pair_counter.items():
    rarity = 1 / freq
    importance = data_centrality.get(data_name, 0.0)
    type_size = type_counts[user_type]
    type_weight = math.log(1 + type_size)
    anomaly_score = rarity * importance * type_weight

    results.append(
        {
            "type": user_type,
            "data": data_name,
            "frequency": freq,
            "score": anomaly_score,
        }
    )

results = sorted(results, key=lambda item: item["score"], reverse=True)

score_threshold = 0.45
high_anomalies = {
    (row["type"], row["data"])
    for row in results
    if row["score"] >= score_threshold
}

anomalous_user_data_edges = set()
for _, row in user_data_df.iterrows():
    user = row["user"]
    data_name = row["data"]
    user_type = user_to_type[user]
    if (user_type, data_name) in high_anomalies:
        anomalous_user_data_edges.add((user, data_name))

net = Network(height="900px", width="100%", directed=True)

for node, attrs in G.nodes(data=True):
    node_type = attrs["type"]

    if node_type == "user":
        color = "red"
        title = f"{node} (User)"
    elif node_type == "user_type":
        color = "orange"
        title = f"{node} (User Type)"
    elif node_type == "data":
        color = "blue"
        title = f"{node} (Data)"
    else:
        color = "green"
        title = f"{node} (Decision / Ward)"

    net.add_node(node, color=color, title=title)

for source, target, attrs in G.edges(data=True):
    edge_type = attrs["type"]
    title = edge_type
    width = 1
    color = None

    if edge_type == "influences":
        weight = attrs.get("weight", 0.0)
        title = f"{edge_type} | weight={weight}"
        width = 1 + 4 * weight
    elif edge_type in {"prefers", "makes"}:
        width = 2
    elif edge_type == "uses" and (source, target) in anomalous_user_data_edges:
        user_type = user_to_type[source]
        score = next(
            row["score"]
            for row in results
            if row["type"] == user_type and row["data"] == target
        )
        color = "red"
        width = 5
        title = f"uses | anomaly=({user_type}, {target}) | score={score:.3f}"

    net.add_edge(source, target, title=title, width=width, color=color)

net.write_html("network_real_v2_anomaly.html")
print("Anomaly-highlighted graph created! Open network_real_v2_anomaly.html")

print("\n=== Highlighted Anomalies ===")
for pair in sorted(high_anomalies):
    print(pair)
