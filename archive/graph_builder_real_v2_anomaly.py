import pandas as pd
import networkx as nx
from pyvis.network import Network
from collections import Counter
import math

# -------------------
# 读取数据
# -------------------
user_profile_df = pd.read_csv("data/user_profile.csv")
user_data_df = pd.read_csv("data/user_data_real.csv")
user_decision_df = pd.read_csv("data/user_decision_real.csv")
tokyo_df = pd.read_csv("data/tokyo_ward_data_extended.csv")

G = nx.DiGraph()

# -------------------
# 标准化
# -------------------
def normalize(series, reverse=False):
    s = series.astype(float)
    min_val = s.min()
    max_val = s.max()

    if max_val == min_val:
        norm = pd.Series([0.5] * len(s), index=s.index)
    else:
        norm = (s - min_val) / (max_val - min_val)

    if reverse:
        norm = 1 - norm

    return norm

tokyo_df = tokyo_df.copy()
tokyo_df["rent_low_score"] = normalize(tokyo_df["avg_1k_rent_low_jpy"], reverse=True)
tokyo_df["rent_high_score"] = normalize(tokyo_df["avg_1k_rent_high_jpy"], reverse=True)
tokyo_df["population_density_score"] = normalize(tokyo_df["population_density_per_km2"], reverse=False)
tokyo_df["school_count_score"] = normalize(tokyo_df["elementary_schools"], reverse=False)
tokyo_df["crime_rate_score"] = normalize(tokyo_df["crime_rate"], reverse=True)
tokyo_df["transport_score_norm"] = normalize(tokyo_df["transport_score"], reverse=False)
tokyo_df["commercial_score_norm"] = normalize(tokyo_df["commercial_score"], reverse=False)
tokyo_df["livability_score_norm"] = normalize(tokyo_df["livability_score"], reverse=False)

# -------------------
# 节点
# -------------------
users = sorted(set(user_profile_df["user"]))
user_types = sorted(set(user_profile_df["type"]))
data_nodes = sorted({
    "rent_low",
    "rent_high",
    "population_density",
    "school_count",
    "crime_rate",
    "transport_score",
    "commercial_score",
    "livability_score"
})
decisions = sorted(set(tokyo_df["ward"]))

for u in users:
    G.add_node(u, type="user")
for t in user_types:
    G.add_node(t, type="user_type")
for d in data_nodes:
    G.add_node(d, type="data")
for dec in decisions:
    G.add_node(dec, type="decision")

# -------------------
# 边：User -> Type
# -------------------
for _, row in user_profile_df.iterrows():
    G.add_edge(row["user"], row["type"], type="has_type")

# -------------------
# 边：Type -> Data
# -------------------
type_to_data = {
    "student": ["school_count", "population_density", "rent_high", "transport_score", "commercial_score"],
    "worker": ["rent_low", "population_density", "transport_score", "commercial_score", "crime_rate"],
    "family": ["school_count", "rent_low", "livability_score", "crime_rate", "population_density"]
}

for t, ds in type_to_data.items():
    for d in ds:
        G.add_edge(t, d, type="prefers")

# -------------------
# 边：User -> Data
# -------------------
for _, row in user_data_df.iterrows():
    G.add_edge(row["user"], row["data"], type="uses")

# -------------------
# 边：Data -> Decision
# -------------------
for _, row in tokyo_df.iterrows():
    ward = row["ward"]
    data_to_value = {
        "rent_low": row["rent_low_score"],
        "rent_high": row["rent_high_score"],
        "population_density": row["population_density_score"],
        "school_count": row["school_count_score"],
        "crime_rate": row["crime_rate_score"],
        "transport_score": row["transport_score_norm"],
        "commercial_score": row["commercial_score_norm"],
        "livability_score": row["livability_score_norm"]
    }
    for data_name, value in data_to_value.items():
        G.add_edge(data_name, ward, type="influences", weight=round(float(value), 4))

# -------------------
# 边：User -> Decision
# -------------------
for _, row in user_decision_df.iterrows():
    G.add_edge(row["user"], row["decision"], type="makes")

# -------------------
# anomaly scoring
# -------------------
user_to_type = dict(zip(user_profile_df["user"], user_profile_df["type"]))
type_counts = Counter(user_to_type.values())

type_data_pairs = []
for _, row in user_data_df.iterrows():
    user = row["user"]
    data = row["data"]
    user_type = user_to_type[user]
    type_data_pairs.append((user_type, data))

pair_counter = Counter(type_data_pairs)
degree_centrality = nx.degree_centrality(G)

data_centrality = {}
for node, score in degree_centrality.items():
    if G.nodes[node].get("type") == "data":
        data_centrality[node] = score

results = []
for (user_type, data), freq in pair_counter.items():
    rarity = 1 / freq
    importance = data_centrality.get(data, 0)
    type_size = type_counts[user_type]
    type_weight = math.log(1 + type_size)
    anomaly_score = rarity * importance * type_weight
    results.append({
        "type": user_type,
        "data": data,
        "frequency": freq,
        "score": anomaly_score
    })

results = sorted(results, key=lambda x: x["score"], reverse=True)

# 取高分 anomaly
score_threshold = 0.45
high_anomalies = {
    (r["type"], r["data"]) for r in results if r["score"] >= score_threshold
}

# 对应到 user -> data 边
anomalous_user_data_edges = set()
for _, row in user_data_df.iterrows():
    user = row["user"]
    data = row["data"]
    user_type = user_to_type[user]
    if (user_type, data) in high_anomalies:
        anomalous_user_data_edges.add((user, data))

# -------------------
# 可视化
# -------------------
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
        weight = attrs.get("weight", 0)
        title = f"{edge_type} | weight={weight}"
        width = 1 + 4 * weight

    elif edge_type == "prefers":
        width = 2

    elif edge_type == "makes":
        width = 2

    elif edge_type == "uses" and (source, target) in anomalous_user_data_edges:
        user_type = user_to_type[source]
        pair = (user_type, target)
        pair_score = next(r["score"] for r in results if r["type"] == user_type and r["data"] == target)
        color = "red"
        width = 5
        title = f"uses | ANOMALY: {pair} | score={pair_score:.3f}"

    net.add_edge(source, target, title=title, width=width, color=color)

net.write_html("network_real_v2_anomaly.html")
print("Anomaly-highlighted graph created! Open network_real_v2_anomaly.html")

print("\n=== Highlighted Anomalies ===")
for pair in sorted(high_anomalies):
    print(pair)