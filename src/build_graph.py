import pandas as pd
import networkx as nx
from pyvis.network import Network


def normalize(series: pd.Series, reverse: bool = False) -> pd.Series:
    values = series.astype(float)
    lo, hi = values.min(), values.max()
    if hi == lo:
        scaled = pd.Series([0.5] * len(values), index=values.index)
    else:
        scaled = (values - lo) / (hi - lo)
    return 1 - scaled if reverse else scaled


user_profile_df = pd.read_csv("data/user_profile.csv")
user_data_df = pd.read_csv("data/user_data_real.csv")
user_decision_df = pd.read_csv("data/user_decision_real.csv")
tokyo_df = pd.read_csv("data/tokyo_ward_data_extended.csv").copy()

tokyo_df["rent_low_score"] = normalize(tokyo_df["avg_1k_rent_low_jpy"], reverse=True)
tokyo_df["rent_high_score"] = normalize(tokyo_df["avg_1k_rent_high_jpy"], reverse=True)
tokyo_df["population_density_score"] = normalize(tokyo_df["population_density_per_km2"])
tokyo_df["school_count_score"] = normalize(tokyo_df["elementary_schools"])
tokyo_df["crime_rate_score"] = normalize(tokyo_df["crime_rate"], reverse=True)
tokyo_df["transport_score_norm"] = normalize(tokyo_df["transport_score"])
tokyo_df["commercial_score_norm"] = normalize(tokyo_df["commercial_score"])
tokyo_df["livability_score_norm"] = normalize(tokyo_df["livability_score"])

G = nx.DiGraph()

users = sorted(set(user_profile_df["user"]))
user_types = sorted(set(user_profile_df["type"]))
data_nodes = sorted({
    "rent_low","rent_high","population_density","school_count",
    "crime_rate","transport_score","commercial_score","livability_score"
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

for _, row in user_profile_df.iterrows():
    G.add_edge(row["user"], row["type"], type="has_type")

type_to_data = {
    "student": ["school_count","population_density","rent_high","transport_score","commercial_score"],
    "worker": ["rent_low","population_density","transport_score","commercial_score","crime_rate"],
    "family": ["school_count","rent_low","livability_score","crime_rate","population_density"]
}

for t, ds in type_to_data.items():
    for d in ds:
        G.add_edge(t, d, type="prefers")

for _, row in user_data_df.iterrows():
    G.add_edge(row["user"], row["data"], type="uses")

for _, row in tokyo_df.iterrows():
    ward = row["ward"]
    feature_map = {
        "rent_low": row["rent_low_score"],
        "rent_high": row["rent_high_score"],
        "population_density": row["population_density_score"],
        "school_count": row["school_count_score"],
        "crime_rate": row["crime_rate_score"],
        "transport_score": row["transport_score_norm"],
        "commercial_score": row["commercial_score_norm"],
        "livability_score": row["livability_score_norm"]
    }
    for data_name, value in feature_map.items():
        G.add_edge(data_name, ward, type="influences", weight=round(float(value), 4))

for _, row in user_decision_df.iterrows():
    G.add_edge(row["user"], row["decision"], type="makes")

net = Network(height="850px", width="100%", directed=True)

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
    if edge_type == "influences":
        weight = attrs.get("weight", 0)
        title = f"{edge_type} | weight={weight}"
        width = 1 + 4 * weight
    elif edge_type in {"prefers","makes"}:
        width = 2
    net.add_edge(source, target, title=title, width=width)

net.write_html("network_real_v2.html")
print("Graph generated: network_real_v2.html")
