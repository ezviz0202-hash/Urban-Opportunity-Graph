import pandas as pd
import networkx as nx
from pyvis.network import Network

# -------------------
# 读取数据
# -------------------
user_profile_df = pd.read_csv("data/user_profile.csv")
user_data_df = pd.read_csv("data/user_data_real.csv")
user_decision_df = pd.read_csv("data/user_decision_real.csv")
tokyo_df = pd.read_csv("data/tokyo_ward_data_starter.csv")

# 创建图
G = nx.DiGraph()

# -------------------
# 标准化 / 标签函数
# -------------------
def label_rent_low(x):
    return "high" if x > 100000 else "low"

def label_rent_high(x):
    return "high" if x > 130000 else "low"

def label_density(x):
    return "high" if x > 15000 else "low"

def label_school(x):
    return "high" if x > 30 else "low"

# 生成标签
tokyo_df = tokyo_df.copy()
tokyo_df["rent_low_label"] = tokyo_df["avg_1k_rent_low_jpy"].apply(label_rent_low)
tokyo_df["rent_high_label"] = tokyo_df["avg_1k_rent_high_jpy"].apply(label_rent_high)
tokyo_df["population_density_label"] = tokyo_df["population_density_per_km2"].apply(label_density)
tokyo_df["school_count_label"] = tokyo_df["elementary_schools"].apply(label_school)

# -------------------
# 节点集合
# -------------------
users = sorted(set(user_profile_df["user"]))
user_types = sorted(set(user_profile_df["type"]))
data_nodes = sorted(set(user_data_df["data"]))
decisions = sorted(set(tokyo_df["ward"]))

# -------------------
# 添加节点
# -------------------
for u in users:
    G.add_node(u, type="user")

for t in user_types:
    G.add_node(t, type="user_type")

for d in data_nodes:
    G.add_node(d, type="data")

for dec in decisions:
    G.add_node(dec, type="decision")

# -------------------
# 添加边：User -> Type
# -------------------
for _, row in user_profile_df.iterrows():
    G.add_edge(row["user"], row["type"], type="has_type")

# -------------------
# 添加边：Type -> Data（偏好）
# -------------------
type_to_data = {
    "student": ["school_count", "population_density", "rent_high"],
    "worker": ["rent_low", "population_density", "school_count"],
    "family": ["school_count", "rent_low", "population_density"]
}

for t, ds in type_to_data.items():
    for d in ds:
        if d in data_nodes:
            G.add_edge(t, d, type="prefers")

# -------------------
# 添加边：User -> Data
# -------------------
for _, row in user_data_df.iterrows():
    G.add_edge(row["user"], row["data"], type="uses")

# -------------------
# 添加边：Data -> Decision（由东京真实数据生成）
# -------------------
for _, row in tokyo_df.iterrows():
    ward = row["ward"]

    # rent_low
    G.add_edge("rent_low", ward, type="influences", weight=row["rent_low_label"])

    # rent_high
    if "rent_high" in data_nodes:
        G.add_edge("rent_high", ward, type="influences", weight=row["rent_high_label"])

    # population_density
    G.add_edge("population_density", ward, type="influences", weight=row["population_density_label"])

    # school_count
    G.add_edge("school_count", ward, type="influences", weight=row["school_count_label"])

# -------------------
# 添加边：User -> Decision
# -------------------
for _, row in user_decision_df.iterrows():
    G.add_edge(row["user"], row["decision"], type="makes")

# -------------------
# 可视化
# -------------------
net = Network(height="800px", width="100%", directed=True)

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

    if edge_type == "influences" and "weight" in attrs:
        title = f"{edge_type} ({attrs['weight']})"

    net.add_edge(source, target, title=title)

net.write_html("network_real.html")
print("Real-data graph created! Open network_real.html")