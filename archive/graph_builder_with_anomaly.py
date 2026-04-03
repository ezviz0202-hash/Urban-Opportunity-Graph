import pandas as pd
import networkx as nx
from pyvis.network import Network
from collections import Counter

# -------------------
# 读取 CSV
# -------------------
user_data_df = pd.read_csv("data/user_data.csv")
data_decision_df = pd.read_csv("data/data_decision.csv")
user_decision_df = pd.read_csv("data/user_decision.csv")
user_profile_df = pd.read_csv("data/user_profile.csv")

# 创建图
G = nx.DiGraph()

# -------------------
# 节点集合
# -------------------
users = sorted(set(user_data_df["user"]).union(set(user_decision_df["user"])))
data_nodes = sorted(set(user_data_df["data"]).union(set(data_decision_df["data"])))
decisions = sorted(set(data_decision_df["decision"]).union(set(user_decision_df["decision"])))
user_types = sorted(set(user_profile_df["type"]))

# 建立 user -> type 映射
user_to_type = dict(zip(user_profile_df["user"], user_profile_df["type"]))

# -------------------
# 添加节点
# -------------------
for u in users:
    G.add_node(u, type="user")

for d in data_nodes:
    G.add_node(d, type="data")

for dec in decisions:
    G.add_node(dec, type="decision")

for t in user_types:
    G.add_node(t, type="user_type")

# -------------------
# 添加边
# -------------------

# User → Type
for _, row in user_profile_df.iterrows():
    G.add_edge(row["user"], row["type"], type="has_type")

# Type → Data (Preference Modeling)
G.add_edge("student", "school", type="prefers")
G.add_edge("student", "crime", type="prefers")

G.add_edge("worker", "transport", type="prefers")
G.add_edge("worker", "population", type="prefers")

G.add_edge("family", "rent", type="prefers")
G.add_edge("family", "school", type="prefers")

# User → Data
for _, row in user_data_df.iterrows():
    G.add_edge(row["user"], row["data"], type="uses")

# Data → Decision
for _, row in data_decision_df.iterrows():
    G.add_edge(row["data"], row["decision"], type="influences")

# User → Decision
for _, row in user_decision_df.iterrows():
    G.add_edge(row["user"], row["decision"], type="makes")

# -------------------
# anomaly 检测：Type-Data 低频组合
# -------------------
type_data_pairs = []

for _, row in user_data_df.iterrows():
    user = row["user"]
    data = row["data"]
    user_type = user_to_type[user]
    type_data_pairs.append((user_type, data))

counter = Counter(type_data_pairs)

# 出现次数 <= 1 视为 anomaly
anomalies = {pair for pair, count in counter.items() if count <= 1}

print("=== Detected Anomalies ===")
for pair in sorted(anomalies):
    print(pair)

# 找到需要高亮的 user -> data 边
anomalous_user_data_edges = set()
for _, row in user_data_df.iterrows():
    user = row["user"]
    data = row["data"]
    user_type = user_to_type[user]
    if (user_type, data) in anomalies:
        anomalous_user_data_edges.add((user, data))

# -------------------
# 可视化
# -------------------
net = Network(height="750px", width="100%", directed=True)

# 添加节点
for node, attrs in G.nodes(data=True):
    if attrs["type"] == "user":
        net.add_node(node, color="red", title=f"{node} (User)")
    elif attrs["type"] == "data":
        net.add_node(node, color="blue", title=f"{node} (Data)")
    elif attrs["type"] == "decision":
        net.add_node(node, color="green", title=f"{node} (Decision)")
    else:
        net.add_node(node, color="orange", title=f"{node} (User Type)")

# 添加边，并高亮 anomaly
for source, target, attrs in G.edges(data=True):
    edge_type = attrs["type"]

    # 默认样式
    edge_color = None
    edge_width = 1
    edge_title = edge_type

    # 高亮 anomalous user -> data edges
    if edge_type == "uses" and (source, target) in anomalous_user_data_edges:
        edge_color = "red"
        edge_width = 4
        user_type = user_to_type[source]
        edge_title = f"{edge_type} | ANOMALY: ({user_type}, {target})"

    net.add_edge(source, target, title=edge_title, color=edge_color, width=edge_width)

net.write_html("network_with_anomaly.html")
print("Graph with anomaly visualization created!")