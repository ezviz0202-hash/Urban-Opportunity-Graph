import pandas as pd
import networkx as nx
from pyvis.network import Network

# -------------------
# 读取 CSV
# -------------------
user_data_df = pd.read_csv("data/user_data.csv")
data_decision_df = pd.read_csv("data/data_decision.csv")
user_decision_df = pd.read_csv("data/user_decision.csv")

# 创建有向图
G = nx.DiGraph()

# -------------------
# 自动识别节点
# -------------------
users = sorted(set(user_data_df["user"]).union(set(user_decision_df["user"])))
data_nodes = sorted(set(user_data_df["data"]).union(set(data_decision_df["data"])))
decisions = sorted(set(data_decision_df["decision"]).union(set(user_decision_df["decision"])))

# 添加节点
for u in users:
    G.add_node(u, type="user")

for d in data_nodes:
    G.add_node(d, type="data")

for dec in decisions:
    G.add_node(dec, type="decision")

# -------------------
# 添加边：User -> Data
# -------------------
for _, row in user_data_df.iterrows():
    G.add_edge(row["user"], row["data"], type="uses")

# -------------------
# 添加边：Data -> Decision
# -------------------
for _, row in data_decision_df.iterrows():
    G.add_edge(row["data"], row["decision"], type="influences")

# -------------------
# 添加边：User -> Decision
# -------------------
for _, row in user_decision_df.iterrows():
    G.add_edge(row["user"], row["decision"], type="makes")

# -------------------
# 可视化
# -------------------
net = Network(height="750px", width="100%", directed=True)

for node, attrs in G.nodes(data=True):
    if attrs["type"] == "user":
        net.add_node(node, color="red", title=f"{node} (User)")
    elif attrs["type"] == "data":
        net.add_node(node, color="blue", title=f"{node} (Data)")
    else:
        net.add_node(node, color="green", title=f"{node} (Decision)")

for source, target, attrs in G.edges(data=True):
    net.add_edge(source, target, title=attrs["type"])

net.write_html("network_from_csv.html")
print("Graph created from CSV! Open network_from_csv.html")