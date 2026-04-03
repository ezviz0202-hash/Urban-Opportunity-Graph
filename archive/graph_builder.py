import networkx as nx
from pyvis.network import Network

# 创建有向图
G = nx.DiGraph()

# -------------------
# 添加节点
# -------------------

# Users
users = ["UserA", "UserB", "UserC", "UserD", "UserE"]

# Data
data_nodes = ["rent", "crime", "school", "transport", "population"]

# Decisions
decisions = ["Shinjuku", "Shibuya", "Setagaya", "Koto"]

# 添加节点属性
for u in users:
    G.add_node(u, type="user")

for d in data_nodes:
    G.add_node(d, type="data")

for dec in decisions:
    G.add_node(dec, type="decision")

# -------------------
# 添加边（更强的共创结构）
# -------------------

# User -> Data (uses)
G.add_edge("UserA", "school", type="uses")
G.add_edge("UserA", "crime", type="uses")

G.add_edge("UserB", "school", type="uses")
G.add_edge("UserB", "crime", type="uses")
G.add_edge("UserB", "transport", type="uses")

G.add_edge("UserC", "transport", type="uses")
G.add_edge("UserC", "population", type="uses")
G.add_edge("UserC", "crime", type="uses")

G.add_edge("UserD", "rent", type="uses")
G.add_edge("UserD", "school", type="uses")

G.add_edge("UserE", "transport", type="uses")
G.add_edge("UserE", "population", type="uses")

# Data -> Decision (influences)
G.add_edge("school", "Shinjuku", type="influences")
G.add_edge("school", "Setagaya", type="influences")

G.add_edge("crime", "Shinjuku", type="influences")
G.add_edge("crime", "Koto", type="influences")

G.add_edge("transport", "Shinjuku", type="influences")
G.add_edge("transport", "Shibuya", type="influences")
G.add_edge("transport", "Koto", type="influences")

G.add_edge("population", "Shibuya", type="influences")
G.add_edge("population", "Koto", type="influences")

G.add_edge("rent", "Setagaya", type="influences")

# User -> Decision (makes)
G.add_edge("UserA", "Shinjuku", type="makes")
G.add_edge("UserB", "Shinjuku", type="makes")
G.add_edge("UserC", "Shibuya", type="makes")
G.add_edge("UserD", "Setagaya", type="makes")
G.add_edge("UserE", "Koto", type="makes")

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

net.write_html("network.html")
print("Graph created! Open network.html")