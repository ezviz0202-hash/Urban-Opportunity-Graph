import networkx as nx
from graph_builder import G

# 计算 centrality
degree_centrality = nx.degree_centrality(G)
betweenness_centrality = nx.betweenness_centrality(G)

print("=== Degree Centrality ===")
for node, val in degree_centrality.items():
    print(f"{node}: {val:.3f}")

print("\n=== Betweenness Centrality ===")
for node, val in betweenness_centrality.items():
    print(f"{node}: {val:.3f}")