import networkx as nx
from networkx.algorithms import community
from graph_builder import G

# 转为无向图（community算法需要）
G_undirected = G.to_undirected()

communities = community.greedy_modularity_communities(G_undirected)

print("=== Communities ===")
for i, comm in enumerate(communities):
    print(f"Community {i+1}: {list(comm)}")