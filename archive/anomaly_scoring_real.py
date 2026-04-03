from collections import Counter
import math
import networkx as nx
from graph_builder_real_v2 import G

# -------------------
# 提取 user -> type 映射
# -------------------
user_to_type = {}
for source, target, attrs in G.edges(data=True):
    if attrs["type"] == "has_type":
        user_to_type[source] = target

# -------------------
# 统计 type 的用户数量
# -------------------
type_counts = Counter(user_to_type.values())

# -------------------
# 统计 Type-Data 频率（基于真实 user_data_real）
# -------------------
type_data_pairs = []

for source, target, attrs in G.edges(data=True):
    if attrs["type"] == "uses":
        user = source
        data = target
        if user in user_to_type:
            user_type = user_to_type[user]
            type_data_pairs.append((user_type, data))

counter = Counter(type_data_pairs)

# -------------------
# 计算 data 节点 centrality
# -------------------
degree_centrality = nx.degree_centrality(G)

data_centrality = {}
for node, score in degree_centrality.items():
    if G.nodes[node].get("type") == "data":
        data_centrality[node] = score

# -------------------
# anomaly scoring
# score = rarity × importance × type_weight
# rarity = 1 / frequency
# importance = degree centrality of data node
# type_weight = log(1 + number of users in this type)
# -------------------
results = []

for (user_type, data), freq in counter.items():
    rarity = 1 / freq
    importance = data_centrality.get(data, 0)

    type_size = type_counts[user_type]
    type_weight = math.log(1 + type_size)

    anomaly_score = rarity * importance * type_weight

    results.append({
        "type": user_type,
        "data": data,
        "frequency": freq,
        "rarity": rarity,
        "importance": importance,
        "type_size": type_size,
        "type_weight": type_weight,
        "score": anomaly_score
    })

results = sorted(results, key=lambda x: x["score"], reverse=True)

# -------------------
# 输出结果
# -------------------
print("=== Real Anomaly Scoring Results ===")
for r in results:
    print(
        f"({r['type']}, {r['data']}) | "
        f"freq={r['frequency']} | "
        f"rarity={r['rarity']:.3f} | "
        f"importance={r['importance']:.3f} | "
        f"type_size={r['type_size']} | "
        f"type_weight={r['type_weight']:.3f} | "
        f"score={r['score']:.3f}"
    )

print("\n=== Top Potential Opportunities ===")
for r in results[:10]:
    print(f"{r['type']} -> {r['data']} | score={r['score']:.3f}")

# -------------------
# 只列真正的低频异常（频次低于该类型主流）
# -------------------
print("\n=== Low-Frequency Real Anomalies ===")
for r in results:
    if r["frequency"] < r["type_size"]:
        print(f"{r['type']} -> {r['data']} | freq={r['frequency']} / type_size={r['type_size']} | score={r['score']:.3f}")
import pandas as pd

results_df = pd.DataFrame(results)
results_df.to_csv("anomaly_results.csv", index=False)
print("\nSaved anomaly_results.csv")