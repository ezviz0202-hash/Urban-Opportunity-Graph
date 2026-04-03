from collections import Counter
import math
import pandas as pd
import networkx as nx
from build_graph import G

user_to_type = {}
for source, target, attrs in G.edges(data=True):
    if attrs["type"] == "has_type":
        user_to_type[source] = target

type_counts = Counter(user_to_type.values())

type_data_pairs = []
for source, target, attrs in G.edges(data=True):
    if attrs["type"] == "uses" and source in user_to_type:
        type_data_pairs.append((user_to_type[source], target))

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
            "rarity": rarity,
            "importance": importance,
            "type_size": type_size,
            "type_weight": type_weight,
            "score": anomaly_score,
        }
    )

results = sorted(results, key=lambda item: item["score"], reverse=True)

print("=== Real Anomaly Scoring Results ===")
for row in results:
    print(
        f"({row['type']}, {row['data']}) | "
        f"freq={row['frequency']} | "
        f"rarity={row['rarity']:.3f} | "
        f"importance={row['importance']:.3f} | "
        f"type_size={row['type_size']} | "
        f"type_weight={row['type_weight']:.3f} | "
        f"score={row['score']:.3f}"
    )

print("\n=== Top Potential Opportunities ===")
for row in results[:10]:
    print(f"{row['type']} -> {row['data']} | score={row['score']:.3f}")

print("\n=== Low-Frequency Real Anomalies ===")
for row in results:
    if row["frequency"] < row["type_size"]:
        print(
            f"{row['type']} -> {row['data']} | "
            f"freq={row['frequency']} / type_size={row['type_size']} | "
            f"score={row['score']:.3f}"
        )

results_df = pd.DataFrame(results)
results_df.to_csv("anomaly_results.csv", index=False)
print("\nSaved anomaly_results.csv")
