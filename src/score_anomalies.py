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
    rarity      = 1 / freq
    importance  = data_centrality.get(data_name, 0.0)
    type_size   = type_counts[user_type]
    type_weight = math.log(1 + type_size)
    anomaly_score = rarity * importance * type_weight

    results.append({
        "type":        user_type,
        "data":        data_name,
        "frequency":   freq,
        "rarity":      rarity,
        "importance":  importance,
        "type_size":   type_size,
        "type_weight": type_weight,
        "score":       anomaly_score,
    })

results = sorted(results, key=lambda x: x["score"], reverse=True)

print("=== Standard Anomaly Scoring ===")
for row in results:
    print(
        f"({row['type']}, {row['data']}) | "
        f"freq={row['frequency']} | "
        f"rarity={row['rarity']:.3f} | "
        f"importance={row['importance']:.3f} | "
        f"score={row['score']:.3f}"
    )

print("\n=== Top Opportunity Signals ===")
for row in results[:10]:
    print(f"  {row['type']} -> {row['data']} | score={row['score']:.3f}")

print("\n=== Low-Frequency Anomalies ===")
for row in results:
    if row["frequency"] < row["type_size"]:
        print(
            f"  {row['type']} -> {row['data']} | "
            f"freq={row['frequency']} / type_size={row['type_size']} | "
            f"score={row['score']:.3f}"
        )

results_df = pd.DataFrame(results)
results_df.to_csv("outputs/anomaly_results.csv", index=False)
print("\nSaved outputs/anomaly_results.csv")

# finds cases where a user type chose a ward that scores poorly on their
# primary concern -- these are the counter-intuitive patterns

user_decision_df = pd.read_csv("data/user_decision_real.csv")
tokyo_df         = pd.read_csv("data/tokyo_ward_data_extended.csv")

def _norm(series, reverse=False):
    lo, hi = series.min(), series.max()
    if hi == lo:
        return pd.Series([0.5] * len(series), index=series.index)
    s = (series - lo) / (hi - lo)
    return 1 - s if reverse else s

tokyo = tokyo_df.copy()
tokyo["school_count_score"]    = _norm(tokyo["elementary_schools"])
tokyo["crime_rate_score"]      = _norm(tokyo["crime_rate"], reverse=True)
tokyo["commercial_score_norm"] = _norm(tokyo["commercial_score"])
tokyo["transport_score_norm"]  = _norm(tokyo["transport_score"])
tokyo["rent_low_score"]        = _norm(tokyo["avg_1k_rent_low_jpy"], reverse=True)
tokyo["livability_score_norm"] = _norm(tokyo["livability_score"])
ward_scores = tokyo.set_index("ward")

# primary concern per user type
PRIMARY_CONCERN = {
    "student": "school_count_score",
    "worker":  "crime_rate_score",
    "family":  "school_count_score",
}

CONCERN_LABEL = {
    "student": "school accessibility",
    "worker":  "safety (low crime)",
    "family":  "school accessibility",
}

FEATURE_COLS = [
    "commercial_score_norm", "transport_score_norm",
    "rent_low_score", "livability_score_norm",
    "school_count_score", "crime_rate_score",
]

median_scores = {
    concern: ward_scores[concern].median()
    for concern in PRIMARY_CONCERN.values()
    if concern in ward_scores.columns
}

contradictions = []
for _, row in user_decision_df.iterrows():
    user  = row["user"]
    ward  = row["decision"]
    utype = user_to_type.get(user)
    if utype is None or utype not in PRIMARY_CONCERN:
        continue

    concern = PRIMARY_CONCERN[utype]
    if ward not in ward_scores.index or concern not in ward_scores.columns:
        continue

    concern_score = float(ward_scores.loc[ward, concern])
    median        = median_scores[concern]

    if concern_score < median:
        available = [f for f in FEATURE_COLS if f in ward_scores.columns and f != concern]
        attractor = max(available, key=lambda f: ward_scores.loc[ward, f])
        contradictions.append({
            "user":                   user,
            "user_type":              utype,
            "chosen_ward":            ward,
            "primary_concern":        CONCERN_LABEL[utype],
            "concern_score":          round(concern_score, 3),
            "median_concern_score":   round(float(median), 3),
            "hidden_attractor":       attractor,
            "attractor_score":        round(float(ward_scores.loc[ward, attractor]), 3),
            "contradiction_strength": round(float(median - concern_score), 3),
        })

contradictions.sort(key=lambda x: x["contradiction_strength"], reverse=True)

print("\n=== Contradiction Choices (Counter-Intuitive Patterns) ===")
print(f"  {len(contradictions)} cases where users chose wards scoring below")
print(f"  median on their primary concern -- the hidden attractors:\n")
for c in contradictions[:10]:
    print(
        f"  [{c['user_type']}] {c['chosen_ward']}"
        f"  | {c['primary_concern']} score={c['concern_score']}"
        f" (median={c['median_concern_score']})"
        f"  | pulled by: {c['hidden_attractor']} ({c['attractor_score']:.2f})"
    )

pd.DataFrame(contradictions).to_csv("outputs/contradiction_choices.csv", index=False)
print("\nSaved outputs/contradiction_choices.csv")
