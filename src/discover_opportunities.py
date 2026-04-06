
import math
from collections import Counter, defaultdict
import pandas as pd
import networkx as nx
from build_graph import G

user_profile_df  = pd.read_csv("data/user_profile.csv")
user_data_df     = pd.read_csv("data/user_data_real.csv")
user_decision_df = pd.read_csv("data/user_decision_real.csv")
user_weights_df  = pd.read_csv("data/user_weights_real.csv")
tokyo_df         = pd.read_csv("data/tokyo_ward_data_extended.csv")

user_to_type = dict(zip(user_profile_df["user"], user_profile_df["type"]))

PRIMARY_CONCERN = {
    "student": "school_count_score",
    "worker":  "crime_rate_score",      # workers want LOW crime -> score = 1-crime
    "family":  "school_count_score",
}

CONCERN_LABEL = {
    "student": "school accessibility",
    "worker":  "safety (low crime)",
    "family":  "school accessibility",
}

def keygraph_bridge_analysis():
   
    # how often each (type, data) pair occurs
    type_data_usage = defaultdict(set)
    for _, row in user_data_df.iterrows():
        ut = user_to_type[row["user"]]
        type_data_usage[row["data"]].add(ut)

    data_freq = Counter()
    for _, row in user_data_df.iterrows():
        data_freq[row["data"]] += 1

    total_usage = sum(data_freq.values())

    between = nx.betweenness_centrality(G, normalized=True)

    results = []
    for data_node, types_connected in type_data_usage.items():
        freq       = data_freq[data_node]
        rarity     = 1 - (freq / total_usage)          # rarer = higher
        n_types    = len(types_connected)               # how many types use it
        centrality = between.get(data_node, 0.0)

      
        bridge_score = rarity * n_types * (1 + centrality)

        results.append({
            "data_node":      data_node,
            "frequency":      freq,
            "rarity":         round(rarity, 4),
            "types_connected": sorted(types_connected),
            "n_types":        n_types,
            "betweenness":    round(centrality, 4),
            "bridge_score":   round(bridge_score, 4),
        })

    return sorted(results, key=lambda x: x["bridge_score"], reverse=True)


def find_contradiction_choices():
  
    def normalize(series, reverse=False):
        lo, hi = series.min(), series.max()
        if hi == lo:
            return pd.Series([0.5] * len(series), index=series.index)
        s = (series - lo) / (hi - lo)
        return 1 - s if reverse else s

    tokyo = tokyo_df.copy()
    tokyo["school_count_score"]    = normalize(tokyo["elementary_schools"])
    tokyo["crime_rate_score"]      = normalize(tokyo["crime_rate"], reverse=True)
    tokyo["commercial_score_norm"] = normalize(tokyo["commercial_score"])
    tokyo["transport_score_norm"]  = normalize(tokyo["transport_score"])
    tokyo["rent_low_score"]        = normalize(tokyo["avg_1k_rent_low_jpy"], reverse=True)
    tokyo["livability_score_norm"] = normalize(tokyo["livability_score"])

    ward_scores = tokyo.set_index("ward")

    contradictions = []
    median_scores = {
        concern: ward_scores[concern].median()
        for concern in PRIMARY_CONCERN.values()
    }

    for _, row in user_decision_df.iterrows():
        user  = row["user"]
        ward  = row["decision"]
        utype = user_to_type[user]

        concern = PRIMARY_CONCERN[utype]
        if ward not in ward_scores.index:
            continue

        ward_score_on_concern = ward_scores.loc[ward, concern]
        median               = median_scores[concern]

      
        if ward_score_on_concern < median:
            # what did pull them there? find the highest-scoring feature
            feature_cols = [
                "commercial_score_norm", "transport_score_norm",
                "rent_low_score", "livability_score_norm",
                "school_count_score", "crime_rate_score",
            ]
            scores_dict = {f: ward_scores.loc[ward, f] for f in feature_cols if f in ward_scores.columns}
            attractor = max(scores_dict, key=scores_dict.get)
            attractor_score = scores_dict[attractor]

            contradictions.append({
                "user":                   user,
                "user_type":              utype,
                "chosen_ward":            ward,
                "primary_concern":        concern,
                "concern_score":          round(float(ward_score_on_concern), 3),
                "median_concern_score":   round(float(median), 3),
                "hidden_attractor":       attractor,
                "attractor_score":        round(float(attractor_score), 3),
                "contradiction_strength": round(float(median - ward_score_on_concern), 3),
            })

    return sorted(contradictions, key=lambda x: x["contradiction_strength"], reverse=True)


def summarise_cross_type_patterns(contradictions):
   
    pair_counts  = Counter()
    pair_wards   = defaultdict(list)

    for c in contradictions:
        key = (c["user_type"], c["hidden_attractor"])
        pair_counts[key] += 1
        pair_wards[key].append(c["chosen_ward"])

    summary = []
    for (utype, attractor), count in pair_counts.most_common():
        summary.append({
            "user_type":       utype,
            "hidden_attractor": attractor,
            "occurrence":       count,
            "example_wards":    list(set(pair_wards[(utype, attractor)]))[:3],
        })
    return summary

if __name__ == "__main__":

    print("=" * 60)
    print("  Opportunity Discovery -- KeyGraph-Inspired Analysis")
    print("=" * 60)

    print("\n[1] KeyGraph Bridge Nodes (hidden opportunity data features)\n")
    bridges = keygraph_bridge_analysis()
    for b in bridges[:8]:
        print(
            f"  {b['data_node']:<25} "
            f"bridge_score={b['bridge_score']:.4f}  "
            f"types={b['types_connected']}  "
            f"freq={b['frequency']}"
        )

    print("\n[2] Contradiction Choices (counter-intuitive ward selections)\n")
    contradictions = find_contradiction_choices()
    print(f"  Total contradiction cases found: {len(contradictions)}")
    print(f"  (users who chose a ward scoring below median on their primary concern)\n")
    for c in contradictions[:8]:
        print(
            f"  [{c['user_type']}] {c['user']} chose {c['chosen_ward']}"
            f"  |  {CONCERN_LABEL[c['user_type']]} score={c['concern_score']} "
            f"(median={c['median_concern_score']})"
            f"  |  hidden attractor: {c['hidden_attractor']} ({c['attractor_score']:.2f})"
        )

    print("\n[3] Systematic Cross-Type Opportunity Patterns\n")
    summary = summarise_cross_type_patterns(contradictions)
    for s in summary:
        print(
            f"  [{s['user_type']}] drawn to [{s['hidden_attractor']}] "
            f"despite primary concern  |  "
            f"occurs {s['occurrence']} times  |  "
            f"example wards: {s['example_wards']}"
        )

    pd.DataFrame(bridges).to_csv("outputs/bridge_nodes.csv", index=False)
    pd.DataFrame(contradictions).to_csv("outputs/contradiction_choices.csv", index=False)
    pd.DataFrame(summary).to_csv("outputs/cross_type_patterns.csv", index=False)
    print("\nSaved: bridge_nodes.csv, contradiction_choices.csv, cross_type_patterns.csv")
