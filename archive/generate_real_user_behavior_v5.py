import pandas as pd
import random
import math

random.seed(42)

# -------------------
# 读取数据
# -------------------
profiles = pd.read_csv("data/user_profile.csv")
tokyo = pd.read_csv("data/tokyo_ward_data_extended.csv")

# -------------------
# 标准化函数
# -------------------
def normalize(series, reverse=False):
    s = series.astype(float)
    min_val = s.min()
    max_val = s.max()

    if max_val == min_val:
        norm = pd.Series([0.5] * len(s), index=s.index)
    else:
        norm = (s - min_val) / (max_val - min_val)

    if reverse:
        norm = 1 - norm

    return norm

# -------------------
# 构造连续特征分数
# -------------------
tokyo = tokyo.copy()

tokyo["rent_low_score"] = normalize(tokyo["avg_1k_rent_low_jpy"], reverse=True)
tokyo["rent_high_score"] = normalize(tokyo["avg_1k_rent_high_jpy"], reverse=True)
tokyo["population_density_score"] = normalize(tokyo["population_density_per_km2"], reverse=False)
tokyo["school_count_score"] = normalize(tokyo["elementary_schools"], reverse=False)
tokyo["crime_rate_score"] = normalize(tokyo["crime_rate"], reverse=True)
tokyo["transport_score_norm"] = normalize(tokyo["transport_score"], reverse=False)
tokyo["commercial_score_norm"] = normalize(tokyo["commercial_score"], reverse=False)
tokyo["livability_score_norm"] = normalize(tokyo["livability_score"], reverse=False)

# -------------------
# type 的基础偏好（用于 decision scoring）
# -------------------
base_preferences = {
    "student": {
        "school_count_score": 0.28,
        "population_density_score": 0.18,
        "rent_high_score": 0.12,
        "transport_score_norm": 0.20,
        "commercial_score_norm": 0.17,
        "crime_rate_score": 0.05
    },
    "worker": {
        "rent_low_score": 0.20,
        "population_density_score": 0.18,
        "transport_score_norm": 0.28,
        "commercial_score_norm": 0.18,
        "crime_rate_score": 0.10,
        "school_count_score": 0.06
    },
    "family": {
        "school_count_score": 0.24,
        "rent_low_score": 0.18,
        "livability_score_norm": 0.24,
        "crime_rate_score": 0.16,
        "population_density_score": 0.08,
        "transport_score_norm": 0.05,
        "commercial_score_norm": 0.05
    }
}

# -------------------
# 每类用户对 data 的选择概率
# 这里决定 user -> data
# -------------------
type_data_probs = {
    "student": {
        "school_count": 0.90,
        "population_density": 0.65,
        "rent_high": 0.35,
        "transport_score": 0.75,
        "commercial_score": 0.70,
        "crime_rate": 0.20
    },
    "worker": {
        "rent_low": 0.75,
        "population_density": 0.60,
        "transport_score": 0.90,
        "commercial_score": 0.75,
        "crime_rate": 0.45,
        "school_count": 0.20
    },
    "family": {
        "school_count": 0.85,
        "rent_low": 0.75,
        "livability_score": 0.85,
        "crime_rate": 0.65,
        "population_density": 0.30,
        "transport_score": 0.20,
        "commercial_score": 0.20
    }
}

# data 节点到 score 特征的映射
data_to_feature = {
    "rent_low": "rent_low_score",
    "rent_high": "rent_high_score",
    "population_density": "population_density_score",
    "school_count": "school_count_score",
    "crime_rate": "crime_rate_score",
    "transport_score": "transport_score_norm",
    "commercial_score": "commercial_score_norm",
    "livability_score": "livability_score_norm"
}

# -------------------
# 至少选择几个 data
# -------------------
def select_user_data(user_type):
    probs = type_data_probs[user_type]
    selected = []

    for data_name, p in probs.items():
        if random.random() < p:
            selected.append(data_name)

    # 至少保留 2 个，避免信息过少
    if len(selected) < 2:
        ranked = sorted(probs.items(), key=lambda x: x[1], reverse=True)
        for data_name, _ in ranked:
            if data_name not in selected:
                selected.append(data_name)
            if len(selected) >= 2:
                break

    return selected

# -------------------
# 根据选中的 data 构造 decision 权重
# -------------------
def build_decision_weights_from_selected(selected_data, user_type):
    base_pref = base_preferences[user_type]

    feature_weights = {}
    for data_name in selected_data:
        feature = data_to_feature[data_name]
        if feature in base_pref:
            feature_weights[feature] = base_pref[feature]

    if not feature_weights:
        feature_weights = base_pref.copy()

    # 个体扰动
    noisy = {}
    for k, v in feature_weights.items():
        perturbed = v + random.uniform(-0.08, 0.08)
        noisy[k] = max(0.01, perturbed)

    total = sum(noisy.values())
    for k in noisy:
        noisy[k] /= total

    return noisy

# -------------------
# 打分
# -------------------
def score_wards(tokyo_df, weights):
    scores = []
    for _, row in tokyo_df.iterrows():
        score = 0
        for feature, weight in weights.items():
            score += row[feature] * weight
        scores.append(score)
    return scores

# -------------------
# softmax 选择
# -------------------
def softmax(xs, temperature=0.55):
    scaled = [x / temperature for x in xs]
    max_x = max(scaled)
    exps = [math.exp(x - max_x) for x in scaled]
    total = sum(exps)
    return [e / total for e in exps]

def choose_by_softmax(tokyo_df, scores, temperature=0.55):
    probs = softmax(scores, temperature=temperature)
    idx = random.choices(range(len(tokyo_df)), weights=probs, k=1)[0]
    chosen = tokyo_df.iloc[idx].copy()
    chosen["choice_prob"] = probs[idx]
    chosen["raw_score"] = scores[idx]
    return chosen

# -------------------
# 生成行为
# -------------------
user_data_rows = []
user_decision_rows = []
user_weight_rows = []

for _, row in profiles.iterrows():
    user = row["user"]
    user_type = row["type"]

    if user_type not in base_preferences:
        continue

    selected_data = select_user_data(user_type)

    for d in selected_data:
        user_data_rows.append({"user": user, "data": d})

    decision_weights = build_decision_weights_from_selected(selected_data, user_type)

    weight_record = {
        "user": user,
        "type": user_type,
        "selected_data": ",".join(selected_data)
    }
    weight_record.update(decision_weights)
    user_weight_rows.append(weight_record)

    scores = score_wards(tokyo, decision_weights)
    chosen = choose_by_softmax(tokyo, scores, temperature=0.55)

    user_decision_rows.append({
        "user": user,
        "decision": chosen["ward"],
        "raw_score": round(chosen["raw_score"], 4),
        "choice_prob": round(chosen["choice_prob"], 4),
        "selected_data": ",".join(selected_data)
    })

# -------------------
# 保存
# -------------------
user_data_real = pd.DataFrame(user_data_rows)
user_decision_real = pd.DataFrame(user_decision_rows)
user_weights = pd.DataFrame(user_weight_rows)

user_data_real.to_csv("data/user_data_real.csv", index=False)
user_decision_real.to_csv("data/user_decision_real.csv", index=False)
user_weights.to_csv("data/user_weights_real.csv", index=False)

print("Generated:")
print("- data/user_data_real.csv")
print("- data/user_decision_real.csv")
print("- data/user_weights_real.csv")

print("\n=== user_data_real preview ===")
print(user_data_real.head(50))

print("\n=== user_decision_real preview ===")
print(user_decision_real)

print("\n=== decision distribution ===")
print(user_decision_real["decision"].value_counts())

print("\n=== user_weights_real preview ===")
print(user_weights)