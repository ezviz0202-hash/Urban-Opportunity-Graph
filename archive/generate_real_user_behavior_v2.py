import pandas as pd
import random

# 为了结果可复现
random.seed(42)

# -------------------
# 读取数据
# -------------------
profiles = pd.read_csv("data/user_profile.csv")
tokyo = pd.read_csv("data/tokyo_ward_data_starter.csv")

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
# 为东京各区生成标准化特征
# reverse=True 表示越低越好
# -------------------
tokyo = tokyo.copy()

tokyo["rent_low_score"] = normalize(tokyo["avg_1k_rent_low_jpy"], reverse=True)
tokyo["rent_high_score"] = normalize(tokyo["avg_1k_rent_high_jpy"], reverse=True)
tokyo["density_score"] = normalize(tokyo["population_density_per_km2"], reverse=False)
tokyo["density_low_score"] = normalize(tokyo["population_density_per_km2"], reverse=True)
tokyo["school_score"] = normalize(tokyo["elementary_schools"], reverse=False)

# -------------------
# 为每个 user type 设定基础偏好
# -------------------
base_preferences = {
    "student": {
        "school_score": 0.45,
        "density_score": 0.30,
        "rent_high_score": 0.25
    },
    "worker": {
        "rent_low_score": 0.40,
        "density_score": 0.35,
        "school_score": 0.25
    },
    "family": {
        "school_score": 0.40,
        "rent_low_score": 0.35,
        "density_low_score": 0.25
    }
}

# -------------------
# 用户关注的数据映射
# -------------------
type_to_data = {
    "student": ["school_count", "population_density", "rent_high"],
    "worker": ["rent_low", "population_density", "school_count"],
    "family": ["school_count", "rent_low", "population_density"]
}

# -------------------
# 给单个用户生成扰动后的权重
# -------------------
def generate_user_weights(base_pref, noise_scale=0.08):
    noisy = {}
    for k, v in base_pref.items():
        perturbed = v + random.uniform(-noise_scale, noise_scale)
        noisy[k] = max(0.01, perturbed)

    total = sum(noisy.values())
    for k in noisy:
        noisy[k] /= total

    return noisy

# -------------------
# 根据权重给每个 ward 打分
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
# 从 top-k 中随机选一个
# -------------------
def choose_from_top_k(tokyo_df, scores, k=2):
    temp = tokyo_df.copy()
    temp["score"] = scores
    temp = temp.sort_values(by="score", ascending=False).reset_index(drop=True)

    top_k = temp.head(k)

    # 根据 score 做加权随机
    total_score = top_k["score"].sum()
    if total_score == 0:
        probs = [1 / len(top_k)] * len(top_k)
    else:
        probs = [s / total_score for s in top_k["score"]]

    chosen_idx = random.choices(range(len(top_k)), weights=probs, k=1)[0]
    return top_k.iloc[chosen_idx]

# -------------------
# 生成用户行为
# -------------------
user_data_rows = []
user_decision_rows = []
user_weight_rows = []

for _, row in profiles.iterrows():
    user = row["user"]
    user_type = row["type"]

    if user_type not in base_preferences:
        continue

    # 添加 user -> data 关注关系
    for d in type_to_data[user_type]:
        user_data_rows.append({"user": user, "data": d})

    # 生成个体化权重
    weights = generate_user_weights(base_preferences[user_type], noise_scale=0.10)

    # 保存权重，方便你检查
    record = {"user": user, "type": user_type}
    record.update(weights)
    user_weight_rows.append(record)

    # 打分
    scores = score_wards(tokyo, weights)

    # 从 top-2 中选择
    chosen = choose_from_top_k(tokyo, scores, k=2)

    user_decision_rows.append({
        "user": user,
        "decision": chosen["ward"]
    })

# -------------------
# 保存输出
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

print("\n=== user_decision_real preview ===")
print(user_decision_real)

print("\n=== user_weights_real preview ===")
print(user_weights)