import pandas as pd
import random
import math

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
# 生成标准化特征
# -------------------
tokyo = tokyo.copy()
tokyo["rent_low_score"] = normalize(tokyo["avg_1k_rent_low_jpy"], reverse=True)
tokyo["rent_high_score"] = normalize(tokyo["avg_1k_rent_high_jpy"], reverse=True)
tokyo["density_score"] = normalize(tokyo["population_density_per_km2"], reverse=False)
tokyo["density_low_score"] = normalize(tokyo["population_density_per_km2"], reverse=True)
tokyo["school_score"] = normalize(tokyo["elementary_schools"], reverse=False)

# -------------------
# 基础偏好
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
# 生成用户个体化权重
# -------------------
def generate_user_weights(base_pref, noise_scale=0.12):
    noisy = {}
    for k, v in base_pref.items():
        perturbed = v + random.uniform(-noise_scale, noise_scale)
        noisy[k] = max(0.01, perturbed)

    total = sum(noisy.values())
    for k in noisy:
        noisy[k] /= total

    return noisy

# -------------------
# 计算各区分数
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
# Softmax 选择
# -------------------
def softmax(xs, temperature=0.4):
    scaled = [x / temperature for x in xs]
    max_x = max(scaled)
    exps = [math.exp(x - max_x) for x in scaled]
    total = sum(exps)
    return [e / total for e in exps]

def choose_by_softmax(tokyo_df, scores, temperature=0.4):
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

    # 记录关注的数据
    for d in type_to_data[user_type]:
        user_data_rows.append({"user": user, "data": d})

    # 个体化权重
    weights = generate_user_weights(base_preferences[user_type], noise_scale=0.12)

    record = {"user": user, "type": user_type}
    record.update(weights)
    user_weight_rows.append(record)

    # 打分并 softmax 选区
    scores = score_wards(tokyo, weights)
    chosen = choose_by_softmax(tokyo, scores, temperature=0.45)

    user_decision_rows.append({
        "user": user,
        "decision": chosen["ward"],
        "raw_score": round(chosen["raw_score"], 4),
        "choice_prob": round(chosen["choice_prob"], 4)
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

print("\n=== decision distribution ===")
print(user_decision_real["decision"].value_counts())

print("\n=== user_weights_real preview ===")
print(user_weights)