import pandas as pd
import random
import math

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
# 真实数据特征标准化
# -------------------
tokyo = tokyo.copy()
tokyo["rent_low_score"] = normalize(tokyo["avg_1k_rent_low_jpy"], reverse=True)
tokyo["rent_high_score"] = normalize(tokyo["avg_1k_rent_high_jpy"], reverse=True)
tokyo["density_score"] = normalize(tokyo["population_density_per_km2"], reverse=False)
tokyo["density_low_score"] = normalize(tokyo["population_density_per_km2"], reverse=True)
tokyo["school_score"] = normalize(tokyo["elementary_schools"], reverse=False)

# -------------------
# 基础偏好（用于 decision scoring）
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
# 每类用户对 data 的选择概率
# 这一步是 v4 的关键
# -------------------
type_data_probs = {
    "student": {
        "school_count": 0.95,
        "population_density": 0.70,
        "rent_high": 0.45
    },
    "worker": {
        "rent_low": 0.90,
        "population_density": 0.80,
        "school_count": 0.40
    },
    "family": {
        "school_count": 0.90,
        "rent_low": 0.85,
        "population_density": 0.35
    }
}

# data 节点到 score 特征的映射
data_to_feature = {
    "school_count": "school_score",
    "population_density": "density_score",
    "rent_high": "rent_high_score",
    "rent_low": "rent_low_score"
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
# 按概率选择 user 使用哪些 data
# 保证至少选1个
# -------------------
def select_user_data(user_type):
    probs = type_data_probs[user_type]
    selected = []

    for data_name, p in probs.items():
        if random.random() < p:
            selected.append(data_name)

    # 防止一个都没选到
    if not selected:
        selected = [max(probs, key=probs.get)]

    return selected

# -------------------
# 根据“用户实际选择的数据”构造 decision 权重
# 只对已选择的数据分配权重
# -------------------
def build_decision_weights_from_selected(selected_data, user_type):
    base_pref = base_preferences[user_type]

    feature_weights = {}
    for data_name in selected_data:
        feature = data_to_feature[data_name]
        if feature in base_pref:
            feature_weights[feature] = base_pref[feature]

    # 如果因为映射问题空了，回退到基础偏好
    if not feature_weights:
        feature_weights = base_pref.copy()

    # 再加一点个体扰动
    noisy = {}
    for k, v in feature_weights.items():
        perturbed = v + random.uniform(-0.10, 0.10)
        noisy[k] = max(0.01, perturbed)

    total = sum(noisy.values())
    for k in noisy:
        noisy[k] /= total

    return noisy

# -------------------
# 给 ward 打分
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
def softmax(xs, temperature=0.45):
    scaled = [x / temperature for x in xs]
    max_x = max(scaled)
    exps = [math.exp(x - max_x) for x in scaled]
    total = sum(exps)
    return [e / total for e in exps]

def choose_by_softmax(tokyo_df, scores, temperature=0.45):
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

    # v4: user 自己选 data
    selected_data = select_user_data(user_type)

    for d in selected_data:
        user_data_rows.append({"user": user, "data": d})

    # 根据“实际选到的数据”构建 decision 权重
    decision_weights = build_decision_weights_from_selected(selected_data, user_type)

    weight_record = {"user": user, "type": user_type}
    weight_record["selected_data"] = ",".join(selected_data)
    weight_record.update(decision_weights)
    user_weight_rows.append(weight_record)

    # decision
    scores = score_wards(tokyo, decision_weights)
    chosen = choose_by_softmax(tokyo, scores, temperature=0.50)

    user_decision_rows.append({
        "user": user,
        "decision": chosen["ward"],
        "raw_score": round(chosen["raw_score"], 4),
        "choice_prob": round(chosen["choice_prob"], 4),
        "selected_data": ",".join(selected_data)
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

print("\n=== user_data_real preview ===")
print(user_data_real.head(30))

print("\n=== user_decision_real preview ===")
print(user_decision_real)

print("\n=== decision distribution ===")
print(user_decision_real["decision"].value_counts())

print("\n=== user_weights_real preview ===")
print(user_weights)