import pandas as pd

# 读取数据
profiles = pd.read_csv("data/user_profile.csv")
tokyo = pd.read_csv("data/tokyo_ward_data_starter.csv")

# -------------------
# 把 ward 特征转成 high/low 标签
# -------------------
def label_rent_low(x):
    return "high" if x > 100000 else "low"

def label_rent_high(x):
    return "high" if x > 130000 else "low"

def label_density(x):
    return "high" if x > 15000 else "low"

def label_school(x):
    return "high" if x > 30 else "low"

tokyo = tokyo.copy()
tokyo["rent_low_label"] = tokyo["avg_1k_rent_low_jpy"].apply(label_rent_low)
tokyo["rent_high_label"] = tokyo["avg_1k_rent_high_jpy"].apply(label_rent_high)
tokyo["density_label"] = tokyo["population_density_per_km2"].apply(label_density)
tokyo["school_label"] = tokyo["elementary_schools"].apply(label_school)

# -------------------
# 用户行为生成
# -------------------
user_data_rows = []
user_decision_rows = []

for _, row in profiles.iterrows():
    user = row["user"]
    user_type = row["type"]

    if user_type == "student":
        # 学生关注 school_count / population_density
        user_data_rows.append({"user": user, "data": "school_count"})
        user_data_rows.append({"user": user, "data": "population_density"})

        # 决策规则：优先 school 高 + density 高，且避免 rent_high 过高
        candidates = tokyo[
            (tokyo["school_label"] == "high") &
            (tokyo["density_label"] == "high")
        ]

        if len(candidates) == 0:
            candidates = tokyo

        # 学生倾向较低高端租金
        chosen = candidates.sort_values(
            by=["avg_1k_rent_high_jpy", "elementary_schools"],
            ascending=[True, False]
        ).iloc[0]

    elif user_type == "worker":
        # worker 关注 rent_low / population_density
        user_data_rows.append({"user": user, "data": "rent_low"})
        user_data_rows.append({"user": user, "data": "population_density"})

        # 决策规则：优先 rent_low 低 + density 高
        candidates = tokyo[
            (tokyo["rent_low_label"] == "low") &
            (tokyo["density_label"] == "high")
        ]

        if len(candidates) == 0:
            candidates = tokyo

        chosen = candidates.sort_values(
            by=["avg_1k_rent_low_jpy", "population_density_per_km2"],
            ascending=[True, False]
        ).iloc[0]

    elif user_type == "family":
        # family 关注 school_count / rent_low
        user_data_rows.append({"user": user, "data": "school_count"})
        user_data_rows.append({"user": user, "data": "rent_low"})

        # 决策规则：优先 school 高 + rent_low 低 + density 低
        candidates = tokyo[
            (tokyo["school_label"] == "high") &
            (tokyo["rent_low_label"] == "low")
        ]

        if len(candidates) == 0:
            candidates = tokyo

        chosen = candidates.sort_values(
            by=["population_density_per_km2", "avg_1k_rent_low_jpy", "elementary_schools"],
            ascending=[True, True, False]
        ).iloc[0]

    else:
        continue

    user_decision_rows.append({"user": user, "decision": chosen["ward"]})

# 转成 DataFrame
user_data_real = pd.DataFrame(user_data_rows)
user_decision_real = pd.DataFrame(user_decision_rows)

# 保存
user_data_real.to_csv("data/user_data_real.csv", index=False)
user_decision_real.to_csv("data/user_decision_real.csv", index=False)

print("Generated:")
print("- data/user_data_real.csv")
print("- data/user_decision_real.csv")

print("\n=== user_data_real preview ===")
print(user_data_real.head(20))

print("\n=== user_decision_real preview ===")
print(user_decision_real)