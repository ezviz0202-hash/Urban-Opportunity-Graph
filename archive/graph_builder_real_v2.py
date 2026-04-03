import pandas as pd
import networkx as nx
from pyvis.network import Network

# -------------------
# 读取数据
# -------------------
user_profile_df = pd.read_csv("data/user_profile.csv")
user_data_df = pd.read_csv("data/user_data_real.csv")
user_decision_df = pd.read_csv("data/user_decision_real.csv")
tokyo_df = pd.read_csv("data/tokyo_ward_data_extended.csv")

# 创建图
G = nx.DiGraph()

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
# 连续特征标准化
# reverse=True 表示越低越好
# -------------------
tokyo_df = tokyo_df.copy()

tokyo_df["rent_low_score"] = normalize(tokyo_df["avg_1k_rent_low_jpy"], reverse=True)
tokyo_df["rent_high_score"] = normalize(tokyo_df["avg_1k_rent_high_jpy"], reverse=True)
tokyo_df["population_density_score"] = normalize(tokyo_df["population_density_per_km2"], reverse=False)
tokyo_df["school_count_score"] = normalize(tokyo_df["elementary_schools"], reverse=False)

# crime 越低越好
tokyo_df["crime_rate_score"] = normalize(tokyo_df["crime_rate"], reverse=True)

# transport / commercial / livability 越高越好
tokyo_df["transport_score_norm"] = normalize(tokyo_df["transport_score"], reverse=False)
tokyo_df["commercial_score_norm"] = normalize(tokyo_df["commercial_score"], reverse=False)
tokyo_df["livability_score_norm"] = normalize(tokyo_df["livability_score"], reverse=False)

# -------------------
# 节点集合
# -------------------
users = sorted(set(user_profile_df["user"]))
user_types = sorted(set(user_profile_df["type"]))
data_nodes = sorted({
    "rent_low",
    "rent_high",
    "population_density",
    "school_count",
    "crime_rate",
    "transport_score",
    "commercial_score",
    "livability_score"
})
decisions = sorted(set(tokyo_df["ward"]))

# -------------------
# 添加节点
# -------------------
for u in users:
    G.add_node(u, type="user")

for t in user_types:
    G.add_node(t, type="user_type")

for d in data_nodes:
    G.add_node(d, type="data")

for dec in decisions:
    G.add_node(dec, type="decision")

# -------------------
# 添加边：User -> Type
# -------------------
for _, row in user_profile_df.iterrows():
    G.add_edge(row["user"], row["type"], type="has_type")

# -------------------
# 添加边：Type -> Data（扩展版偏好）
# 注意：这是“类型偏好模板”，真实 user->data 仍然来自 user_data_real.csv
# -------------------
type_to_data = {
    "student": [
        "school_count",
        "population_density",
        "rent_high",
        "commercial_score",
        "transport_score"
    ],
    "worker": [
        "rent_low",
        "population_density",
        "transport_score",
        "commercial_score",
        "crime_rate"
    ],
    "family": [
        "school_count",
        "rent_low",
        "livability_score",
        "crime_rate",
        "population_density"
    ]
}

for t, ds in type_to_data.items():
    for d in ds:
        G.add_edge(t, d, type="prefers")

# -------------------
# 添加边：User -> Data
# 这里只接真实行为输出
# 如果 user_data_real 里没有新维度，图里这些新维度暂时只会通过 type 偏好出现
# -------------------
for _, row in user_data_df.iterrows():
    G.add_edge(row["user"], row["data"], type="uses")

# -------------------
# 添加边：Data -> Decision（连续权重）
# -------------------
for _, row in tokyo_df.iterrows():
    ward = row["ward"]

    data_to_value = {
        "rent_low": row["rent_low_score"],
        "rent_high": row["rent_high_score"],
        "population_density": row["population_density_score"],
        "school_count": row["school_count_score"],
        "crime_rate": row["crime_rate_score"],
        "transport_score": row["transport_score_norm"],
        "commercial_score": row["commercial_score_norm"],
        "livability_score": row["livability_score_norm"]
    }

    for data_name, value in data_to_value.items():
        G.add_edge(
            data_name,
            ward,
            type="influences",
            weight=round(float(value), 4)
        )

# -------------------
# 添加边：User -> Decision
# -------------------
for _, row in user_decision_df.iterrows():
    G.add_edge(row["user"], row["decision"], type="makes")

# -------------------
# 可视化
# -------------------
net = Network(height="850px", width="100%", directed=True)

for node, attrs in G.nodes(data=True):
    node_type = attrs["type"]

    if node_type == "user":
        color = "red"
        title = f"{node} (User)"
    elif node_type == "user_type":
        color = "orange"
        title = f"{node} (User Type)"
    elif node_type == "data":
        color = "blue"
        title = f"{node} (Data)"
    else:
        color = "green"
        title = f"{node} (Decision / Ward)"

    net.add_node(node, color=color, title=title)

for source, target, attrs in G.edges(data=True):
    edge_type = attrs["type"]

    title = edge_type
    width = 1

    if edge_type == "influences":
        weight = attrs.get("weight", 0)
        title = f"{edge_type} | weight={weight}"
        width = 1 + 4 * weight

    elif edge_type == "prefers":
        width = 2

    elif edge_type == "makes":
        width = 2

    net.add_edge(source, target, title=title, width=width)

net.write_html("network_real_v2.html")
print("Extended real-data graph created! Open network_real_v2.html")

# -------------------
# 打印检查信息
# -------------------
print("\n=== Extended Data -> Decision Weights ===")
for _, row in tokyo_df.iterrows():
    print(
        f"{row['ward']}: "
        f"rent_low={row['rent_low_score']:.3f}, "
        f"rent_high={row['rent_high_score']:.3f}, "
        f"density={row['population_density_score']:.3f}, "
        f"school={row['school_count_score']:.3f}, "
        f"crime={row['crime_rate_score']:.3f}, "
        f"transport={row['transport_score_norm']:.3f}, "
        f"commercial={row['commercial_score_norm']:.3f}, "
        f"livability={row['livability_score_norm']:.3f}"
    )