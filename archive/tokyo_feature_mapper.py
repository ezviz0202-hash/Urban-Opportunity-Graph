import pandas as pd
import networkx as nx

# 读取东京数据
df = pd.read_csv("data/tokyo_ward_data_starter.csv")

G = nx.DiGraph()

# -------------------
# 添加 decision 节点（ward）
# -------------------
for _, row in df.iterrows():
    ward = row["ward"]
    G.add_node(ward, type="decision")

# -------------------
# 创建 data nodes（标准化）
# -------------------
data_nodes = [
    "rent_low",
    "rent_high",
    "population_density",
    "school_count"
]

for d in data_nodes:
    G.add_node(d, type="data")

# -------------------
# 建立 Data → Decision 关系（基于数值）
# -------------------

for _, row in df.iterrows():
    ward = row["ward"]

    # rent
    rent_low = row["avg_1k_rent_low_jpy"]
    rent_high = row["avg_1k_rent_high_jpy"]

    if rent_low > 100000:
        G.add_edge("rent_low", ward, weight="high")
    else:
        G.add_edge("rent_low", ward, weight="low")

    if rent_high > 130000:
        G.add_edge("rent_high", ward, weight="high")
    else:
        G.add_edge("rent_high", ward, weight="low")

    # population density
    density = row["population_density_per_km2"]
    if density > 15000:
        G.add_edge("population_density", ward, weight="high")
    else:
        G.add_edge("population_density", ward, weight="low")

    # school
    school = row["elementary_schools"]
    if school > 30:
        G.add_edge("school_count", ward, weight="high")
    else:
        G.add_edge("school_count", ward, weight="low")

# -------------------
# 输出结构（检查用）
# -------------------
print("=== Data → Decision Mapping ===")
for u, v, attrs in G.edges(data=True):
    print(f"{u} -> {v} | {attrs}")