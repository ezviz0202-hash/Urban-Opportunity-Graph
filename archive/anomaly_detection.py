from collections import Counter
from graph_builder_with_profile import G

# -------------------
# 统计 Type → Data 使用频率
# -------------------

type_data_pairs = []

for source, target, attrs in G.edges(data=True):
    if attrs["type"] == "uses":
        # 找这个 user 的 type
        user = source
        data = target
        
        # 找 user → type
        for u, t, a in G.edges(data=True):
            if u == user and a["type"] == "has_type":
                type_data_pairs.append((t, data))

# 统计频率
counter = Counter(type_data_pairs)

print("=== Type-Data Frequency ===")
for pair, count in counter.items():
    print(f"{pair}: {count}")

# -------------------
# 找异常（低频）
# -------------------

print("\n=== Potential Anomalies ===")

threshold = 1  # 出现次数 ≤1 视为异常

for pair, count in counter.items():
    if count <= threshold:
        print(f"Anomaly: {pair} (count={count})")