import math
import random
import pandas as pd

random.seed(42)

def normalize(series: pd.Series, reverse: bool = False) -> pd.Series:
    values = series.astype(float)
    lo, hi = values.min(), values.max()
    if hi == lo:
        scaled = pd.Series([0.5] * len(values), index=values.index)
    else:
        scaled = (values - lo) / (hi - lo)
    return 1 - scaled if reverse else scaled

def softmax(values, temperature: float = 0.55):
    scaled = [v / temperature for v in values]
    max_v = max(scaled)
    exp_values = [math.exp(v - max_v) for v in scaled]
    total = sum(exp_values)
    return [v / total for v in exp_values]

def choose_by_softmax(df: pd.DataFrame, scores, temperature: float = 0.55):
    probs = softmax(scores, temperature=temperature)
    idx = random.choices(range(len(df)), weights=probs, k=1)[0]
    chosen = df.iloc[idx].copy()
    chosen["choice_prob"] = probs[idx]
    chosen["raw_score"] = scores[idx]
    return chosen

profiles = pd.read_csv("data/user_profile.csv")
tokyo = pd.read_csv("data/tokyo_ward_data_extended.csv").copy()

tokyo["rent_low_score"] = normalize(tokyo["avg_1k_rent_low_jpy"], reverse=True)
tokyo["rent_high_score"] = normalize(tokyo["avg_1k_rent_high_jpy"], reverse=True)
tokyo["population_density_score"] = normalize(tokyo["population_density_per_km2"])
tokyo["school_count_score"] = normalize(tokyo["elementary_schools"])
tokyo["crime_rate_score"] = normalize(tokyo["crime_rate"], reverse=True)
tokyo["transport_score_norm"] = normalize(tokyo["transport_score"])
tokyo["commercial_score_norm"] = normalize(tokyo["commercial_score"])
tokyo["livability_score_norm"] = normalize(tokyo["livability_score"])

base_preferences = {
    "student": {"school_count_score":0.28,"population_density_score":0.18,"rent_high_score":0.12,"transport_score_norm":0.20,"commercial_score_norm":0.17,"crime_rate_score":0.05},
    "worker": {"rent_low_score":0.20,"population_density_score":0.18,"transport_score_norm":0.28,"commercial_score_norm":0.18,"crime_rate_score":0.10,"school_count_score":0.06},
    "family": {"school_count_score":0.24,"rent_low_score":0.18,"livability_score_norm":0.24,"crime_rate_score":0.16,"population_density_score":0.08,"transport_score_norm":0.05,"commercial_score_norm":0.05},
}

type_data_probs = {
    "student": {"school_count":0.90,"population_density":0.65,"rent_high":0.35,"transport_score":0.75,"commercial_score":0.70,"crime_rate":0.20},
    "worker": {"rent_low":0.75,"population_density":0.60,"transport_score":0.90,"commercial_score":0.75,"crime_rate":0.45,"school_count":0.20},
    "family": {"school_count":0.85,"rent_low":0.75,"livability_score":0.85,"crime_rate":0.65,"population_density":0.30,"transport_score":0.20,"commercial_score":0.20},
}

data_to_feature = {
    "rent_low":"rent_low_score","rent_high":"rent_high_score","population_density":"population_density_score","school_count":"school_count_score","crime_rate":"crime_rate_score","transport_score":"transport_score_norm","commercial_score":"commercial_score_norm","livability_score":"livability_score_norm"
}

def select_user_data(user_type):
    probs = type_data_probs[user_type]
    selected = [d for d,p in probs.items() if random.random()<p]
    if len(selected)<2:
        ranked = sorted(probs.items(), key=lambda x:x[1], reverse=True)
        for d,_ in ranked:
            if d not in selected:
                selected.append(d)
            if len(selected)>=2:
                break
    return selected

def build_weights(selected_data,user_type):
    base = base_preferences[user_type]
    weights={}
    for d in selected_data:
        f=data_to_feature[d]
        if f in base:
            weights[f]=base[f]
    if not weights:
        weights=base.copy()
    noisy={}
    for k,v in weights.items():
        noisy[k]=max(0.01,v+random.uniform(-0.08,0.08))
    total=sum(noisy.values())
    return {k:v/total for k,v in noisy.items()}

def score_wards(df,weights):
    return [sum(row[f]*w for f,w in weights.items()) for _,row in df.iterrows()]

user_data=[]
user_decision=[]
user_weights=[]

for _,p in profiles.iterrows():
    user=p["user"]; t=p["type"]
    if t not in base_preferences: continue
    sel=select_user_data(t)
    for d in sel: user_data.append({"user":user,"data":d})
    w=build_weights(sel,t)
    rec={"user":user,"type":t,"selected_data":",".join(sel)}
    rec.update(w); user_weights.append(rec)
    scores=score_wards(tokyo,w)
    c=choose_by_softmax(tokyo,scores)
    user_decision.append({"user":user,"decision":c["ward"],"raw_score":round(c["raw_score"],4),"choice_prob":round(c["choice_prob"],4),"selected_data":",".join(sel)})

pd.DataFrame(user_data).to_csv("data/user_data_real.csv",index=False)
pd.DataFrame(user_decision).to_csv("data/user_decision_real.csv",index=False)
pd.DataFrame(user_weights).to_csv("data/user_weights_real.csv",index=False)

print("Behavior data generated.")
