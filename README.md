# Urban Opportunity Graph

**Can a graph reveal why workers choose dangerous neighborhoods?**

This project builds a graph-based simulation framework to discover hidden opportunity structures in Tokyo's urban data. Instead of optimizing individual decisions, it models how heterogeneous users — students, workers, and families — interact with multi-dimensional urban data to make ward-level decisions, then surfaces the structural anomalies that conventional preference models miss.

Core finding: **workers chose Shinjuku despite its crime safety score of 0.0 out of 1.0** — the lowest among all 23 wards. The hidden attractor was commercial density, scoring 1.0. This counter-intuitive pattern reveals that urban opportunity zones can hide behind negative signals, invisible to single-objective optimization.

---

## What This Project Does

```
Tokyo ward data ──► User behavior simulation ──► Interaction graph
                                                        │
                                    ┌───────────────────┼──────────────────┐
                                    ▼                   ▼                  ▼
                             Anomaly scoring      Bridge node         Contradiction
                             rarity × centrality   analysis            detection
                                    └───────────────────┴──────────────────┘
                                                        │
                                              Hidden opportunity structures
```

Three heterogeneous user types interact with eight urban data features to make ward-level decisions. The graph that emerges from these interactions is not designed — it is discovered through co-creation. Anomaly detection then surfaces rare but structurally significant patterns in this interaction space.

---

## Key Findings

### Finding 1: Workers are pulled into unsafe wards by commercial gravity

| User type | Chosen ward | Safety score | Median safety | Hidden attractor    | Attractor score |
|-----------|-------------|-------------|---------------|---------------------|----------------|
| worker    | Shinjuku    | **0.000**   | 0.719         | commercial density  | **1.000**      |
| worker    | Shibuya     | 0.125       | 0.719         | commercial density  | 0.974          |
| worker    | Shinagawa   | 0.688       | 0.719         | transport access    | 0.920          |

Workers who explicitly weighted crime rate in their data selection still chose wards with the lowest safety scores in Tokyo. The structural attractor was commercial density — a feature outside their stated primary concern. This is a systematic pattern, not noise: urban opportunity zones can hide behind negative safety signals.

### Finding 2: Families trade school access for commercial vibrancy

Families choosing Shinjuku (school accessibility = 0.256, below median 0.326) were pulled by commercial density (1.0). This reveals a latent family subtype invisible to conventional preference optimization — one for whom commercial access overrides the assumed primary concern.

### Finding 3: Bridge nodes in the data layer

Standard anomaly scoring surfaces `family → commercial_score` as the strongest signal (score = 0.912), driven by rarity combined with high node centrality. The least expected connection carries the most structural information.

```
Top anomaly signals:
  family  → commercial_score  | score=0.912
  worker  → crime_rate        | score=0.697
  student → commercial_score  | score=0.456
```

---

## Methodology

**Data**: Tokyo 23-ward dataset with 8 urban indicators — rent (low/high range), population density, elementary schools, crime rate, transport accessibility, commercial activity, livability index. Compiled from Tokyo Metropolitan Government Open Data Portal and e-Stat.

**User simulation** (`generate_user_behavior.py`): Each user type has base preferences encoded as feature weights. Softmax-weighted selection (temperature=0.55) introduces behavioral heterogeneity across individuals of the same type.

**Graph construction** (`build_graph.py`): Directed heterogeneous graph with four node layers:

```
User ──has_type──► UserType ──prefers──► DataFeature ──influences──► Ward
 └──────────────────uses──────────────────────┘           └──makes──┘
```

**Anomaly scoring** (`score_anomalies.py`):
```
anomaly_score = rarity × centrality × log(1 + type_size)
```

**Contradiction detection** (`score_anomalies.py`): Identifies wards chosen despite scoring below median on the user type's primary concern. Records the highest-scoring feature in that ward as the hidden attractor.

**Bridge node analysis** (`discover_opportunities.py`):
```
bridge_score = rarity × n_types_connected × (1 + betweenness_centrality)
```
Finds data features that connect multiple user type clusters — structurally rare but cross-cutting nodes in the interaction space.

---

## How to Run

```bash
pip install -r requirements.txt

python src/generate_user_behavior.py   # simulate user-data interactions
python src/build_graph.py              # construct the interaction graph
python src/score_anomalies.py          # anomaly scoring + contradiction detection
python src/discover_opportunities.py   # bridge node analysis
python src/visualize_anomalies.py      # generate highlighted visualization
```

---

## Output Files

```
outputs/
  network_real_v2.html           full interaction graph (interactive)
  network_real_v2_anomaly.html   anomaly-highlighted graph
  anomaly_results.csv            standard anomaly scores
  contradiction_choices.csv      counter-intuitive ward selections
  bridge_nodes.csv               bridge node scores
  cross_type_patterns.csv        aggregated cross-type opportunity patterns
```

---

## Project Structure

```
src/
  generate_user_behavior.py    softmax-based user behavior simulation
  build_graph.py               heterogeneous graph construction (NetworkX + PyVis)
  score_anomalies.py           anomaly scoring + contradiction detection
  discover_opportunities.py    bridge node analysis + cross-type patterns
  visualize_anomalies.py       anomaly-highlighted graph visualization
data/                          Tokyo ward dataset + simulated behavioral data
outputs/                       generated graphs and analysis results
archive/                       development history
```

---

## Future Directions

- Replace simulated behavior with real mobility or transaction logs
- Temporal graph evolution: track how opportunity structures shift over time
- Multi-city extension: compare opportunity structures across Tokyo, Osaka, Nagoya
- Integration with privacy-preserving data sharing frameworks for multi-source urban data

---

## References

- Ohsawa, Y. (1996). KeyGraph: Automatic Indexing by Co-occurrence Graph Based on Building Construction Metaphor. *Proc. Advanced Digital Library Conference*, IEEE.
- Ohsawa, Y. (2003). *Chance Discovery*. Springer.