Base: `intfloat/multilingual-e5-small / cross-encoder/mmarco-mMiniLMv2-L12-H384-v1`  
New: `intfloat/multilingual-e5-small / models/llm_judge/mmarco-minilm-human-labels`

| Locale | NDCG@10 | Base [95% CI] | New [95% CI] | Δ [95% CI] | p (paired, 2-sided) | Queries |
|---|---|---|---|---|---|---|
| jp | E5 (dense), full corpus | 0.495 [0.476, 0.516] | 0.495 [0.476, 0.516] | +0.000 [+0.000, +0.000] | 1.000 | 1000 |
| jp | Hybrid, full corpus | 0.548 [0.529, 0.567] | 0.548 [0.529, 0.567] | +0.000 [+0.000, +0.000] | 1.000 | 1000 |
| jp | Hybrid + rerank top 20, full corpus | 0.567 [0.548, 0.586] | 0.575 [0.556, 0.594] | +0.008 [+0.001, +0.015] | 0.025 | 1000 |
| jp | E5 on judged candidates | 0.737 [0.724, 0.751] | 0.737 [0.724, 0.751] | +0.000 [+0.000, +0.000] | 1.000 | 1000 |
| jp | Cross-encoder on judged candidates | 0.736 [0.703, 0.764] | 0.759 [0.728, 0.788] | +0.024 [+0.006, +0.043] | 0.011 | 200 |
| jp | hybrid α (tuned on dev) | 0.8 | 0.8 | | | |
| jp | recall@100, hybrid | 0.762 | 0.762 | | | |
| us | E5 (dense), full corpus | 0.500 [0.481, 0.519] | 0.500 [0.481, 0.519] | +0.000 [+0.000, +0.000] | 1.000 | 1000 |
| us | Hybrid, full corpus | 0.532 [0.515, 0.551] | 0.532 [0.515, 0.551] | +0.000 [+0.000, +0.000] | 1.000 | 1000 |
| us | Hybrid + rerank top 20, full corpus | 0.560 [0.541, 0.577] | 0.557 [0.538, 0.576] | -0.002 [-0.010, +0.005] | 0.484 | 1000 |
| us | E5 on judged candidates | 0.742 [0.731, 0.754] | 0.742 [0.731, 0.754] | +0.000 [+0.000, +0.000] | 1.000 | 1000 |
| us | Cross-encoder on judged candidates | 0.728 [0.698, 0.757] | 0.749 [0.719, 0.777] | +0.020 [+0.003, +0.038] | 0.020 | 200 |
| us | hybrid α (tuned on dev) | 0.8 | 0.8 | | | |
| us | recall@100, hybrid | 0.815 | 0.815 | | | |
