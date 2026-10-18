Base: `intfloat/multilingual-e5-small / cross-encoder/mmarco-mMiniLMv2-L12-H384-v1`  
New: `models/e5-small-esci / models/mmarco-minilm-esci`

| Locale | NDCG@10 | Base [95% CI] | New [95% CI] | Δ [95% CI] | p (paired, 2-sided) | Queries |
|---|---|---|---|---|---|---|
| jp | E5 (dense), full corpus | 0.495 [0.476, 0.516] | 0.560 [0.541, 0.579] | +0.065 [+0.054, +0.077] | < 0.001 | 1000 |
| jp | Hybrid, full corpus | 0.548 [0.529, 0.567] | 0.582 [0.564, 0.602] | +0.034 [+0.025, +0.042] | < 0.001 | 1000 |
| jp | Hybrid + rerank top 20, full corpus | 0.567 [0.548, 0.586] | 0.595 [0.577, 0.614] | +0.029 [+0.020, +0.038] | < 0.001 | 1000 |
| jp | E5 on judged candidates | 0.737 [0.724, 0.751] | 0.755 [0.741, 0.768] | +0.017 [+0.010, +0.025] | < 0.001 | 1000 |
| jp | Cross-encoder on judged candidates | 0.736 [0.703, 0.764] | 0.774 [0.744, 0.802] | +0.038 [+0.018, +0.059] | 0.002 | 200 |
| jp | hybrid α (tuned on dev) | 0.8 | 0.8 | | | |
| jp | recall@100, hybrid | 0.762 | 0.795 | | | |
| us | E5 (dense), full corpus | 0.500 [0.481, 0.519] | 0.542 [0.523, 0.561] | +0.041 [+0.031, +0.052] | < 0.001 | 1000 |
| us | Hybrid, full corpus | 0.532 [0.515, 0.551] | 0.559 [0.541, 0.577] | +0.027 [+0.019, +0.035] | < 0.001 | 1000 |
| us | Hybrid + rerank top 20, full corpus | 0.560 [0.541, 0.577] | 0.576 [0.558, 0.593] | +0.016 [+0.005, +0.026] | 0.001 | 1000 |
| us | E5 on judged candidates | 0.742 [0.731, 0.754] | 0.761 [0.750, 0.774] | +0.019 [+0.012, +0.027] | < 0.001 | 1000 |
| us | Cross-encoder on judged candidates | 0.728 [0.698, 0.757] | 0.771 [0.742, 0.799] | +0.043 [+0.024, +0.062] | < 0.001 | 200 |
| us | hybrid α (tuned on dev) | 0.8 | 0.8 | | | |
| us | recall@100, hybrid | 0.815 | 0.851 | | | |
