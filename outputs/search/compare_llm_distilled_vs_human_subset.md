Base: `intfloat/multilingual-e5-small / models/llm_judge/mmarco-minilm-human-labels`  
New: `intfloat/multilingual-e5-small / models/llm_judge/mmarco-minilm-llm-labels`

| Locale | NDCG@10 | Base [95% CI] | New [95% CI] | Δ [95% CI] | p (paired, 2-sided) | Queries |
|---|---|---|---|---|---|---|
| jp | E5 (dense), full corpus | 0.495 [0.476, 0.516] | 0.495 [0.476, 0.516] | +0.000 [+0.000, +0.000] | 1.000 | 1000 |
| jp | Hybrid, full corpus | 0.548 [0.529, 0.567] | 0.548 [0.529, 0.567] | +0.000 [+0.000, +0.000] | 1.000 | 1000 |
| jp | Hybrid + rerank top 20, full corpus | 0.575 [0.556, 0.594] | 0.569 [0.550, 0.589] | -0.006 [-0.012, +0.000] | 0.074 | 1000 |
| jp | E5 on judged candidates | 0.737 [0.724, 0.751] | 0.737 [0.724, 0.751] | +0.000 [+0.000, +0.000] | 1.000 | 1000 |
| jp | Cross-encoder on judged candidates | 0.759 [0.728, 0.788] | 0.747 [0.716, 0.776] | -0.012 [-0.027, +0.002] | 0.112 | 200 |
| jp | hybrid α (tuned on dev) | 0.8 | 0.8 | | | |
| jp | recall@100, hybrid | 0.762 | 0.762 | | | |
| us | E5 (dense), full corpus | 0.500 [0.481, 0.519] | 0.500 [0.481, 0.519] | +0.000 [+0.000, +0.000] | 1.000 | 1000 |
| us | Hybrid, full corpus | 0.532 [0.515, 0.551] | 0.532 [0.515, 0.551] | +0.000 [+0.000, +0.000] | 1.000 | 1000 |
| us | Hybrid + rerank top 20, full corpus | 0.557 [0.538, 0.576] | 0.545 [0.527, 0.563] | -0.012 [-0.019, -0.006] | < 0.001 | 1000 |
| us | E5 on judged candidates | 0.742 [0.731, 0.754] | 0.742 [0.731, 0.754] | +0.000 [+0.000, +0.000] | 1.000 | 1000 |
| us | Cross-encoder on judged candidates | 0.749 [0.719, 0.777] | 0.723 [0.692, 0.752] | -0.026 [-0.040, -0.012] | < 0.001 | 200 |
| us | hybrid α (tuned on dev) | 0.8 | 0.8 | | | |
| us | recall@100, hybrid | 0.815 | 0.815 | | | |
