Base: `intfloat/multilingual-e5-small / cross-encoder/mmarco-mMiniLMv2-L12-H384-v1`  
New: `intfloat/multilingual-e5-small / models/llm_judge/mmarco-minilm-llm-labels`

| Locale | NDCG@10 | Base [95% CI] | New [95% CI] | Δ [95% CI] | p (paired, 2-sided) | Queries |
|---|---|---|---|---|---|---|
| jp | E5 (dense), full corpus | 0.495 [0.476, 0.516] | 0.495 [0.476, 0.516] | +0.000 [+0.000, +0.000] | 1.000 | 1000 |
| jp | Hybrid, full corpus | 0.548 [0.529, 0.567] | 0.548 [0.529, 0.567] | +0.000 [+0.000, +0.000] | 1.000 | 1000 |
| jp | Hybrid + rerank top 20, full corpus | 0.567 [0.548, 0.586] | 0.569 [0.550, 0.589] | +0.003 [-0.005, +0.010] | 0.520 | 1000 |
| jp | E5 on judged candidates | 0.737 [0.724, 0.751] | 0.737 [0.724, 0.751] | +0.000 [+0.000, +0.000] | 1.000 | 1000 |
| jp | Cross-encoder on judged candidates | 0.736 [0.703, 0.764] | 0.747 [0.716, 0.776] | +0.012 [-0.009, +0.032] | 0.271 | 200 |
| jp | hybrid α (tuned on dev) | 0.8 | 0.8 | | | |
| jp | recall@100, hybrid | 0.762 | 0.762 | | | |
| us | E5 (dense), full corpus | 0.500 [0.481, 0.519] | 0.500 [0.481, 0.519] | +0.000 [+0.000, +0.000] | 1.000 | 1000 |
| us | Hybrid, full corpus | 0.532 [0.515, 0.551] | 0.532 [0.515, 0.551] | +0.000 [+0.000, +0.000] | 1.000 | 1000 |
| us | Hybrid + rerank top 20, full corpus | 0.560 [0.541, 0.577] | 0.545 [0.527, 0.563] | -0.015 [-0.023, -0.007] | < 0.001 | 1000 |
| us | E5 on judged candidates | 0.742 [0.731, 0.754] | 0.742 [0.731, 0.754] | +0.000 [+0.000, +0.000] | 1.000 | 1000 |
| us | Cross-encoder on judged candidates | 0.728 [0.698, 0.757] | 0.723 [0.692, 0.752] | -0.005 [-0.022, +0.012] | 0.540 | 200 |
| us | hybrid α (tuned on dev) | 0.8 | 0.8 | | | |
| us | recall@100, hybrid | 0.815 | 0.815 | | | |
