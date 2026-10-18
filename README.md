# Bilingual product search and recommendation

[![tests](https://github.com/Praruj/bilingual-product-search/actions/workflows/ci.yml/badge.svg)](https://github.com/Praruj/bilingual-product-search/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.10-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-service-009688)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)

Product search for **Japanese and English** shoppers on Amazon's public Shopping Queries dataset (ESCI),
plus next-purchase recommendation on real e-commerce transactions, served by a small FastAPI app on a
laptop CPU. Each number is measured on held-out data and comes with a 95% bootstrap confidence interval.

| Search, NDCG@10 on 1,000 held-out queries | Japanese (amazon.co.jp) | English (amazon.com) |
|---|---|---|
| BM25, naive whitespace tokens | 0.280 | 0.422 |
| BM25, proper tokenizer (MeCab / words) | 0.492 | 0.455 |
| Multilingual E5-small (dense) | 0.495 | 0.500 |
| **Hybrid** (BM25 + E5, weight tuned on train queries) | **0.548** | **0.532** |
| Hybrid + cross-encoder rerank of top 20 | 0.567 | 0.560 |
| Recall@100 of Exact products, hybrid | 0.762 | 0.815 |

| Recommendation, NDCG@20 on next-quarter purchases (2,336 customers) | |
|---|---|
| Popularity | 0.107 |
| EASE (best learned model) | 0.260 |
| **Re-recommend the customer's own past purchases** | **0.336** |

![Search results](outputs/figures/search_results.png)

## What I found

**1. For Japanese, tokenization matters more than the choice of model.** Japanese is written without spaces,
so an English-style analyzer treats a whole query like `ステンレス水筒` as one token. Splitting words with
MeCab raises BM25 from 0.280 to 0.492 NDCG@10, a bigger jump than any model change after it.
Dictionary-free character bigrams score the same as MeCab (0.492) but give a 2.6× larger index (57 MB vs
22 MB for 50k products).

**2. Dense and sparse retrieval fail on different queries, so combining them is the biggest model win.**
On Japanese, BM25 (0.492) and E5 (0.495) are statistically tied, yet the hybrid reaches 0.548 (+0.053,
p < 0.001, paired bootstrap). On English, E5 clearly beats BM25 (0.500 vs 0.455) and the hybrid still adds
+0.032 (p < 0.001). A tuned weighted sum beat parameter-free reciprocal rank fusion in both languages
(0.548 vs 0.534; 0.532 vs 0.507, where RRF is not significantly better than E5 alone). The weight was
chosen on separate train-split queries, never on the test queries.

![Hybrid weight](outputs/figures/hybrid_weight.png)

**3. 7% of "Japanese" queries contain no Japanese at all, and the dense model is weaker on them.** Model
numbers, brands and English words (for example `ubuntu`) make up 7.2% of the amazon.co.jp test queries. On
those 72 queries E5 scores 0.402 against BM25's 0.469, which fits the idea that exact string matching helps
with part numbers. The hybrid covers both (0.504). With only 72 queries this gap is suggestive, not proven.

**4. An off-the-shelf cross-encoder is not a free upgrade.** Reranking the hybrid top 20 adds +0.019
(Japanese) and +0.028 (English) NDCG@10, but costs about 1.2 s per query on this CPU. When it only
reorders the judged candidates of each query (the official ESCI ranking task), it does **no better than E5
alone** (Japanese 0.736 vs 0.737; English 0.728 vs 0.742). The model was trained on MS MARCO web passages, not
product listings. Fine-tuning it on ESCI's training labels is the obvious next step, and needs a GPU.

**5. In recommendation, the strongest baseline is "buy it again".** In this retailer's data, 38% of what a
customer buys in the test quarter they had bought before (many customers are wholesalers who re-stock). Simply
re-recommending their own most frequent items scores 0.336 NDCG@20, well above EASE (0.260), ALS (0.247) and
item-kNN (0.209). Blending EASE into it (0.337) is not a significant gain (p = 0.17). The learned models only
earn their place on **new** items, where EASE beats 30-day popularity by 1.7× (0.093 vs 0.055).

![Recommendation results](outputs/figures/recs_results.png)

## Serving

`uvicorn bps.service:app --app-dir src` serves both catalogues and the recommender:

```
GET /search?q=ワイヤレスイヤホン&mode=hybrid      # locale auto-detected from the script; mode = bm25 | dense | hybrid | rerank
GET /search?q=water+bottle&locale=us&k=10
GET /recommend/12347?k=10&new_only=true         # unknown customers fall back to best sellers
```

Measured over real HTTP with uvicorn on the same laptop (Intel Core, 7th gen, 2 PyTorch threads), 200 held-out
queries per language sent one at a time:

| Mode | Median latency (p95) | NDCG@10 JA / EN |
|---|---|---|
| BM25 | 3.5 ms (4.7 ms) | 0.492 / 0.455 |
| Dense E5 | 32 ms (43 ms) | 0.495 / 0.500 |
| **Hybrid** | **32 ms (40 ms)** | **0.548 / 0.532** |
| Hybrid + rerank top 20 | 1,171 ms (1,243 ms) | 0.567 / 0.560 |
| Recommendation (EASE) | 3.3 ms (4.5 ms) | |

Almost all of the dense and hybrid time is encoding the query with E5 (BM25 and the 50k × 384 dot product
take under 2 ms), so the hybrid costs nothing over dense alone. On disk, each language's index is 50 MB
(E5 vectors in float16 38 MB, BM25 matrix 9–12 MB) plus the product text; the server holds both catalogues,
both neural models and the recommender in 1.6 GB of RAM. The reranker is the one mode too slow for
interactive search on this CPU. A GPU, a smaller or distilled reranker, or reranking fewer candidates would
be needed to use it in production.

![Latency vs quality](outputs/figures/latency_quality.png)

## How it works

**Data.** [Shopping Queries Dataset](https://github.com/amazon-science/esci-data) (Reddy et al., 2022,
Apache 2.0): real Amazon queries with products labelled Exact, Substitute, Complement or Irrelevant. For each
locale, 1,000 test queries are sampled from the official test split and 300 dev queries from the train split
(seeded). The corpus is every product judged for those queries plus distractors (products judged for *other*
queries in the same locale) up to 50,000 products. Product text is title, brand, colour and the first 300
characters of the bullet points.

**Metrics.** NDCG@10 with the ESCI gains used in KDD Cup 2022 (E = 1, S = 0.1, C = 0.01, I = 0) and
recall@100 of Exact products. 95% intervals resample queries (2,000 bootstrap draws); p-values are paired
bootstraps on the same queries.

**Retrieval.**
- BM25 (k1 = 1.2, b = 0.75) written as one sparse matrix product, under 1 ms per query for 50k products.
  Text is NFKC-normalised first, so full-width `ＵＳＢ` matches `usb`. Japanese uses MeCab with UniDic-lite
  (`fugashi`).
- Dense: `intfloat/multilingual-e5-small` (118M parameters, 384-dim), documents truncated to 64 tokens,
  exact inner-product search over all vectors. No ANN index is needed at this size.
- Hybrid: `α · dense + (1 − α) · BM25` on min-max normalised scores, α picked on dev queries (0.8 for both
  languages; the high weight partly reflects that min-max over the whole corpus squeezes cosine scores into a
  narrow band). RRF (k = 60) is reported for comparison.
- Rerank: `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1` (multilingual, MS MARCO) on the hybrid top 20.

**Recommendation.** [Online Retail II](https://archive.ics.uci.edu/dataset/502/online+retail+ii) (UCI,
CC BY 4.0): 803k purchase lines from a UK online gift retailer, Dec 2009 to Dec 2011, after removing
cancellations, guest checkouts and non-product codes (postage, fees). Time-based split: hyperparameters are
tuned on June–August 2011 with training data before June, then models are retrained on everything before
September and tested on September to December 9, 2011, for customers who bought in both periods.
Recall@20 is normalised by min(|bought|, 20). Models: popularity (all time and last 30 days), own-history
re-buy, item-kNN cosine, implicit ALS (Hu et al., 2008; exact solves, written in NumPy), and EASE (Steck, 2019).

## Fine-tuning on ESCI

**Status: set up, not yet run.** The numbers above are all off-the-shelf models. The before/after table
below will be filled from `scripts/compare_runs.py` once the Colab training and the laptop re-evaluation
have run; nothing here is an estimate.

**Training data.** `scripts/make_finetune_data.py` takes ESCI `small_version` train-split queries for jp and
us and removes the 300 dev queries per language that the hybrid weight is tuned on, leaving 27,572 queries and
614,450 judged (query, product) pairs. It asserts that no training query is an evaluation query and that no
training query has the same text as any test-split query. Product text is the same as in the evaluation corpus.

**Models** ([notebooks/colab_finetune.ipynb](notebooks/colab_finetune.ipynb), one T4 GPU, fixed
hyperparameters chosen before seeing any test result):
- E5-small: contrastive loss over in-batch negatives plus one hard negative per query, taken from that query's
  products labelled Irrelevant or Substitute; batches hold one language and each query at most once, so a
  query's other Exact products are never treated as negatives. 2 epochs, up to 4 Exact products per query per
  epoch, batch 128, learning rate 3e-5.
- Cross-encoder: binary cross-entropy on every judged pair with the ESCI gains as soft targets
  (E 1, S 0.1, C 0.01, I 0), 1 epoch, batch 64, learning rate 2e-5.

**Evaluation.** `scripts/run_search.py --encoder … --reranker … --tag finetuned` reruns the whole pipeline on
the same 1,000 test queries per language and the same 50k-product corpora, re-tuning the hybrid weight on the
same dev queries. `scripts/compare_runs.py --new finetuned` reports each NDCG@10 before and after with 95%
intervals, the paired difference with its interval, and a two-sided paired bootstrap p-value.

<!-- results table from outputs/search/compare_finetuned.md goes here -->

## Limitations

- **Unjudged products count as irrelevant.** ESCI only labels about 28 products per query, so a relevant
  product that no annotator saw scores zero. This penalises every method, probably dense retrieval more
  since it finds products with no word overlap. Absolute numbers are lower bounds; the comparisons are the
  point.
- The corpus is 50k products per locale, not Amazon's full catalogue, because encoding runs at about 40
  products per second on this laptop (2 threads). A larger corpus makes every method look worse.
- Nothing is fine-tuned on ESCI. Both neural models are used as published.
- The latency is one laptop, one request at a time. Throughput under concurrency is not measured.
- Online Retail II is one UK retailer with many wholesale buyers, so "buy it again" is unusually strong.
  On a marketplace like Rakuten Ichiba, the balance between re-buys and discovery will be different.
- No cross-lingual evaluation: Japanese queries are only scored against the Japanese catalogue, because
  ESCI has no labels for English queries on Japanese products.

## Reproduce

```bash
python -m venv .venv && .venv/Scripts/activate           # Windows; use bin/activate elsewhere
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt

# data (not in the repo)
#   ESCI (Apache 2.0), into data/raw/esci/:
#     https://github.com/amazon-science/esci-data/tree/main/shopping_queries_dataset
#     shopping_queries_dataset_examples.parquet (51 MB), shopping_queries_dataset_products.parquet (1.1 GB)
#   Online Retail II (CC BY 4.0), into data/raw/online_retail/:
#     https://archive.ics.uci.edu/static/public/502/online+retail+ii.zip (46 MB)

python scripts/run_recs.py        # ~6 min
python scripts/run_search.py      # ~2 h on a 2-core laptop CPU, mostly encoding and reranking
python scripts/build_index.py     # artifacts for the API in outputs/index/
python scripts/bench_service.py   # starts the API and measures latency
python scripts/report.py          # figures in outputs/figures

# optional: fine-tuning (training on a Colab GPU, evaluation on the laptop)
python scripts/make_finetune_data.py   # data/finetune/esci_train_pairs.parquet -> upload to MyDrive/bps/
#   run notebooks/colab_finetune.ipynb on Colab, download MyDrive/bps/models/ into models/
python scripts/run_search.py --encoder models/e5-small-esci --reranker models/mmarco-minilm-esci --tag finetuned
python scripts/compare_runs.py --new finetuned
pytest -q                         # 16 tests on synthetic data, no downloads
```

## 日本語概要

Amazonの公開データセット「Shopping Queries（ESCI）」を使った日本語・英語の商品検索と、実際のEC購買データによる
レコメンドのプロジェクトです。FastAPIでノートPCのCPU上で動かし、すべての指標を未使用のテストデータで評価しています。

- **検索**（NDCG@10、各言語1,000クエリ）：日本語は空白区切りのBM25が0.280、MeCabで形態素解析すると0.492、
  多言語E5が0.495、両者のハイブリッドが**0.548**。英語はBM25 0.455、E5 0.500、ハイブリッド**0.532**。
- **レコメンド**（NDCG@20、2,336人の次の四半期の購買）：人気順0.107、EASE 0.260、
  **過去に買った商品の再推薦0.336**。

主な発見：
1. 日本語検索では、モデルの選択よりも単語分割（トークナイズ）の方が効果が大きい。
2. BM25とベクトル検索は苦手なクエリが異なるため、組み合わせると統計的に有意に改善する（重みは学習用クエリで調整）。
3. amazon.co.jpのクエリの約7%は型番や英単語のみで、そこではベクトル検索がBM25に負ける。
4. 既製のクロスエンコーダによる再ランキングは1クエリ約1.2秒かかり、ESCIの公式設定ではE5単体と同等。
   ESCIでのファインチューニングが次の課題。
5. レコメンドでは「同じ商品をまた買う」が最強のベースライン。学習モデルが勝つのは新しい商品の推薦のみ（EASEは人気順の1.7倍）。

## License

Code: MIT. Data is not redistributed: ESCI is Apache 2.0 (Amazon), Online Retail II is CC BY 4.0 (Chen, 2012,
UCI Machine Learning Repository).
