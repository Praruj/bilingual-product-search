# Where the LLM says Irrelevant and the human says Exact: a read of 20 cases

20 random `view` pairs (test split) with human label E and LLM argmax I (`sample(20, random_state=0)`).
Verdicts are one reader's judgement (written by Claude while preparing the README), not a second annotation
pass, so treat the split as a rough indication only.

| # | Locale | Query | Product (start) | Verdict |
|---|---|---|---|---|
| 0 | us | 72 inch vanity without top | ARIEL 60" grey bathroom vanity base cabinet | human label doubtful (60" ≠ 72") |
| 1 | jp | no1グリップ バックラインなし | NO1グリップ 48シリーズ バックラインあり | human label doubtful (あり vs なし) |
| 2 | us | bole salts | Dr. Berg's gallbladder support supplements | human label doubtful |
| 3 | us | blade blade metal fusion | The Twin Gemios | human label doubtful (no information) |
| 4 | jp | buraketto | Korean-language essay book | human label doubtful |
| 5 | us | .40 cal suppressor oil filter not as it seems | Bosch car oil filter | human label doubtful |
| 6 | us | black and decker sawzall blades | SUNGATOR reciprocating saw blades | LLM error (generic blades fit) |
| 7 | us | 1 acre bug zapper outdoor without octenal attractant | DynaTrap replacement lure cartridge | human label doubtful (refill, not a zapper) |
| 8 | jp | いしいひさいち | ののちゃんセレクション (his comic) | LLM error (author query) |
| 9 | jp | fracarkos | sewing machine, brand FRACARKOS | LLM error (brand query) |
| 10 | us | amzcool | drink cup holder, brand AMZCOOL | LLM error (brand query) |
| 11 | us | 4k tv swivel for 2 leg tvs | big-screen TV swivel base | LLM error |
| 12 | jp | led電球ペット用 | aquarium LED light | human label doubtful |
| 13 | jp | 20リットル ゴミ袋 取っ手なし | 10–15 L bags with handles (取っ手付き) | human label doubtful |
| 14 | jp | k.y.factory | K.Y.FACTORY ring | LLM error (brand query) |
| 15 | us | boys crop top | long-sleeve athletic shirts | human label doubtful |
| 16 | jp | 2級土木施工管理技士テキスト | past-exam book for the same exam | LLM error (at most Substitute) |
| 17 | us | 1 acre bug zapper outdoor without octenal attractant | LUFFYLIVE electric bug zapper | LLM error |
| 18 | us | 10" glass japanese floating glass without netting balls | stainless-steel gazing globe | human label doubtful |
| 19 | us | black rifle coffe ground | Black Rifle coffee rounds (pods) | LLM error (Substitute) |

Count: human label doubtful 11, LLM error 9. The LLM's typical miss is a bare brand or author name, where
any product of that brand is what the shopper wants; the typical human-label problem is a stated attribute
(size, "without X") that the product contradicts.
