# Technical Manual Assistant (RAG)

A retrieval-augmented question-answering service over equipment maintenance manuals. It finds the relevant
passages with FAISS, has a local Llama 3.1 8B model (through Ollama) answer from them, cites the page it used,
declines when the manuals do not contain the answer, and can read wiring diagrams with LLaVA. Served with
FastAPI (with latency monitoring) and a Streamlit chat page.

```
question ──> embed (BGE-small) ──> FAISS top-k chunks ──> similarity gate ──> prompt ──> Llama 3.1 8B ──> answer + pages
                                          │                                      ▲
                                          └─ figure page? ──> page image ──> LLaVA 7B reading ──┘
```

## Corpus

Four public U.S. Bureau of Reclamation maintenance manuals (FIST series), **608 pages**, fetched by
`scripts/fetch_manuals.py` and not stored in this repository:

| manual | pages |
|---|---|
| FIST 3-8, protective relays: operation, maintenance and field test | 125 |
| FIST 3-16, maintenance of power circuit breakers | 99 |
| FIST 4-1A, maintenance scheduling for mechanical equipment | 222 |
| FIST 4-1B, maintenance scheduling for electrical equipment | 162 |

## Test sets

- **120 answerable questions** (`eval/questions.jsonl`), written from pages sampled at random (fixed seed), each
  with the short facts a correct answer must contain and the page it comes from. A script checks that every
  fact appears on that page. Other pages that hold the same anchor phrase and facts also count as correct
  retrievals. They are split into **60 development** questions (used to choose settings) and **60 test**
  questions (`eval/split.json`).
- **40 unanswerable questions** (`eval/unanswerable.jsonl`): 10 off-topic and 30 about equipment, makes and
  models that these manuals do not cover. The make or model in each question does not appear in the corpus, or
  appears only in passing (Siemens-Allis breakers in a watt-loss example, Woodward actuators in a governor
  checklist, "laptop" and "firmware" in relay-testing notes); those passages were read and none answers the question.
- **13 diagram questions** (`eval/diagrams.jsonl`) about figures in the relay and breaker manuals.

An answer is scored correct when it contains every fact group (`eval/common.py`); a decline is never correct.
This is a keyword check, so it can pass a wrong answer that happens to mention the right words and fail a
correct answer that is phrased differently.

## Results

### Retrieval: 3 chunking strategies x 2 embedding models

A question is a hit at k when one of the top k chunks is on a correct page. Chunks never cross a page boundary.

| chunking | embedder | chunks | dev hit@5 | dev MRR | test hit@1 | test hit@5 | test MRR |
|---|---|---|---|---|---|---|---|
| 500 characters, 50 overlap | MiniLM-L6 | 2,452 | 93.3% | 0.754 | 63.3% | 83.3% | 0.716 |
| lines packed to 1,000 characters | MiniLM-L6 | 1,360 | 90.0% | 0.697 | 61.7% | 78.3% | 0.686 |
| whole page | MiniLM-L6 | 557 | 86.7% | 0.738 | 66.7% | 86.7% | 0.738 |
| **500 characters, 50 overlap** | **BGE-small** | 2,452 | **95.0%** | 0.840 | **75.0%** | **88.3%** | **0.809** |
| lines packed to 1,000 characters | BGE-small | 1,360 | 93.3% | 0.787 | 70.0% | 86.7% | 0.757 |
| whole page | BGE-small | 557 | 93.3% | 0.792 | 65.0% | 86.7% | 0.747 |

The configuration was chosen on the dev questions (best hit@5, then MRR). BGE-small is ahead of MiniLM on dev hit@5 and on MRR in every chunking
strategy. The differences between chunkings are small, and with 60 questions each one moves hit@5 by about
1.7 points, so only the embedder effect and the gap between the best and worst rows are clearly more than noise.

### Answer accuracy (Llama 3.1 8B, top 4 chunks)

Four prompt templates on the 60 dev questions, with the chosen retrieval:

| prompt | accuracy | declined | answers citing a page | cited page correct |
|---|---|---|---|---|
| minimal ("use the context to answer") | 86.7% | 0% | 30% | 67% |
| grounded + cite | 83.3% | 11.7% | 98% | 92% |
| **technician** | 86.7% | 5% | 91% | 92% |
| concise + source line | 85.0% | 1.7% | 97% | 88% |

The four prompts are within two questions of each other, which is noise at this sample size. "technician" tied
with "minimal" on accuracy and was kept because its answers cite a page. (The first run broke that tie by retrieval
hit rate, which is identical for every prompt, so it fell through to "minimal"; the tie-break was then changed to
citation rate before any test result for the shipped prompt existed. The first-pass output is kept in
`reports/generation_first_pass.json`.)

Baseline against the final system (baseline = 500-character chunks + MiniLM + minimal prompt):

| | test (60) | all 120 | page in retrieved sources | answers citing a page | cited page correct |
|---|---|---|---|---|---|
| baseline | 75.0% | 79.2% | 86.7% | 20.8% | 64% |
| chosen retrieval, minimal prompt | 81.7% | 84.2% | 90.8% | 30.0% | 75% |
| **final** (chosen retrieval + technician prompt) | 78.3% | 82.5% | 90.8% | 94.6% | 93% |

The gain in answer accuracy from baseline to final is **+3.3 points on both the 60 test and all 120 questions, and
it is not statistically distinguishable from zero** (paired bootstrap 95% interval: -6.7 to +13.3 points on test,
-3.3 to +10.0 on all 120; `reports/generation_paired.json`). What did change clearly is that the right page is in
the retrieved sources more often, and that answers now cite their page (21% to 95%, and the cited page is correct
93% of the time). The technician prompt also makes the model decline 9 of the 120 answerable questions (7.5%);
those count as wrong above. "All 120" includes the dev questions that were used to choose settings, so the test
column is the cleaner number.

### Declining when the manuals do not answer

On the 40 unanswerable questions:

| setup | unsupported answers |
|---|---|
| baseline (minimal prompt, no instruction to decline) | 5 of 40 (12.5%), by hand-labelling (`eval/guardrail_labels.json`) |
| technician prompt (told to answer "I could not find this in the manuals") | 0 of 40 |
| technician prompt + similarity gate | 0 of 40 |

The similarity gate (decline without calling the model when the best chunk scores below 0.719, the value that lets
95% of dev answerable questions through) on its own declines 95% of the unanswerable dev questions and 85% of the
unanswerable test questions, at the cost of wrongly declining 5% and 3.3% of the answerable ones. In this
evaluation the prompt instruction alone already declined everything, so the gate adds nothing to the count and
costs some false declines; its practical value is skipping a model call (about 7 s) for questions with no good
passage. Near-topic questions (a Siemens breaker, a Woodward governor) are where the similarity score is least
useful: their median top score is 0.68 against 0.80 for answerable questions and 0.48 for off-topic ones.
The 12.5% baseline rate is small and rests on 5 judgments, so the size of any improvement is poorly determined.

### Diagrams (LLaVA 7B)

13 questions about figures. Text-only: 10 of 13 correct. With a LLaVA reading of the retrieved figure page
added to the prompt: 10 of 13. One question was fixed and one broke, so the vision step makes no measurable
difference here. Many figures in these PDFs carry their labels as text, so the text path already sees them;
LLaVA also misreads simple connections (it described a "Test Open" lead as going to "coil 2"). The step is wired
in and switched on for questions that mention a diagram, figure or wiring, but this test does not show it helping.

### Latency

25 consecutive questions through the API on an Apple M3 with 16 GB while other applications were open:
median **7.5 s**, 95th percentile **11.1 s** per answer, of which retrieval is about 21 ms (median). Almost all of
the time is the 8B model generating. `/metrics` reports this live.

## API

```bash
python scripts/fetch_manuals.py
python scripts/build_index.py            # builds data/index/default with the configuration chosen above
ollama pull llama3.1:8b && ollama pull llava:7b
uvicorn rag.service.api:app --port 8000
streamlit run app/streamlit_app.py       # chat page
```

| endpoint | |
|---|---|
| `POST /ask` `{"question": "...", "k": 4}` | answer, `sources` (manual, page, similarity, snippet), `cited_pages`, `declined`, `timings_ms` |
| `GET /health` | index size, models, whether Ollama is reachable |
| `GET /metrics` | request, error and decline counts; p50/p95 of total, retrieval and generation time over the last 500 requests |
| `GET /metrics/prometheus` | the same counters in Prometheus text format |

Every response carries `X-Request-Id` and `X-Process-Time-ms` headers and each request is logged with its id.
Settings through environment variables: `RAG_PROMPT`, `RAG_THRESHOLD` (similarity gate, off by default),
`RAG_VISION=0|1`, `RAG_TEXT_MODEL`, `RAG_VISION_MODEL`, `OLLAMA_URL`, `RAG_INDEX_DIR`.

The Dockerfile and `docker-compose.yml` (Ollama + API + UI) are written but **have not been built or run**:
Docker is not installed on the machine this was developed on.

## Reproduce the evaluation

```bash
pip install -r requirements.txt
python scripts/fetch_manuals.py
python -m eval.run_retrieval             # about 9 min; embeds the corpus 6 times
python -m eval.run_prompts               # needs Ollama; slow, answers are cached in reports/gen_cache.jsonl
python -m eval.analyze
python -m eval.run_guardrail && python -m eval.summarize_guardrail
python -m eval.run_vision
pytest                                   # 23 tests, no model downloads
```

## Limits

- One corpus of four related manuals, one language model, one run per configuration at temperature 0.
- 60 test questions give wide intervals. A difference of a few points between configurations is not evidence.
- The answerable questions were written from single pages, which makes them easier than real multi-page questions.
- Scoring is by keyword match, not by a human or a judge model.
- OpenMP is pinned to one thread in `rag/__init__.py`: torch and faiss each bundle an OpenMP runtime on macOS and
  crash when both run multi-threaded in one process.
- Page numbers are PDF page numbers, not the printed numbers in the page footer.

## Data

The FIST manuals are U.S. federal government publications of the Bureau of Reclamation Power Resources Office
(https://www.usbr.gov/power/data/fist_pub.html).
