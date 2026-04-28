# FinMMEval Task 3 — Team 1 Hybrid Endpoint

Submission for [CLEF 2026 FinMMEval Lab — Task 3 (Financial Decision Making)](https://mbzuai-nlp.github.io/CLEF-2026-FinMMEval-Lab/).

A FastAPI endpoint that consumes the daily organizer payload (news, momentum, price history, optional 10-K / 10-Q for TSLA) and returns `BUY`, `HOLD`, or `SELL`. Combines FinBERT sentiment, MiniLM-based filing-news alignment, recent price trend, organizer momentum label, and yfinance peer / sector / macro signals into a per-asset weighted score with grid-search-tuned weights and thresholds.

**Team:** Zihao Li, Tanawat Ponggittila, Lingyi Wei, Yang Zhang — AIT 626, George Mason University.

## Architecture

```
Daily request (00:00 UTC)
        │
        ▼
   FastAPI /                   ┌────────────────────────────────────────┐
        │                      │ TSLA path                              │
        ├──────────────────────┤   • FinBERT sentiment per news item    │
        │                      │   • MiniLM embed news + filing chunks  │
        │                      │   • Materiality-weighted news sentiment│
        │                      │   • Filing baseline tone               │
        │                      └────────────────────────────────────────┘
        │                      ┌────────────────────────────────────────┐
        ├──────────────────────┤ Common signals                         │
        │                      │   • 3- and 7-day price trend           │
        │                      │   • Organizer momentum label           │
        │                      │   • yfinance peers (5d return, avg)    │
        │                      │   • yfinance sector ETF (5d return)    │
        │                      │   • yfinance macro = -ΔVIX (5d)        │
        │                      └────────────────────────────────────────┘
        ▼
  Weighted sum → threshold → {BUY | HOLD | SELL}
```

Weights and thresholds are per-asset, tuned via grid search on the official training dataset (`TheFinAI/CLEF_Task3_Trading`) against an objective heavily weighted toward Cumulative Return with a Sharpe Ratio component and a soft Max Drawdown penalty (the official primary and secondary metrics).

## Files

| Path | Purpose |
|---|---|
| `app.py` | FastAPI server with the hybrid scoring logic and tuned weights baked in. |
| `requirements.txt` | Python dependencies. |
| `Dockerfile` | Containerizes `app.py` and pre-downloads FinBERT and MiniLM into the image. |
| `fly.toml` | Fly.io app configuration (always-on, 2 GB shared CPU). |
| `backtest.ipynb` | Backtest + grid-search notebook used to tune `tuned_params.json`. |
| `tuned_params.json` | Output of the grid search — weights and thresholds dropped into `app.py`. |
| `extract_wrds_ibes.ipynb` | (Unused in current submission) WRDS IBES analyst-data extractor, kept for the optional future analyst-signal extension. |
| `extract_refinitiv.ipynb` | (Unused in current submission) Refinitiv RDP / Eikon analyst-data extractor. |

## Running locally

```bash
pip install -r requirements.txt
pip install torch --index-url https://download.pytorch.org/whl/cpu
uvicorn app:app --host 0.0.0.0 --port 8080
```

Then in another terminal:

```bash
curl -X POST http://localhost:8080/ -H "Content-Type: application/json" \
  -d '{"date":"2025-01-15","price":{"BTC":67890},"news":{"BTC":["Bitcoin ETF inflows remain strong"]},"symbol":["BTC"],"momentum":{"BTC":"bullish"},"10k":null,"10q":null,"history_price":{"BTC":[{"date":"2025-01-13","price":67720},{"date":"2025-01-14","price":67810}]}}'
```

## Deploying

```bash
fly apps create finmmeval-aaa
fly deploy
```

The first deploy builds the Docker image (~6–8 min — pre-downloads FinBERT + MiniLM, ~530 MB total). Subsequent deploys are faster.

## Reproducing the tuning

```bash
jupyter lab backtest.ipynb
```

Run cells top-to-bottom. The slow steps are the per-row feature extraction (FinBERT + MiniLM on news and filings) and the grid search. With a CUDA GPU, the whole notebook completes in <10 min; on CPU expect 30–45 min. Output `tuned_params.json` should match the values currently committed.

## Reproducibility

- Python: 3.11
- Models: `ProsusAI/finbert`, `sentence-transformers/all-MiniLM-L6-v2`
- Dataset: `TheFinAI/CLEF_Task3_Trading` (HuggingFace)
- External data: `yfinance` (no key) — peers, sector ETF, VIX
- Random seeds: not applicable — deterministic grid search; sentiment models loaded with default weights.
- Hardware: any x86-64 Linux with ≥2 GB RAM. CUDA optional, used only to speed up `backtest.ipynb`.

## License

MIT — see [LICENSE](LICENSE).
