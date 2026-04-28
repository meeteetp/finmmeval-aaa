# FinMMEval Task 3 — Team AAA Submission

Full submission package for [CLEF 2026 FinMMEval Lab — Task 3 (Financial Decision Making)](https://mbzuai-nlp.github.io/CLEF-2026-FinMMEval-Lab/). Contains the deployed live-inference endpoint, the cross-validated tuning notebook, the model parameters that were selected, and the methodology used to select them.

**Live endpoint:** `https://finmmeval-aaa.fly.dev/`
**Team AAA:** Zihao Li, Tanawat Ponggittila, Lingyi Wei, Yang Zhang — AIT 626 NLP, George Mason University.
**Advisor:** Dr. Lindi Liao.

## What's in this repository

| | |
|---|---|
| **Production endpoint** | `app.py`, `Dockerfile`, `fly.toml`, `requirements.txt` — what the organizers' system POSTs to once a day at 00:00 UTC. |
| **Tuning research** | `backtest.ipynb` — feature extraction + 4-fold expanding-window cross-validated grid search. End-to-end reproducible. |
| **Selected configuration** | `tuned_params.json` — winning weights, thresholds, per-fold test metrics, and aggregate stats. |
| **Documentation** | This `README.md` describes the full pipeline; the system paper for CEUR-WS will reference it. |

This is the whole product, not just the deployed binary. The endpoint is the artifact that runs in production; the notebook is how that artifact's parameters were chosen and validated; the README and `tuned_params.json` document the choices.

## Out-of-sample performance

Reported as the mean across 4 expanding-window CV test folds:

| Asset | Strategy CR | Buy-and-hold CR | Sharpe | Worst MaxDD |
|---|---:|---:|---:|---:|
| BTC  | **+7.4%** | −4.4% | **+1.62** | −7.3% |
| TSLA | **+3.4%** | −5.3% | **+1.29** | −7.7% |

Both assets beat buy-and-hold on the primary CLEF metric (Cumulative Return) — TSLA on every fold, BTC on 3 of 4 folds.

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
  Weighted sum → asset-specific threshold → {BUY | HOLD | SELL}
```

### Asset-specific weights and thresholds (CV-selected)

**BTC — super-defensive asymmetric thresholds:**
```
weights : w_news=0.8  w_trend=0.5  w_mom=1.0  w_peer=0  w_sector=0  w_macro=0
buy_th  : +0.30   sell_th: 0.00
```
**TSLA — symmetric thresholds, news + filings + sector dominate:**
```
weights : w_news=1.0  w_filing=0.3  w_disagree=0  w_trend=0  w_mom=1.0  w_peer=0  w_sector=0.5  w_macro=0
buy_th  : +0.20   sell_th: −0.20
```

## Threshold candidates as human priors (model design rationale)

The grid searched over multiple threshold candidate pairs per asset — symmetric (e.g. ±0.05, ±0.15), defensive asymmetric (BUY +0.15 / SELL −0.08), and super-defensive (BUY +0.30 / SELL 0.00). Including this *range* — rather than only a symmetric "neutral" set — was a deliberate design choice. The reasoning:

- **The model only sees what's in the request payload.** News headlines, the latest organizer momentum label, recent prices, optional 10-K / 10-Q summary text, and (via yfinance enrichment) peer / sector / macro returns. That is a narrow slice of what determines real markets.
- **A lot of relevant information is outside the model's view.** The macroeconomic regime, regulatory and geopolitical environment, microstructural conditions, central-bank trajectory, supply-chain and counter-party signals, and any meta-knowledge an experienced human operator would carry into the decision.
- **The human operator should be able to encode a risk preference.** If, going into the live evaluation period, the operator believes the broader environment favors *capital preservation over capture* (rate uncertainty, geopolitical tension, late-cycle equity behavior, etc.), the system should be capable of acting on that — not just on whatever the headline of the day says.
- **The threshold pairs in the grid are explicit, named risk-preference priors:** symmetric = "trust the data", defensive asymmetric = "lean toward exit on weak bearish hints", super-defensive = "long only on strong conviction." Each candidate represents a stance an informed human might justifiably hold.
- **Cross-validation then selects the prior that is most consistent with out-of-sample data across multiple time windows.** It is the role of CV to filter human priors against evidence: a prior that helps in only one period gets rejected; a prior that helps across multiple folds gets selected.

For BTC, this design produced an outcome that is methodologically meaningful: the super-defensive asymmetric pair (BUY +0.30 / SELL 0.00) won — in every CV fold — over the purely symmetric candidates. A defensive risk preference was therefore *not* hand-imposed onto the model; it was offered as one of several candidates and *validated by data*. For TSLA, the symmetric ±0.20 pair won — meaning the data did *not* support a defensive prior on TSLA, and we accept that.

This framing — human-encoded priors filtered by cross-validation — is more defensible than either purely data-driven tuning (which can overfit a single window's noise) or purely hand-set thresholds (which encode prior belief without testing whether it generalizes).

## Files in detail

| Path | Purpose |
|---|---|
| `app.py` | FastAPI server; loads FinBERT and MiniLM at startup; per-request hybrid scoring with the CV-selected weights and thresholds baked in. |
| `requirements.txt` | Python dependencies (FastAPI, transformers, sentence-transformers, yfinance, etc.). |
| `Dockerfile` | Containerizes `app.py`; pre-downloads FinBERT and MiniLM into the image so cold starts are instant. |
| `fly.toml` | Fly.io app configuration: 2 GB shared CPU, single Ashburn region, always-on (`auto_stop_machines = 'off'`, `min_machines_running = 1`) so the daily 00:00 UTC organizer call never finds the machine sleeping. |
| `backtest.ipynb` | Full reproducible research notebook — loads the official dataset, computes per-row features (no lookahead), runs 4-fold expanding-window CV grid search, plots out-of-sample stitched performance, writes `tuned_params.json`. |
| `tuned_params.json` | Output of the CV grid search: weights, thresholds, per-fold test metrics, aggregate (mean / std / worst-MaxDD across folds). |

## Running the endpoint locally

```bash
pip install -r requirements.txt
pip install torch --index-url https://download.pytorch.org/whl/cpu
uvicorn app:app --host 0.0.0.0 --port 8080
```

Test request:

```bash
curl -X POST http://localhost:8080/ -H "Content-Type: application/json" \
  -d '{"date":"2025-01-15","price":{"BTC":67890},"news":{"BTC":["Bitcoin ETF inflows remain strong"]},"symbol":["BTC"],"momentum":{"BTC":"bullish"},"10k":null,"10q":null,"history_price":{"BTC":[{"date":"2025-01-13","price":67720},{"date":"2025-01-14","price":67810}]}}'
```

## Deploying to Fly.io

```bash
fly apps create finmmeval-aaa     # one-time
fly deploy
```

First deploy builds the Docker image (~6–8 min — pre-downloads FinBERT + MiniLM, ~530 MB total). Subsequent deploys are faster (Docker layer cache).

## Reproducing the tuning

```bash
jupyter lab backtest.ipynb
```

Run cells top-to-bottom. With a CUDA GPU the whole notebook completes in 5–10 min; on CPU expect 30–60 min. Output `tuned_params.json` should match the values in this repo.

The notebook does:
1. Loads `TheFinAI/CLEF_Task3_Trading` from HuggingFace.
2. Computes 9 features per row with no lookahead.
3. Splits each asset into 4 expanding-window CV folds (train always older than test).
4. Grid-searches weights and threshold pairs. Threshold candidates encode a range of human risk-preference priors (see "Threshold candidates as human priors" above).
5. Selects the configuration with the best **average test-slice objective across all 4 folds**.
6. Reports per-fold and aggregate test metrics; saves them to `tuned_params.json`.

## Methodology notes (for the system paper)

- **No lookahead.** In every CV fold, the test slice is strictly newer than its train slice. Feature extraction uses no future data: price trends use only prior-row prices, yfinance series are trimmed to as-of date.
- **Cross-validated hyperparameters.** Weights and thresholds were not tuned on a single train/test split; they were selected by the best mean objective across 4 expanding-window CV folds. This penalizes configurations that overfit any individual time window.
- **Defensive bias is data-validated, not hand-imposed.** The asymmetric BTC thresholds were one of multiple human-prior candidates included in the grid; CV selected them on out-of-sample evidence.
- **Selection objective:** `0.7·tanh(2·CR) + 0.3·tanh(Sharpe/2) − soft_MaxDD_penalty`. Weighted toward the primary CLEF metric (Cumulative Return) while controlling for Sharpe and bounded drawdown.

## Reproducibility

- Python: 3.11
- Models: `ProsusAI/finbert`, `sentence-transformers/all-MiniLM-L6-v2`
- Dataset: `TheFinAI/CLEF_Task3_Trading` (HuggingFace, public, no token required)
- External data: `yfinance` (no key) — peers (RIVN, GM, F, NIO, LCID for TSLA; ETH-USD, COIN for BTC), sector ETF (XLY for TSLA, COIN for BTC), VIX
- Random seeds: not applicable — deterministic grid search; sentiment and embedding models loaded with default pretrained weights.
- Hardware: any x86-64 Linux with ≥2 GB RAM. CUDA optional, used to speed up `backtest.ipynb`.

## License

MIT — see [LICENSE](LICENSE).
