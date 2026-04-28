import hashlib
import logging
import time
from datetime import timedelta

import numpy as np
import pandas as pd
import yfinance as yf
from fastapi import FastAPI, Request
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, pipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI()

# ----------------------------- model loading -----------------------------
logger.info("Loading FinBERT...")
_tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
_finbert = pipeline(
    "text-classification",
    model="ProsusAI/finbert",
    tokenizer=_tokenizer,
    top_k=None,
    truncation=True,
    max_length=512,
    device=-1,
)
logger.info("Loading MiniLM...")
_embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
logger.info("Models ready.")

# ----------------------------- tuned params -----------------------------
PARAMS = {
    "BTC": {
        "weights": {"w_news": 1.0, "w_trend": 1.0, "w_mom": 1.0,
                    "w_peer": 0.5, "w_sector": 0.0, "w_macro": 1.0},
        "buy_th": 0.15, "sell_th": -0.15,
    },
    "TSLA": {
        "weights": {"w_news": 0.6, "w_filing": 0.3, "w_disagree": 0.0,
                    "w_trend": 0.0, "w_mom": 0.5,
                    "w_peer": 0.0, "w_sector": 0.5, "w_macro": 0.0},
        "buy_th": 0.20, "sell_th": -0.20,
    },
}

PEERS = {"TSLA": ["RIVN", "GM", "F", "NIO", "LCID"], "BTC": ["ETH-USD", "COIN"]}
SECTOR = {"TSLA": "XLY", "BTC": "COIN"}
MACRO_TICKER = "^VIX"

# ----------------------------- caches -----------------------------
_filing_cache: dict = {}
_yf_cache: dict = {}
_YF_TTL = 6 * 3600  # 6 hours


# ----------------------------- helpers -----------------------------
def _chunk_text(text: str, max_tokens: int = 510):
    if not isinstance(text, str) or not text.strip():
        return []
    ids = _tokenizer.encode(text, add_special_tokens=False, truncation=False)
    chunks = []
    for i in range(0, len(ids), max_tokens):
        ch = _tokenizer.decode(ids[i:i + max_tokens], skip_special_tokens=True)
        if ch.strip():
            chunks.append(ch)
    return chunks


def _finbert_score(text: str) -> float:
    chunks = _chunk_text(text)
    if not chunks:
        return 0.0
    vals = []
    for ch in chunks:
        labels = {x["label"].lower(): x["score"] for x in _finbert(ch)[0]}
        vals.append(labels.get("positive", 0.0) - labels.get("negative", 0.0))
    return float(np.mean(vals))


def _embed(texts):
    if not texts:
        return np.zeros((0, 384), dtype=np.float32)
    return _embedder.encode(texts, normalize_embeddings=True, show_progress_bar=False)


def _split_filing_paragraphs(filing_strings, max_chars: int = 1500):
    paras = []
    for s in filing_strings:
        for chunk in str(s).split("\n\n"):
            chunk = chunk.strip()
            if len(chunk) < 80:
                continue
            for i in range(0, len(chunk), max_chars):
                paras.append(chunk[i:i + max_chars])
    return paras[:60]  # cap for speed


def _filing_signals(filing_strings):
    if not filing_strings:
        return None
    h = hashlib.md5(("||".join(map(str, filing_strings))).encode("utf-8")).hexdigest()
    if h in _filing_cache:
        return _filing_cache[h]
    paras = _split_filing_paragraphs(filing_strings)
    if not paras:
        _filing_cache[h] = None
        return None
    sentiments = np.array([_finbert_score(p) for p in paras])
    embeds = _embed(paras)
    out = {"sentiments": sentiments, "embeds": embeds, "baseline_tone": float(sentiments.mean())}
    _filing_cache[h] = out
    return out


def _price_trend(history) -> float:
    if not history or len(history) < 2:
        return 0.0
    try:
        h = sorted(history, key=lambda x: x["date"])
        last = float(h[-1]["price"])
        ref3 = float(h[-min(3, len(h))]["price"])
        ref7 = float(h[-min(7, len(h))]["price"])
        r3 = (last - ref3) / ref3 if ref3 > 0 else 0.0
        r7 = (last - ref7) / ref7 if ref7 > 0 else 0.0
        return float(np.clip(0.5 * r3 + 0.5 * r7, -0.15, 0.15))
    except Exception:
        return 0.0


def _yf_returns(tickers: tuple, asof_str: str, lookback: int = 5) -> float:
    """5-day return averaged across tickers, as of asof_str. 0.0 on any failure."""
    key = (tickers, asof_str, lookback)
    now = time.time()
    cached = _yf_cache.get(key)
    if cached and now - cached[0] < _YF_TTL:
        return cached[1]
    try:
        asof = pd.to_datetime(asof_str)
        end = (asof + timedelta(days=1)).strftime("%Y-%m-%d")
        start = (asof - timedelta(days=30)).strftime("%Y-%m-%d")
        df = yf.download(list(tickers), start=start, end=end,
                         progress=False, auto_adjust=True, threads=False)["Close"]
        if isinstance(df, pd.Series):
            df = df.to_frame()
        rets = []
        for t in tickers:
            if t in df.columns:
                s = df[t].dropna()
                if len(s) >= lookback + 1:
                    last, prior = s.iloc[-1], s.iloc[-lookback - 1]
                    if prior > 0:
                        rets.append((last - prior) / prior)
        val = float(np.mean(rets)) if rets else 0.0
    except Exception as e:
        logger.warning("yfinance fetch failed for %s: %s", tickers, e)
        val = 0.0
    _yf_cache[key] = (now, val)
    return val


def _peer_signal(symbol: str, asof_str: str) -> float:
    return _yf_returns(tuple(PEERS.get(symbol, [])), asof_str, 5)


def _sector_signal(symbol: str, asof_str: str) -> float:
    t = SECTOR.get(symbol)
    return _yf_returns((t,), asof_str, 5) if t else 0.0


def _macro_signal(asof_str: str) -> float:
    raw = -_yf_returns((MACRO_TICKER,), asof_str, 5)  # falling VIX = risk-on
    return float(np.clip(raw, -0.10, 0.10))


# ----------------------------- decision logic -----------------------------
def _predict(payload: dict) -> str:
    symbols = payload.get("symbol") or []
    symbol = symbols[0] if symbols else "BTC"
    asof = str(payload.get("date") or "")

    cfg = PARAMS.get(symbol, PARAMS["BTC"])
    w = cfg["weights"]

    news_map = payload.get("news") or {}
    news_list = [n for n in (news_map.get(symbol) or []) if isinstance(n, str) and n.strip()]
    news_sents = np.array([_finbert_score(n) for n in news_list[:10]]) if news_list else np.array([0.0])
    raw_news = float(news_sents.mean())

    weighted_news = raw_news
    filing_tone = 0.0
    disagreement = 0.0
    if symbol == "TSLA":
        filings = []
        for key in ("10k", "10q"):
            v = payload.get(key)
            if isinstance(v, dict):
                filings.extend([str(x) for x in (v.get(symbol) or []) if x])
            elif isinstance(v, list):
                filings.extend([str(x) for x in v if x])
        sig = _filing_signals(filings)
        if sig and news_list:
            news_emb = _embed(news_list[:10])
            sims = news_emb @ sig["embeds"].T
            mat = sims.max(axis=1).clip(0)
            if mat.sum() > 1e-6:
                weighted_news = float((mat * news_sents).sum() / mat.sum())
            filing_tone = sig["baseline_tone"]
            disagreement = filing_tone - weighted_news

    history = (payload.get("history_price") or {}).get(symbol) or []
    trend = _price_trend(history)

    mom = str((payload.get("momentum") or {}).get(symbol) or "neutral").lower()
    mom_adj = 0.05 if mom == "bullish" else (-0.05 if mom == "bearish" else 0.0)

    peer = _peer_signal(symbol, asof)
    sector = _sector_signal(symbol, asof)
    macro = _macro_signal(asof)

    if symbol == "BTC":
        score = (
            w["w_news"]   * raw_news
          + w["w_trend"]  * trend
          + w["w_mom"]    * mom_adj
          + w["w_peer"]   * peer
          + w["w_sector"] * sector
          + w["w_macro"]  * macro
        )
    else:  # TSLA
        score = (
            w["w_news"]      * weighted_news
          + w["w_filing"]    * filing_tone
          + w["w_disagree"]  * (-disagreement)
          + w["w_trend"]     * trend
          + w["w_mom"]       * mom_adj
          + w["w_peer"]      * peer
          + w["w_sector"]    * sector
          + w["w_macro"]     * macro
        )

    logger.info(
        "symbol=%s raw_news=%.3f weighted_news=%.3f filing=%.3f trend=%.3f "
        "peer=%.3f sector=%.3f macro=%.3f score=%.3f",
        symbol, raw_news, weighted_news, filing_tone, trend, peer, sector, macro, score,
    )

    if score >= cfg["buy_th"]:
        return "BUY"
    if score <= cfg["sell_th"]:
        return "SELL"
    return "HOLD"


# ----------------------------- routes -----------------------------
@app.post("/")
async def endpoint(request: Request):
    try:
        payload = await request.json()
        return {"recommended_action": _predict(payload)}
    except Exception as e:
        logger.error("Prediction error: %s", e)
        return {"recommended_action": "HOLD"}


@app.get("/health")
def health():
    return {"status": "ok"}
