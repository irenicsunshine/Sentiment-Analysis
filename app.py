#!/usr/bin/env python3
"""
Advanced Sentiment Analysis Web Application
Flask backend with multi-backend SOTA engine + Neon Postgres history.
"""

import os
import sys
import json
import logging
from datetime import datetime
from contextlib import contextmanager

import csv
import io
import psycopg2
import psycopg2.extras

from flask import Flask, render_template, request, jsonify, Response

# ─── Path setup ───────────────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ─── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# ─── App setup ────────────────────────────────────────────────────────────────
app = Flask(__name__)

_engine = None
MAX_TEXT_LENGTH = 5_000
MAX_BATCH_SIZE  = 50

DATABASE_URL = os.getenv("DATABASE_URL")

# Run on every cold start (CREATE TABLE IF NOT EXISTS is safe to repeat)
try:
    if DATABASE_URL:
        import psycopg2 as _pg
        _c = _pg.connect(DATABASE_URL)
        _cur = _c.cursor()
        _cur.execute("""
            CREATE TABLE IF NOT EXISTS analysis_history (
                id              SERIAL PRIMARY KEY,
                text            TEXT,
                sentiment       VARCHAR(20),
                sentiment_score FLOAT,
                confidence      FLOAT,
                intensity       VARCHAR(50),
                emotions        TEXT,
                sarcasm_score   FLOAT,
                timestamp       TIMESTAMPTZ DEFAULT NOW()
            );
        """)
        _c.commit()
        _c.close()
except Exception as _e:
    logger.warning("DB init skipped: %s", _e)


# ─── DB helpers ───────────────────────────────────────────────────────────────

@contextmanager
def get_db():
    """Open a short-lived Postgres connection, commit & close on exit."""
    conn = psycopg2.connect(DATABASE_URL)
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db():
    """Create history table if it doesn't exist yet."""
    if not DATABASE_URL:
        return
    with get_db() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS analysis_history (
                    id              SERIAL PRIMARY KEY,
                    text            TEXT,
                    sentiment       VARCHAR(20),
                    sentiment_score FLOAT,
                    confidence      FLOAT,
                    intensity       VARCHAR(50),
                    emotions        TEXT,
                    sarcasm_score   FLOAT,
                    timestamp       TIMESTAMPTZ DEFAULT NOW()
                );
            """)
    logger.info("DB initialised")


def db_insert(record: dict):
    if not DATABASE_URL:
        return
    with get_db() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO analysis_history
                    (text, sentiment, sentiment_score, confidence, intensity, emotions, sarcasm_score, timestamp)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            """, (
                record["text"],
                record["sentiment"],
                record["sentiment_score"],
                record["confidence"],
                record["intensity"],
                json.dumps(record.get("emotions", {})),
                record["sarcasm_score"],
                record["timestamp"],
            ))


def db_fetch(limit: int = 100) -> list:
    if not DATABASE_URL:
        return []
    with get_db() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute("""
                SELECT text, sentiment, sentiment_score, confidence,
                       intensity, emotions, sarcasm_score,
                       timestamp AT TIME ZONE 'UTC' AS timestamp
                FROM analysis_history
                ORDER BY timestamp DESC
                LIMIT %s
            """, (limit,))
            rows = cur.fetchall()
    result = []
    for row in rows:
        r = dict(row)
        r["emotions"] = json.loads(r["emotions"]) if r["emotions"] else {}
        r["timestamp"] = r["timestamp"].isoformat()
        result.append(r)
    return list(reversed(result))


def db_clear():
    if not DATABASE_URL:
        return
    with get_db() as conn:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM analysis_history;")


def db_stats() -> dict:
    if not DATABASE_URL:
        return {}
    with get_db() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT
                    COUNT(*) AS total,
                    AVG(confidence) AS avg_conf,
                    AVG(sentiment_score) AS avg_score,
                    SUM(CASE WHEN sentiment='positive' THEN 1 ELSE 0 END) AS pos,
                    SUM(CASE WHEN sentiment='negative' THEN 1 ELSE 0 END) AS neg,
                    SUM(CASE WHEN sentiment='neutral'  THEN 1 ELSE 0 END) AS neu
                FROM analysis_history;
            """)
            row = cur.fetchone()
    return {
        "total": row[0] or 0,
        "avg_conf": float(row[1] or 0),
        "avg_score": float(row[2] or 0),
        "pos": row[3] or 0,
        "neg": row[4] or 0,
        "neu": row[5] or 0,
    }


# ─── Engine ───────────────────────────────────────────────────────────────────

def get_engine():
    global _engine
    if _engine is None:
        from src.sentiment_engine import SentimentEngine
        _engine = SentimentEngine(
            use_ml_model=True,
            use_transformer=os.getenv("USE_TRANSFORMER", "0") in ("1", "true", "yes"),
        )
        logger.info("SentimentEngine ready. Backends: %s", _engine.available_backends)
    return _engine


# ─── Routes ───────────────────────────────────────────────────────────────────

@app.route("/")
def home():
    return render_template("index.html")


@app.route("/health")
def health():
    engine = get_engine()
    return jsonify({
        "status": "ok",
        "backends": engine.available_backends,
        "db": "connected" if DATABASE_URL else "none",
    })


@app.route("/analyze", methods=["POST"])
def analyze():
    data = request.get_json(silent=True) or {}
    text = (data.get("text") or "").strip()

    if not text:
        return jsonify({"error": "No text provided"}), 400
    if len(text) > MAX_TEXT_LENGTH:
        return jsonify({"error": f"Text too long (max {MAX_TEXT_LENGTH} characters)"}), 400

    try:
        engine = get_engine()
        result = engine.analyze(text)
        result["timestamp"] = datetime.utcnow().isoformat()

        db_insert({
            "text": text[:200],
            "sentiment": result["sentiment"],
            "sentiment_score": result["sentiment_score"],
            "confidence": result["confidence"],
            "intensity": result["intensity"],
            "emotions": result.get("emotions", {}),
            "sarcasm_score": result["sarcasm_score"],
            "timestamp": result["timestamp"],
        })

        return jsonify(result)
    except Exception as exc:
        logger.exception("Analysis error")
        return jsonify({"error": "Analysis failed", "detail": str(exc)}), 500


@app.route("/batch", methods=["POST"])
def batch_analyze():
    data = request.get_json(silent=True) or {}
    texts = data.get("texts", [])

    if not isinstance(texts, list) or not texts:
        return jsonify({"error": "Provide a JSON list under 'texts'"}), 400
    if len(texts) > MAX_BATCH_SIZE:
        return jsonify({"error": f"Max batch size is {MAX_BATCH_SIZE}"}), 400

    texts = [(t[:MAX_TEXT_LENGTH] if isinstance(t, str) else "") for t in texts]

    try:
        engine = get_engine()
        results = engine.analyze_batch(texts)
        for r in results:
            r["timestamp"] = datetime.utcnow().isoformat()
        return jsonify({"results": results, "count": len(results)})
    except Exception as exc:
        logger.exception("Batch analysis error")
        return jsonify({"error": "Batch analysis failed", "detail": str(exc)}), 500


@app.route("/history")
def history():
    limit = min(int(request.args.get("limit", 100)), 500)
    return jsonify(db_fetch(limit))


@app.route("/stats")
def stats():
    s = db_stats()
    if not s or s["total"] == 0:
        return jsonify({
            "total_analyses": 0,
            "sentiment_counts": {"positive": 0, "negative": 0, "neutral": 0},
            "average_confidence": 0,
            "average_score": 0,
            "chart_url": None,
        })

    return jsonify({
        "total_analyses": s["total"],
        "sentiment_counts": {"positive": s["pos"], "negative": s["neg"], "neutral": s["neu"]},
        "average_confidence": round(s["avg_conf"], 2),
        "average_score": round(s["avg_score"], 4),
        "chart_url": _create_stats_chart(s),
    })


@app.route("/export_csv")
def export_csv():
    rows = db_fetch(500)
    if not rows:
        return jsonify({"error": "No history to export"}), 404
    buf = io.StringIO()
    fields = ["text", "sentiment", "sentiment_score", "confidence", "intensity", "sarcasm_score", "timestamp"]
    writer = csv.DictWriter(buf, fieldnames=fields, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(rows)
    return Response(
        buf.getvalue(),
        mimetype="text/csv",
        headers={"Content-Disposition": "attachment; filename=sentiment_history.csv"},
    )


@app.route("/clear_history", methods=["POST"])
def clear_history():
    db_clear()
    return jsonify({"message": "History cleared"})


def _create_stats_chart(s: dict) -> None:
    # Chart removed — UI renders its own SVG donut from /history data
    return None


# ─── Entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    init_db()
    port  = int(os.getenv("PORT", 5001))
    debug = os.getenv("FLASK_DEBUG", "false").lower() in ("1", "true", "yes")
    logger.info("Starting Sentiment Analysis app on port %d …", port)
    get_engine()
    app.run(debug=debug, host="0.0.0.0", port=port)
