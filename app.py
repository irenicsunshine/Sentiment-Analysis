#!/usr/bin/env python3
"""
Advanced Sentiment Analysis Web Application
Flask backend with multi-backend SOTA engine.
"""

import os
import sys
import json
import logging
from datetime import datetime

import pandas as pd
import matplotlib
matplotlib.use("Agg")  # headless backend
import matplotlib.pyplot as plt
import io
import base64

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

# Global engine & history
_engine = None
analysis_history = []

MAX_TEXT_LENGTH = 5_000  # characters
MAX_BATCH_SIZE = 50


def get_engine():
    """Lazy-load and cache the sentiment engine."""
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
        "history_count": len(analysis_history),
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
        result["timestamp"] = datetime.now().isoformat()

        # Store lightweight record in history
        analysis_history.append({
            "text": text[:200],
            "sentiment": result["sentiment"],
            "sentiment_score": result["sentiment_score"],
            "confidence": result["confidence"],
            "intensity": result["intensity"],
            "emotions": result["emotions"],
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

    # Sanitize
    texts = [(t[:MAX_TEXT_LENGTH] if isinstance(t, str) else "") for t in texts]

    try:
        engine = get_engine()
        results = engine.analyze_batch(texts)
        for r in results:
            r["timestamp"] = datetime.now().isoformat()
        return jsonify({"results": results, "count": len(results)})
    except Exception as exc:
        logger.exception("Batch analysis error")
        return jsonify({"error": "Batch analysis failed", "detail": str(exc)}), 500


@app.route("/history")
def history():
    limit = min(int(request.args.get("limit", 100)), 500)
    return jsonify(analysis_history[-limit:])


@app.route("/stats")
def stats():
    if not analysis_history:
        return jsonify({
            "total_analyses": 0,
            "sentiment_counts": {"positive": 0, "negative": 0, "neutral": 0},
            "average_confidence": 0,
            "average_score": 0,
            "chart_url": None,
        })

    pos = sum(1 for a in analysis_history if a["sentiment"] == "positive")
    neg = sum(1 for a in analysis_history if a["sentiment"] == "negative")
    neu = sum(1 for a in analysis_history if a["sentiment"] == "neutral")
    avg_conf = sum(a["confidence"] for a in analysis_history) / len(analysis_history)
    avg_score = sum(a["sentiment_score"] for a in analysis_history) / len(analysis_history)

    chart_url = _create_stats_chart()

    return jsonify({
        "total_analyses": len(analysis_history),
        "sentiment_counts": {"positive": pos, "negative": neg, "neutral": neu},
        "average_confidence": round(avg_conf, 2),
        "average_score": round(avg_score, 4),
        "chart_url": chart_url,
    })


@app.route("/export_csv")
def export_csv():
    """Download analysis history as CSV."""
    if not analysis_history:
        return jsonify({"error": "No history to export"}), 404

    df = pd.DataFrame(analysis_history)
    csv_data = df.to_csv(index=False)
    return Response(
        csv_data,
        mimetype="text/csv",
        headers={"Content-Disposition": "attachment; filename=sentiment_history.csv"},
    )


@app.route("/clear_history", methods=["POST"])
def clear_history():
    global analysis_history
    analysis_history = []
    return jsonify({"message": "History cleared"})


# ─── Chart helper ─────────────────────────────────────────────────────────────

def _create_stats_chart() -> str | None:
    try:
        df = pd.DataFrame(analysis_history)

        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        fig.patch.set_facecolor("#1e1e2e")

        # Pie chart
        ax1 = axes[0]
        counts = df["sentiment"].value_counts()
        colors = {
            "positive": "#a6e3a1",
            "negative": "#f38ba8",
            "neutral": "#cdd6f4",
        }
        pie_colors = [colors.get(s, "#cdd6f4") for s in counts.index]
        ax1.pie(
            counts.values,
            labels=counts.index,
            autopct="%1.1f%%",
            colors=pie_colors,
            textprops={"color": "#cdd6f4"},
        )
        ax1.set_title("Sentiment Distribution", color="#cdd6f4")
        ax1.set_facecolor("#1e1e2e")

        # Score histogram
        ax2 = axes[1]
        ax2.set_facecolor("#313244")
        ax2.hist(
            df["sentiment_score"],
            bins=20,
            color="#89b4fa",
            edgecolor="#1e1e2e",
            alpha=0.85,
        )
        ax2.axvline(x=0, color="#f38ba8", linestyle="--", alpha=0.8, linewidth=1.5)
        ax2.set_xlabel("Sentiment Score", color="#cdd6f4")
        ax2.set_ylabel("Frequency", color="#cdd6f4")
        ax2.set_title("Score Distribution", color="#cdd6f4")
        ax2.tick_params(colors="#cdd6f4")
        for spine in ax2.spines.values():
            spine.set_edgecolor("#45475a")

        plt.tight_layout()

        buf = io.BytesIO()
        plt.savefig(buf, format="png", bbox_inches="tight", facecolor="#1e1e2e")
        buf.seek(0)
        encoded = base64.b64encode(buf.getvalue()).decode()
        plt.close(fig)
        return encoded
    except Exception as exc:
        logger.warning("Chart creation failed: %s", exc)
        return None


# ─── Entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    port = int(os.getenv("PORT", 5001))
    debug = os.getenv("FLASK_DEBUG", "false").lower() in ("1", "true", "yes")

    logger.info("Starting Sentiment Analysis app on port %d …", port)
    # Warm up engine before first request
    get_engine()
    app.run(debug=debug, host="0.0.0.0", port=port)
