#!/usr/bin/env python3
"""
State-of-the-art Multi-Backend Sentiment Analysis Engine

Features:
  - Multi-backend inference: VADER (rule-based) + sklearn ML + optional Transformer
  - 3-class sentiment: positive / neutral / negative
  - 6-emotion detection: joy, sadness, anger, fear, surprise, disgust
  - Aspect-based sentiment analysis
  - Sarcasm detection heuristics
  - Batch processing
  - LRU result caching
  - Graceful degradation (works even with only VADER)
  - Confidence-weighted ensemble fusion
"""

import os
import re
import sys
import logging
from functools import lru_cache
from typing import Dict, List, Optional
import numpy as np

logger = logging.getLogger(__name__)

# ─── Curated Emotion Lexicon (NRC-style) ──────────────────────────────────────
EMOTION_LEXICON: Dict[str, set] = {
    "joy": {
        "happy", "happiness", "joy", "love", "wonderful", "fantastic", "amazing",
        "excellent", "beautiful", "delight", "delightful", "pleasure", "pleasurable",
        "excited", "exciting", "thrilled", "thrilling", "elated", "cheerful", "glad",
        "smile", "laugh", "fun", "enjoy", "celebrate", "celebration", "great",
        "perfect", "awesome", "brilliant", "superb", "magnificent", "bliss",
        "ecstatic", "euphoric", "jubilant", "overjoyed", "radiant", "warm",
        "heartwarming", "content", "satisfied", "grateful", "thankful", "lucky",
        "blessed", "glorious", "marvelous", "sensational", "extraordinary",
        "charming", "splendid", "divine", "heavenly", "adore", "cherish"
    },
    "sadness": {
        "sad", "sadness", "depressed", "depression", "miserable", "unhappy", "sorrow",
        "sorrowful", "grief", "grieving", "tears", "cry", "crying", "lonely", "loneliness",
        "loss", "hurt", "pain", "heartbroken", "disappointed", "disappointment",
        "despair", "despairing", "gloomy", "melancholy", "unfortunate", "tragic",
        "tragedy", "awful", "horrible", "terrible", "suffering", "suffer", "anguish",
        "misery", "wretched", "hopeless", "helpless", "defeated", "broken", "crushed",
        "devastated", "forlorn", "mournful", "bereaved", "distressed", "troubled",
        "regret", "remorse", "guilt", "shame", "embarrassed", "failure", "failed"
    },
    "anger": {
        "angry", "anger", "rage", "furious", "fury", "mad", "hate", "hatred",
        "outrage", "outraged", "annoyed", "annoying", "frustrated", "frustration",
        "hostile", "hostility", "aggressive", "aggression", "violent", "violence",
        "attack", "blame", "terrible", "horrible", "infuriating", "infuriated",
        "livid", "seething", "bitter", "resentful", "resentment", "contempt",
        "disgusted", "indignant", "irate", "wrathful", "wrath", "ferocious",
        "vengeful", "enraged", "incensed", "irritated", "irritation", "furious",
        "outrageous", "abusive", "cruel", "harsh", "unfair", "wrong", "ridiculous"
    },
    "fear": {
        "fear", "fearful", "afraid", "scared", "terror", "terrified", "horrified",
        "horror", "anxious", "anxiety", "nervous", "nervousness", "worried", "worry",
        "panic", "dread", "dreading", "nightmare", "danger", "dangerous", "threat",
        "threatening", "scary", "frightening", "frightened", "alarming", "alarmed",
        "dreadful", "creepy", "eerie", "sinister", "ominous", "petrified", "phobia",
        "paranoid", "uneasy", "apprehensive", "tense", "shaky", "vulnerable",
        "insecure", "unstable", "risky", "hazardous", "shocking", "disturbing"
    },
    "surprise": {
        "surprised", "surprise", "shock", "shocked", "amazed", "astonished",
        "astonishment", "unexpected", "unbelievable", "incredible", "wow",
        "sudden", "stunning", "stunned", "remarkable", "extraordinary", "astounding",
        "astounded", "bewildered", "baffled", "speechless", "dumbfounded",
        "flabbergasted", "startled", "staggering", "breathtaking", "phenomenal",
        "unimaginable", "unforeseeable", "shocking", "revelation", "discovered",
        "unforeseen", "unanticipated", "spontaneous", "impromptu", "out-of-nowhere"
    },
    "disgust": {
        "disgusting", "disgust", "gross", "revolting", "vile", "nasty",
        "repulsive", "offensive", "horrible", "sickening", "nauseating", "loathe",
        "loathing", "abhor", "abhorrent", "detest", "detestable", "repelled",
        "awful", "foul", "putrid", "rotten", "filthy", "dirty", "hideous",
        "repugnant", "obnoxious", "objectionable", "odious", "abominable",
        "distasteful", "unpleasant", "appalling", "reprehensible", "corrupt",
        "immoral", "shameful", "perverse", "vulgar", "obscene", "indecent"
    }
}

# Slang phrase overrides — compound scores that bypass VADER's word-level lexicon.
# Keyed by lowercase phrase substring; first match wins.
# Compound range: -1.0 (very negative) → +1.0 (very positive)
SLANG_OVERRIDES: Dict[str, float] = {
    # ── Positive slang ────────────────────────────────────────────
    "fuck yeah": 0.85,
    "fucking awesome": 0.90,
    "fucking love": 0.82,
    "fucking brilliant": 0.85,
    "fucking great": 0.82,
    "so fucking good": 0.82,
    "damn good": 0.72,
    "hell yeah": 0.80,
    "holy shit this is amazing": 0.88,
    "holy crap that was awesome": 0.82,
    "shit this is great": 0.78,
    "wtf this is amazing": 0.82,
    "this is sick": 0.65,
    "this slaps": 0.70,
    "absolute banger": 0.72,
    "goated": 0.82,
    "bussin": 0.70,
    "no cap this is fire": 0.78,
    "lowkey obsessed": 0.62,
    "this hits different": 0.65,
    "ate and left no crumbs": 0.75,
    "understood the assignment": 0.68,
    "big w": 0.70,
    "certified w": 0.72,
    "slay": 0.60,
    "insane in the best way": 0.72,
    "badass": 0.65,
    # ── Negative slang ────────────────────────────────────────────
    "what the fuck": -0.45,
    "this is bullshit": -0.75,
    "piece of shit": -0.80,
    "fucking terrible": -0.85,
    "fucking awful": -0.85,
    "holy shit this is bad": -0.80,
    "this sucks ass": -0.75,
    "this is trash": -0.72,
    "dogshit": -0.80,
    "mid af": -0.40,
    "big l": -0.65,
    "fell off": -0.50,
    "flopped": -0.55,
    "pick me behavior": -0.50,
}

# Sarcasm indicator patterns (regex)
SARCASM_PATTERNS = [
    r"\byeah right\b", r"\boh sure\b", r"\bas if\b", r"\bright\.\.\.",
    r"\boh really\b", r"\bwow thanks\b", r"\btotally not\b",
    r"\bso funny\b.*\bnot\b", r"\bnice try\b", r"\bgreat job\b.*\bnot\b",
    r"\bnot\b.*\bgreat\b", r"\bwonderful\b.*\bsaid no one\b",
    r"\bsure\b.*\bthat('s| is) going to work\b",
]


class SentimentEngine:
    """
    Multi-backend sentiment analysis engine with ensemble fusion.

    Backends (in priority order for accuracy):
      1. Transformer  — HuggingFace pipeline, optional, ~500 MB
      2. ML model     — sklearn ensemble trained locally
      3. VADER        — rule-based, always available, great baseline

    All backends are optional; the engine degrades gracefully.
    """

    def __init__(
        self,
        use_transformer: bool = False,
        use_ml_model: bool = True,
        model_path: Optional[str] = None,
        fe_path: Optional[str] = None,
        transformer_model: str = "cardiffnlp/twitter-roberta-base-sentiment-latest",
    ):
        self.use_transformer = use_transformer or (
            os.getenv("USE_TRANSFORMER", "0").lower() in ("1", "true", "yes")
        )
        self.use_ml_model = use_ml_model

        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.model_path = model_path or os.path.join(base_dir, "models", "sentiment_model.pkl")
        self.fe_path = fe_path or os.path.join(base_dir, "models", "feature_extractor.pkl")
        self.transformer_model_name = transformer_model

        self._vader = None
        self._ml_model = None
        self._ml_fe = None
        self._ml_preprocessor = None
        self._transformer_pipeline = None

        self._init_vader()
        if use_ml_model:
            self._init_ml_model()
        if self.use_transformer:
            self._init_transformer()

    # ── Initializers ──────────────────────────────────────────────────────────

    def _init_vader(self):
        try:
            from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
            self._vader = SentimentIntensityAnalyzer()
            logger.info("VADER initialized")
        except ImportError:
            logger.warning("vaderSentiment not installed. Run: pip install vaderSentiment")

    def _init_ml_model(self):
        try:
            import joblib
            # Import relative to package root
            _parent = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            if _parent not in sys.path:
                sys.path.insert(0, _parent)
            from src.preprocess import TextPreprocessor

            if os.path.exists(self.model_path) and os.path.exists(self.fe_path):
                self._ml_model = joblib.load(self.model_path)
                self._ml_fe = joblib.load(self.fe_path)
                self._ml_preprocessor = TextPreprocessor(remove_stopwords=True, lemmatize=True)
                logger.info("ML model loaded from %s", self.model_path)
            else:
                logger.warning("ML model files not found — skipping ML backend")
        except Exception as exc:
            logger.warning("Could not load ML model: %s", exc)

    def _init_transformer(self):
        try:
            from transformers import pipeline as hf_pipeline
            logger.info("Loading transformer model '%s' (may download ~500 MB)…",
                        self.transformer_model_name)
            try:
                self._transformer_pipeline = hf_pipeline(
                    "sentiment-analysis",
                    model=self.transformer_model_name,
                    return_all_scores=True,
                    truncation=True,
                    max_length=512,
                )
                logger.info("Transformer initialized: %s", self.transformer_model_name)
            except Exception:
                # Fall back to default DistilBERT
                self._transformer_pipeline = hf_pipeline(
                    "sentiment-analysis",
                    return_all_scores=True,
                    truncation=True,
                    max_length=512,
                )
                logger.info("Transformer initialized: default DistilBERT")
        except ImportError:
            logger.warning("transformers not installed. Transformer backend disabled.")
        except Exception as exc:
            logger.warning("Transformer init failed: %s", exc)

    # ── Public API ─────────────────────────────────────────────────────────────

    def analyze(self, text: str, use_cache: bool = True) -> dict:
        """Analyze a single text. Returns comprehensive result dict."""
        if not text or not text.strip():
            return self._empty_result(text)
        text = text.strip()
        if use_cache:
            return self._analyze_cached(text)
        return self._analyze_impl(text)

    def analyze_batch(self, texts: List[str]) -> List[dict]:
        """Analyze a list of texts and return list of result dicts."""
        return [self.analyze(t) for t in texts]

    @property
    def available_backends(self) -> List[str]:
        out = []
        if self._vader:
            out.append("vader")
        if self._ml_model:
            out.append("ml")
        if self._transformer_pipeline:
            out.append("transformer")
        return out

    # ── Core Implementation ────────────────────────────────────────────────────

    @lru_cache(maxsize=512)
    def _analyze_cached(self, text: str) -> dict:
        return self._analyze_impl(text)

    def _analyze_impl(self, text: str) -> dict:
        backend_results: Dict[str, dict] = {}

        # Run each available backend
        if self._vader:
            r = self._analyze_vader(text)
            if r:
                backend_results["vader"] = r

        if self._ml_model and self._ml_fe and self._ml_preprocessor:
            r = self._analyze_ml(text)
            if r:
                backend_results["ml"] = r

        if self._transformer_pipeline:
            r = self._analyze_transformer(text)
            if r:
                backend_results["transformer"] = r

        # Ensemble fusion
        fused = self._ensemble_fusion(backend_results, text)

        # Enrichments
        fused["emotions"] = self._detect_emotions(text)
        fused["aspects"] = self._extract_aspects(text)
        fused["sarcasm_score"] = self._detect_sarcasm(
            text, backend_results.get("vader")
        )
        fused["backends_used"] = list(backend_results.keys())
        fused["text"] = text
        fused["word_count"] = len(text.split())
        fused["char_count"] = len(text)

        return fused

    # ── Per-Backend Analyzers ─────────────────────────────────────────────────

    def _analyze_vader(self, text: str) -> Optional[dict]:
        try:
            scores = self._vader.polarity_scores(text)
            c = scores["compound"]

            # Apply slang phrase overrides — VADER's word-level lexicon can't handle
            # idioms like "fuck yeah" (positive) where individual tokens are negative.
            text_lower = text.lower()
            for phrase, override_c in SLANG_OVERRIDES.items():
                if phrase in text_lower:
                    c = override_c
                    break

            sentiment = "positive" if c >= 0.05 else ("negative" if c <= -0.05 else "neutral")

            # Recompute probabilities from compound when overridden
            if sentiment == "positive":
                pos_p = round(min(0.5 + abs(c) * 0.45, 0.95), 3)
                neg_p = round(max(0.5 - abs(c) * 0.5, 0.02), 3)
                neu_p = round(max(1.0 - pos_p - neg_p, 0.02), 3)
            elif sentiment == "negative":
                neg_p = round(min(0.5 + abs(c) * 0.45, 0.95), 3)
                pos_p = round(max(0.5 - abs(c) * 0.5, 0.02), 3)
                neu_p = round(max(1.0 - pos_p - neg_p, 0.02), 3)
            else:
                pos_p = scores["pos"]
                neg_p = scores["neg"]
                neu_p = scores["neu"]

            return {
                "sentiment": sentiment,
                "score": c,
                "probabilities": {
                    "negative": neg_p,
                    "neutral": neu_p,
                    "positive": pos_p,
                },
            }
        except Exception as exc:
            logger.warning("VADER analysis failed: %s", exc)
            return None

    def _analyze_ml(self, text: str) -> Optional[dict]:
        try:
            cleaned = self._ml_preprocessor.clean_text(text)
            if not cleaned.strip():
                return None
            features = self._ml_fe.transform([cleaned])

            prediction = int(self._ml_model.predict(features)[0])

            if hasattr(self._ml_model, "predict_proba"):
                proba = self._ml_model.predict_proba(features)[0]
                neg_p, pos_p = float(proba[0]), float(proba[1])
            else:
                import scipy.special
                decision = self._ml_model.decision_function(features)
                pos_p = float(scipy.special.expit(decision.ravel()[0]))
                neg_p = 1.0 - pos_p

            # Push low-confidence predictions toward neutral
            max_p = max(pos_p, neg_p)
            neu_p = max(0.0, 0.6 - max_p) if max_p < 0.6 else 0.0
            if neu_p > 0:
                pos_p -= neu_p / 2
                neg_p -= neu_p / 2

            sentiment = "positive" if prediction == 1 else "negative"
            if neu_p > max(pos_p, neg_p):
                sentiment = "neutral"

            return {
                "sentiment": sentiment,
                "score": pos_p - neg_p,
                "probabilities": {
                    "negative": max(neg_p, 0.0),
                    "neutral": neu_p,
                    "positive": max(pos_p, 0.0),
                },
            }
        except Exception as exc:
            logger.warning("ML model analysis failed: %s", exc)
            return None

    def _analyze_transformer(self, text: str) -> Optional[dict]:
        try:
            raw = self._transformer_pipeline(text[:512])[0]
            label_map: Dict[str, float] = {}
            for item in raw:
                lbl = item["label"].lower()
                score = item["score"]
                if "pos" in lbl or lbl == "label_2":
                    label_map["positive"] = score
                elif "neg" in lbl or lbl == "label_0":
                    label_map["negative"] = score
                else:
                    label_map["neutral"] = score

            label_map.setdefault("positive", 0.0)
            label_map.setdefault("negative", 0.0)
            label_map.setdefault("neutral", 0.0)

            total = sum(label_map.values()) or 1.0
            label_map = {k: v / total for k, v in label_map.items()}

            sentiment = max(label_map, key=label_map.get)
            return {
                "sentiment": sentiment,
                "score": label_map["positive"] - label_map["negative"],
                "probabilities": label_map,
            }
        except Exception as exc:
            logger.warning("Transformer analysis failed: %s", exc)
            return None

    # ── Ensemble Fusion ────────────────────────────────────────────────────────

    def _ensemble_fusion(self, results: Dict[str, dict], text: str) -> dict:
        if not results:
            return self._empty_result(text)

        # Weights reflect typical accuracy order
        WEIGHTS = {"transformer": 0.50, "ml": 0.35, "vader": 0.15}

        available = {k: WEIGHTS[k] for k in WEIGHTS if k in results}
        total_w = sum(available.values()) or 1.0
        available = {k: v / total_w for k, v in available.items()}

        fused = {"positive": 0.0, "negative": 0.0, "neutral": 0.0}
        for backend, res in results.items():
            w = available.get(backend, 0.0)
            for label, prob in res["probabilities"].items():
                fused[label] = fused.get(label, 0.0) + prob * w

        sentiment = max(fused, key=fused.get)
        score = fused["positive"] - fused["negative"]
        confidence = max(fused.values()) * 100

        return {
            "sentiment": sentiment,
            "sentiment_score": round(score, 4),
            "confidence": round(confidence, 2),
            "intensity": self._score_to_intensity(score),
            "probabilities": {
                "positive": round(fused["positive"] * 100, 2),
                "negative": round(fused["negative"] * 100, 2),
                "neutral": round(fused["neutral"] * 100, 2),
            },
        }

    @staticmethod
    def _score_to_intensity(score: float) -> str:
        if score >= 0.6:
            return "Very Positive"
        if score >= 0.2:
            return "Positive"
        if score >= -0.2:
            return "Neutral"
        if score >= -0.6:
            return "Negative"
        return "Very Negative"

    # ── Emotion Detection ─────────────────────────────────────────────────────

    def _detect_emotions(self, text: str) -> Dict[str, float]:
        """
        Lexicon-based 6-emotion detection.
        Returns independent 0-100 scores per emotion (not normalized to sum=100).
        """
        words = re.findall(r"\b\w+\b", text.lower())  # allow duplicates for count
        word_set = set(words)
        n_words = max(len(words), 1)

        # Each emotion score = min(match_count / text_word_count * 200, 100)
        # This means ~1 match in 10 words → ~20%, caps at 100%
        raw: Dict[str, int] = {e: len(word_set & lex) for e, lex in EMOTION_LEXICON.items()}
        scores = {e: round(min(c / n_words * 200, 100.0), 1) for e, c in raw.items()}

        # Blend in VADER-based adjustments for joy/sadness/anger
        if self._vader:
            v = self._vader.polarity_scores(text)
            c = v["compound"]
            # Map VADER pos/neg subcores to emotion intensity
            if c > 0.2:
                scores["joy"] = round(min(max(scores["joy"], v["pos"] * 80), 100.0), 1)
            if c < -0.2:
                # Distribute negative compound among sadness/anger based on existing ratio
                neg_total = scores["sadness"] + scores["anger"] + scores["fear"] + 0.001
                for em in ("sadness", "anger", "fear"):
                    boost = (scores[em] / neg_total) * v["neg"] * 80
                    scores[em] = round(min(scores[em] + boost, 100.0), 1)

        return scores

    # ── Aspect-Based Sentiment ────────────────────────────────────────────────

    def _extract_aspects(self, text: str) -> List[dict]:
        """
        Sentence-level aspect extraction with per-aspect VADER scoring.
        """
        if not self._vader:
            return []

        sentences = re.split(r"[.!?;]", text)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 8]

        seen: set = set()
        aspects: List[dict] = []

        aspect_re = re.compile(
            r"\bthe ([\w]+(?: [\w]+)?)\b|"      # "the acting", "the story"
            r"\b([\w]+ing)\b|"                   # gerunds: "acting", "writing"
            r"\b([\w]+)\s+(?:was|is|were|are)\b" # "plot was", "cast is"
        )

        # Common stopwords / non-aspects to ignore
        ignore = {
            "the", "was", "is", "are", "were", "this", "that", "it",
            "being", "having", "nothing", "something", "anything", "everything"
        }

        for sentence in sentences[:6]:
            vader_score = self._vader.polarity_scores(sentence)["compound"]
            if vader_score >= 0.05:
                sent = "positive"
            elif vader_score <= -0.05:
                sent = "negative"
            else:
                sent = "neutral"

            for m in aspect_re.finditer(sentence.lower()):
                aspect = next((g for g in m.groups() if g), None)
                if not aspect:
                    continue
                aspect = aspect.strip()
                if len(aspect) < 3 or aspect in ignore or aspect in seen:
                    continue
                seen.add(aspect)
                aspects.append({
                    "aspect": aspect,
                    "sentiment": sent,
                    "score": round(vader_score, 3),
                })
                if len(aspects) >= 8:
                    break
            if len(aspects) >= 8:
                break

        return aspects

    # ── Sarcasm Detection ─────────────────────────────────────────────────────

    def _detect_sarcasm(self, text: str, vader_result: Optional[dict]) -> float:
        """
        Heuristic sarcasm probability (0–1).
        Signals: known phrases, sentiment-word contradiction, punctuation overuse.
        """
        text_lower = text.lower()
        score = 0.0

        for pattern in SARCASM_PATTERNS:
            if re.search(pattern, text_lower):
                score += 0.3

        if vader_result:
            compound = vader_result["score"]
            positive_words = len(re.findall(
                r"\b(amazing|fantastic|great|wonderful|excellent|perfect|awesome|brilliant|superb)\b",
                text_lower,
            ))
            if positive_words > 0 and compound < -0.2:
                score += 0.4
            elif positive_words >= 2 and compound < 0.0:
                score += 0.2

        if text.count("!") > 2:
            score += 0.1
        all_caps_words = sum(1 for w in text.split() if w.isupper() and len(w) > 1)
        if all_caps_words > 1:
            score += 0.1

        return min(round(score, 2), 1.0)

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _empty_result(text: str) -> dict:
        return {
            "text": text or "",
            "sentiment": "neutral",
            "sentiment_score": 0.0,
            "confidence": 0.0,
            "intensity": "Neutral",
            "probabilities": {"positive": 33.3, "negative": 33.3, "neutral": 33.4},
            "emotions": {e: 0.0 for e in EMOTION_LEXICON},
            "aspects": [],
            "sarcasm_score": 0.0,
            "backends_used": [],
            "word_count": 0,
            "char_count": 0,
        }
