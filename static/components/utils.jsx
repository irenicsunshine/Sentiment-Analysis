/* global React */
const { useState, useEffect, useRef, useMemo, useCallback } = React;

// ====== Utility: number formatting ======
const fmt = (n, d = 2) => (n >= 0 ? '+' : '') + n.toFixed(d);
const pct = (n) => Math.round(n * 100) + '%';
const round = (n, d = 2) => Math.round(n * Math.pow(10, d)) / Math.pow(10, d);
const clamp = (n, lo, hi) => Math.max(lo, Math.min(hi, n));

// ====== Sample texts ======
const SAMPLES = [
  "Oh wonderful, another monday morning meeting that could've been an email.",
  "The new espresso machine is genuinely transformative — pulls beautiful shots.",
  "Customer service was helpful but the wait was unreasonable and the resolution was partial.",
  "Just neutral observations about the quarterly earnings report.",
  "Absolutely thrilled with how this came together. Best decision in months.",
  "The interface is fine. The performance is bad. The pricing is criminal.",
];

const BATCH_SAMPLE = `The product quality has improved dramatically this year.
Shipping was slow but support made it right.
Honestly the worst experience I've had with any brand.
Service was acceptable. Nothing remarkable, nothing terrible.
Five stars. Will absolutely buy again.
The app crashes every time I try to checkout. Unusable.
Loved the packaging, felt premium and considered.
Could be better. Could be worse. It is what it is.`;

// ====== Mock fallback ======
function mockAnalyze(text) {
  const lower = text.toLowerCase();
  let score = 0;
  const positives = ['great','love','wonderful','amazing','beautiful','best','excellent','transformative','thrilled','perfect','helpful','five star','five stars','improved','loved','premium'];
  const negatives = ['bad','hate','terrible','worst','awful','crash','crashes','unusable','criminal','slow','unreasonable','wait','partial','disaster','disappointing'];
  positives.forEach(w => { if (lower.includes(w)) score += 0.22; });
  negatives.forEach(w => { if (lower.includes(w)) score -= 0.24; });
  score = clamp(score, -1, 1);
  const sentiment = score > 0.15 ? 'positive' : score < -0.15 ? 'negative' : 'neutral';
  const sarcasm = /oh (wonderful|great|amazing|perfect)|sure[,.]|^"yeah right"|/.test(lower) && lower.includes('monday') ? 0.78 : Math.random() * 0.3;
  // Token scoring — match positives/negatives word-by-word
  const word_scores = [];
  const re = /\b[\w']+\b/g;
  let m;
  while ((m = re.exec(text)) !== null) {
    const w = m[0].toLowerCase();
    let s = 0;
    if (positives.some(p => w === p || w.includes(p))) s = 0.6 + Math.random() * 0.3;
    else if (negatives.some(n => w === n || w.includes(n))) s = -(0.6 + Math.random() * 0.3);
    if (s !== 0) word_scores.push({ token: m[0], start: m.index, end: m.index + m[0].length, score: round(s, 2) });
  }
  return {
    sentiment, sentiment_score: round(score, 3),
    confidence: round(0.65 + Math.abs(score) * 0.3, 2),
    intensity: Math.abs(score) > 0.5 ? 'strong' : Math.abs(score) > 0.2 ? 'moderate' : 'mild',
    emotions: {
      joy: sentiment === 'positive' ? 0.6 + Math.random() * 0.3 : Math.random() * 0.2,
      sadness: sentiment === 'negative' ? 0.4 + Math.random() * 0.3 : Math.random() * 0.15,
      anger: sentiment === 'negative' ? 0.3 + Math.random() * 0.4 : Math.random() * 0.1,
      fear: Math.random() * 0.15,
      surprise: Math.random() * 0.25,
      disgust: sentiment === 'negative' ? 0.2 + Math.random() * 0.3 : Math.random() * 0.1,
    },
    sarcasm_score: round(sarcasm, 2),
    aspects: extractAspects(text, sentiment),
    word_scores,
    backends: {
      vader:  { sentiment, score: round(score + (Math.random() - 0.5) * 0.1, 2) },
      svm:    { sentiment, score: round(score + (Math.random() - 0.5) * 0.1, 2) },
      transformer: { sentiment, score: round(score + (Math.random() - 0.5) * 0.08, 2) },
    },
  };
}
function extractAspects(text, overall) {
  const nouns = text.match(/\b([A-Z]?[a-z]{4,})\b/g) || [];
  const uniq = [...new Set(nouns.map(n => n.toLowerCase()))].slice(0, 3);
  return uniq.map(t => ({
    term: t,
    sentiment: overall,
    score: round(((overall === 'positive' ? 0.3 : overall === 'negative' ? -0.3 : 0) + (Math.random() - 0.5) * 0.4), 2),
  }));
}

// ====== Claude analyzer ======
async function analyzeWithClaude(text) {
  const prompt = `You are a precise sentiment analysis engine. Analyze the following text and return ONLY a valid JSON object (no markdown, no commentary).

Text: ${JSON.stringify(text)}

Return this exact schema:
{
  "sentiment": "positive" | "neutral" | "negative",
  "sentiment_score": number from -1 to 1,
  "confidence": number from 0 to 1,
  "intensity": "mild" | "moderate" | "strong",
  "emotions": { "joy": 0-1, "sadness": 0-1, "anger": 0-1, "fear": 0-1, "surprise": 0-1, "disgust": 0-1 },
  "sarcasm_score": 0-1,
  "aspects": [ { "term": "<word/phrase>", "sentiment": "positive|neutral|negative", "score": -1 to 1 } ],
  "word_scores": [ { "token": "<exact word from input>", "score": -1 to 1 } ],
  "backends": {
    "vader": { "sentiment": "...", "score": -1 to 1 },
    "svm": { "sentiment": "...", "score": -1 to 1 },
    "transformer": { "sentiment": "...", "score": -1 to 1 }
  }
}

Include 2-5 aspects. For word_scores, list ONLY the 4-10 words/short phrases that most drove the sentiment (positive or negative), using the EXACT spelling/casing from the input. Skip neutral filler. Vary backend scores slightly to simulate ensemble disagreement.`;
  try {
    const raw = await window.claude.complete(prompt);
    const cleaned = raw.replace(/^```json\s*|\s*```$/g, '').trim();
    const json = JSON.parse(cleaned);
    // Resolve word_scores → character spans against the original text
    if (Array.isArray(json.word_scores)) {
      json.word_scores = json.word_scores.map(ws => {
        const idx = text.indexOf(ws.token);
        return idx === -1 ? null : { ...ws, start: idx, end: idx + ws.token.length };
      }).filter(Boolean);
    }
    return json;
  } catch (e) {
    console.warn('Claude analyze failed, falling back to mock', e);
    return mockAnalyze(text);
  }
}

// ====== Format a timestamp ======
function tsShort(d) {
  const dt = new Date(d);
  const now = new Date();
  const sameDay = dt.toDateString() === now.toDateString();
  return sameDay
    ? dt.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
    : dt.toLocaleDateString([], { month: 'short', day: 'numeric' }) + ' ' + dt.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
}

// ====== Export to window ======
Object.assign(window, {
  fmt, pct, round, clamp, SAMPLES, BATCH_SAMPLE,
  mockAnalyze, analyzeWithClaude, tsShort,
});
