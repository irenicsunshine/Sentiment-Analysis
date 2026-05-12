/* global React */

function Result({ data, history, tweaks }) {
  const s = data;
  const sentimentClass = s.sentiment;
  const word = s.sentiment.toUpperCase();
  const emotions = s.emotions || {};
  const topEmotion = Object.entries(emotions).sort((a, b) => b[1] - a[1])[0];
  const vis = tweaks?.scoreVis || 'bar';

  // Build highlighted text from word_scores + original text
  const highlighted = useMemo(() => {
    if (!s.text || !Array.isArray(s.word_scores) || s.word_scores.length === 0) return null;
    const spans = s.word_scores
      .filter(w => typeof w.start === 'number' && typeof w.end === 'number')
      .sort((a, b) => a.start - b.start);
    if (spans.length === 0) return null;
    const out = [];
    let cursor = 0;
    spans.forEach((ws, i) => {
      if (ws.start < cursor) return;
      if (ws.start > cursor) out.push(<span key={`t${i}`}>{s.text.slice(cursor, ws.start)}</span>);
      const cls = ws.score > 0.15 ? 'positive' : ws.score < -0.15 ? 'negative' : 'neutral';
      out.push(
        <mark key={`m${i}`} className={cls} title={`score ${fmt(ws.score, 2)}`}>
          {s.text.slice(ws.start, ws.end)}
          <span className="ws-score">{fmt(ws.score, 2)}</span>
        </mark>
      );
      cursor = ws.end;
    });
    if (cursor < s.text.length) out.push(<span key="end">{s.text.slice(cursor)}</span>);
    return out;
  }, [s.text, s.word_scores]);

  return (
    <div className="result">
      <div className="result-head">
        <div>
          <div className="label-mono" style={{ marginBottom: 8 }}>SENTIMENT · {s.intensity?.toUpperCase()}</div>
          <div className={`sentiment-word ${sentimentClass}`}>
            <em>{word.toLowerCase()}</em>.
          </div>
        </div>
        <div className="sentiment-meta">
          <div className="score-num">{fmt(s.sentiment_score, 3)}</div>
          <div className="intensity">Confidence · {pct(s.confidence)}</div>
        </div>
      </div>

      {/* Highlighted text */}
      {highlighted && (
        <div>
          <div className="subhead">
            <span>Token-level signal</span>
            <span className="badge">{s.word_scores.length} marked</span>
          </div>
          <div className="highlighted-text">{highlighted}</div>
        </div>
      )}

      {/* Score viz — switched by Tweaks */}
      <div>
        <div className="subhead">
          <span>Polarity</span>
          <span className="badge">{vis.toUpperCase()}</span>
        </div>
        {vis === 'bar' && <ScoreBar score={s.sentiment_score} sentiment={s.sentiment} />}
        {vis === 'donut' && <div style={{ display: 'flex', justifyContent: 'center' }}><ScoreDonut score={s.sentiment_score} sentiment={s.sentiment} /></div>}
        {vis === 'radial' && <div style={{ display: 'flex', justifyContent: 'center' }}><ScoreRadial score={s.sentiment_score} sentiment={s.sentiment} /></div>}
        {vis === 'sparkline' && <div style={{ display: 'flex', justifyContent: 'center' }}><ScoreSparkline history={history || []} score={s.sentiment_score} sentiment={s.sentiment} /></div>}
      </div>

      <div className="result-grid">
        {/* Emotions */}
        <div>
          <div className="subhead">
            <span>Emotions</span>
            {topEmotion && <span className="badge" style={{ color: 'var(--accent)' }}>{topEmotion[0].toUpperCase()}</span>}
          </div>
          <div className="emotion-list">
            {Object.entries(emotions).map(([name, v]) => (
              <div key={name} className={`emotion-row ${topEmotion && topEmotion[0] === name ? 'dominant' : ''}`}>
                <div className="name">{name}</div>
                <div className="bar"><div className="bar-fill" style={{ width: `${clamp(v, 0, 1) * 100}%` }}></div></div>
                <div className="pct">{Math.round((v || 0) * 100)}</div>
              </div>
            ))}
          </div>
        </div>

        {/* Aspects */}
        <div>
          <div className="subhead">
            <span>Aspects</span>
            {s.aspects && <span className="badge">{s.aspects.length}</span>}
          </div>
          {s.aspects && s.aspects.length > 0 ? (
            <div className="aspects">
              {s.aspects.map((a, i) => (
                <div key={i} className={`aspect-row ${a.sentiment}`}>
                  <div className="term">{a.term}</div>
                  <div className="score">{fmt(a.score, 2)}</div>
                </div>
              ))}
            </div>
          ) : (
            <div style={{ color: 'var(--muted)', fontSize: 12, fontFamily: 'var(--font-mono)' }}>No aspects extracted</div>
          )}
        </div>
      </div>

      {/* Sarcasm */}
      {s.sarcasm_score > 0.4 && (
        <div className="sarcasm-card">
          <div className="icon">!</div>
          <div className="copy">
            <div className="title">Sarcasm flag</div>
            <div className="desc">Irony patterns detected — surface meaning may invert intent.</div>
          </div>
          <div className="pct">{Math.round(s.sarcasm_score * 100)}%</div>
        </div>
      )}

      {/* Backends — only if enabled */}
      {tweaks?.showBackends && s.backends && (
        <div className="backends">
          {Object.entries(s.backends).map(([k, v]) => (
            <div key={k} className="backend">
              <div className="b-name">{k}</div>
              <div className={`b-vote ${v.sentiment}`}>{v.sentiment}</div>
              <div className="b-score">score · {fmt(v.score, 2)}</div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

Object.assign(window, { Result });
