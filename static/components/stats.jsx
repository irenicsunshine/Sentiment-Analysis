/* global React */

function Stats({ history }) {
  const counts = useMemo(() => {
    const c = { positive: 0, negative: 0, neutral: 0 };
    (history || []).forEach(h => { c[h.sentiment] = (c[h.sentiment] || 0) + 1; });
    return c;
  }, [history]);
  const total = counts.positive + counts.negative + counts.neutral;
  const avgScore = total === 0 ? 0 : (history || []).reduce((a, h) => a + h.sentiment_score, 0) / total;
  const avgConf  = total === 0 ? 0 : (history || []).reduce((a, h) => a + h.confidence, 0) / total;
  const sarcastic = (history || []).filter(h => h.sarcasm_score > 0.5).length;

  const trend = (history || []).map(h => ({ score: h.sentiment_score, ts: h.timestamp }));

  // top emotion across all
  const emoTotals = {};
  (history || []).forEach(h => {
    Object.entries(h.emotions || {}).forEach(([k, v]) => { emoTotals[k] = (emoTotals[k] || 0) + v; });
  });
  const sortedEmo = Object.entries(emoTotals).sort((a, b) => b[1] - a[1]);
  const emoMax = sortedEmo[0]?.[1] || 1;

  return (
    <div>
      <div className="screen-head">
        <div>
          <h1 className="screen-title">Stats <em>overview</em></h1>
          <div className="screen-sub">Live across {total} analyses</div>
        </div>
      </div>

      <div className="stats-grid">
        <div className="card stat-card">
          <div className="stat-label">Total</div>
          <div className="stat-value">{total}</div>
          <div className="stat-delta">analyses on record</div>
        </div>
        <div className={`card stat-card ${avgScore > 0.15 ? 'positive' : avgScore < -0.15 ? 'negative' : 'neutral'}`}>
          <div className="stat-label">Avg score</div>
          <div className="stat-value">{fmt(avgScore, 2)}</div>
          <div className="stat-delta">{avgScore > 0.15 ? 'leaning positive' : avgScore < -0.15 ? 'leaning negative' : 'roughly balanced'}</div>
        </div>
        <div className="card stat-card">
          <div className="stat-label">Avg confidence</div>
          <div className="stat-value">{Math.round(avgConf * 100)}<span style={{ fontSize: 24, color: 'var(--muted)' }}>%</span></div>
          <div className="stat-delta">ensemble agreement</div>
        </div>
        <div className="card stat-card">
          <div className="stat-label">Sarcasm flags</div>
          <div className="stat-value" style={{ color: 'var(--sarc)' }}>{sarcastic}</div>
          <div className="stat-delta">irony detected · {total > 0 ? Math.round(sarcastic / total * 100) : 0}%</div>
        </div>
      </div>

      <div className="stats-row2">
        <div className="card">
          <div className="card-header">Distribution</div>
          <div className="card-body">
            <div className="donut-wrap">
              <SentimentDonut counts={counts} />
              <div className="donut-legend">
                <div className="l-row">
                  <span className="swatch" style={{ background: 'var(--pos)' }}></span>
                  <span className="l-name">Positive</span>
                  <span className="l-pct">{total ? Math.round(counts.positive / total * 100) : 0}%</span>
                  <span className="l-count">· {counts.positive}</span>
                </div>
                <div className="l-row">
                  <span className="swatch" style={{ background: 'var(--neu)' }}></span>
                  <span className="l-name">Neutral</span>
                  <span className="l-pct">{total ? Math.round(counts.neutral / total * 100) : 0}%</span>
                  <span className="l-count">· {counts.neutral}</span>
                </div>
                <div className="l-row">
                  <span className="swatch" style={{ background: 'var(--neg)' }}></span>
                  <span className="l-name">Negative</span>
                  <span className="l-pct">{total ? Math.round(counts.negative / total * 100) : 0}%</span>
                  <span className="l-count">· {counts.negative}</span>
                </div>
              </div>
            </div>
          </div>
        </div>

        <div className="card">
          <div className="card-header">Score trend · last {trend.length}</div>
          <div className="card-body">
            <TrendChart data={trend} />
          </div>
        </div>
      </div>

      {sortedEmo.length > 0 && (
        <div className="card" style={{ marginTop: 'var(--gap)' }}>
          <div className="card-header">Emotion totals across history</div>
          <div className="card-body">
            <div className="emotion-list">
              {sortedEmo.map(([name, v], i) => (
                <div key={name} className={`emotion-row ${i === 0 ? 'dominant' : ''}`}>
                  <div className="name">{name}</div>
                  <div className="bar"><div className="bar-fill" style={{ width: `${(v / emoMax) * 100}%` }}></div></div>
                  <div className="pct">{v.toFixed(1)}</div>
                </div>
              ))}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

Object.assign(window, { Stats });
