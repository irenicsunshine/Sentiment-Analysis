/* global React */

function History({ history, onClear, onOpen }) {
  const list = [...(history || [])].reverse();
  return (
    <div>
      <div className="screen-head">
        <div>
          <h1 className="screen-title">History <em>log</em></h1>
          <div className="screen-sub">{list.length} analyses · most recent first</div>
        </div>
        <div className="screen-meta">
          <button className="btn btn-ghost" onClick={onClear} disabled={list.length === 0}>Clear all</button>
        </div>
      </div>

      <div className="card">
        <div className="history-row" style={{ background: 'var(--bg-soft)', cursor: 'default', padding: '12px var(--pad)' }}>
          <div className="label-mono">Time</div>
          <div className="label-mono">Text</div>
          <div className="label-mono">Sentiment</div>
          <div className="label-mono">Score</div>
          <div className="label-mono">Conf</div>
        </div>
        {list.length === 0 ? (
          <div style={{ padding: '80px 24px', textAlign: 'center', color: 'var(--muted)', fontFamily: 'var(--font-mono)', fontSize: 11, letterSpacing: '0.08em', textTransform: 'uppercase' }}>
            No analyses yet
          </div>
        ) : list.map((h, i) => (
          <div key={i} className="history-row" onClick={() => onOpen && onOpen(h)}>
            <div className="ts">{tsShort(h.timestamp)}</div>
            <div className="snippet">{h.text}</div>
            <div className={`lbl ${h.sentiment}`}>{h.sentiment}</div>
            <div className="score-cell">{fmt(h.sentiment_score, 2)}</div>
            <div className="score-cell">{pct(h.confidence)}</div>
          </div>
        ))}
      </div>
    </div>
  );
}

Object.assign(window, { History });
