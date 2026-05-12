/* global React */

function Batch({ onBatchAnalyze, results, loading, tweaks }) {
  const [text, setText] = useState('');
  const lines = text.split('\n').filter(l => l.trim().length > 0);
  const overLimit = lines.length > 50;

  const handleRun = () => {
    if (lines.length === 0 || overLimit) return;
    onBatchAnalyze(lines);
  };

  const downloadCSV = () => {
    if (!results || results.length === 0) return;
    const rows = [['#', 'text', 'sentiment', 'score', 'confidence', 'intensity', 'sarcasm']];
    results.forEach((r, i) => {
      rows.push([i + 1, JSON.stringify(r.text), r.sentiment, r.sentiment_score, r.confidence, r.intensity, r.sarcasm_score]);
    });
    const csv = rows.map(r => r.join(',')).join('\n');
    const blob = new Blob([csv], { type: 'text/csv' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url; a.download = 'sentiment_batch.csv'; a.click();
    URL.revokeObjectURL(url);
  };

  const summary = useMemo(() => {
    if (!results) return null;
    const c = { positive: 0, negative: 0, neutral: 0 };
    results.forEach(r => { c[r.sentiment] = (c[r.sentiment] || 0) + 1; });
    return c;
  }, [results]);

  return (
    <div>
      <div className="screen-head">
        <div>
          <h1 className="screen-title">Batch <em>mode</em></h1>
          <div className="screen-sub">Up to 50 texts · One per line · CSV export</div>
        </div>
        <div className="screen-meta">
          <button className="btn btn-ghost" onClick={() => setText(BATCH_SAMPLE)}>Load sample</button>
          <button className="btn btn-ghost" onClick={downloadCSV} disabled={!results || results.length === 0}>Export CSV</button>
        </div>
      </div>

      <div className="batch-grid">
        <div className="card batch-input">
          <div className="composer-meta">
            <div className="left">
              <span>Lines · <span style={{ color: overLimit ? 'var(--neg)' : 'var(--text)' }}>{lines.length}</span> / 50</span>
              <span className="seq">ONE PER LINE</span>
            </div>
            <button className="btn btn-primary" onClick={handleRun} disabled={lines.length === 0 || overLimit || loading}>
              {loading ? `Running ${results?.length || 0}/${lines.length}…` : 'Run batch'}
            </button>
          </div>
          <textarea
            placeholder={'Paste one text per line…\n\nThe new espresso machine is amazing.\nShipping was a disaster.\nMeh — average product.'}
            value={text}
            onChange={(e) => setText(e.target.value)}
          />
        </div>

        <div className="batch-results">
          <div className="composer-meta">
            <div className="left">
              <span>Results</span>
              {summary && (
                <span className="seq">
                  <span style={{ color: 'var(--pos)' }}>{summary.positive} +</span>{'  '}
                  <span style={{ color: 'var(--neu)' }}>{summary.neutral} ◐</span>{'  '}
                  <span style={{ color: 'var(--neg)' }}>{summary.negative} −</span>
                </span>
              )}
            </div>
          </div>
          <div style={{ maxHeight: 440, overflow: 'auto' }}>
            {(!results || results.length === 0) ? (
              <div className="batch-results-empty">
                {loading ? 'Awaiting results…' : 'Run a batch to see results'}
              </div>
            ) : (
              <table className="batch-table">
                <thead>
                  <tr>
                    <th>#</th>
                    <th>Text</th>
                    <th>Sentiment</th>
                    <th>Score</th>
                    <th>Conf</th>
                  </tr>
                </thead>
                <tbody>
                  {results.map((r, i) => (
                    <tr key={i}>
                      <td className="row-num">{String(i + 1).padStart(2, '0')}</td>
                      <td className="text-cell" title={r.text}>{r.text}</td>
                      <td className={`sent ${r.sentiment}`}>
                        <span className="batch-spark"><span className={`fill ${r.sentiment}`} style={{
                          width: `${Math.abs(r.sentiment_score) * 50}%`,
                          left: r.sentiment_score >= 0 ? '50%' : `${50 + r.sentiment_score * 50}%`
                        }}></span></span>{' '}{r.sentiment.slice(0, 3)}
                      </td>
                      <td className="score-cell">{fmt(r.sentiment_score, 2)}</td>
                      <td className="score-cell">{pct(r.confidence)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}

Object.assign(window, { Batch });
