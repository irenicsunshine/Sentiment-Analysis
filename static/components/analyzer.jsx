/* global React */

function Analyzer({ onAnalyze, loading, lastResult, history, tweaks }) {
  const [text, setText] = useState('');
  const [mode, setMode] = useState('paste'); // paste | guided
  const [live, setLive] = useState(false);
  const taRef = useRef(null);
  const liveTimer = useRef(null);
  const lastLiveText = useRef('');

  const charCount = text.length;
  const overLimit = charCount > 5000;
  const canAnalyze = text.trim().length > 0 && !loading && !overLimit;

  const handleAnalyze = () => {
    if (!canAnalyze) return;
    onAnalyze(text.trim());
  };
  const handleKey = (e) => {
    if ((e.metaKey || e.ctrlKey) && e.key === 'Enter') {
      e.preventDefault();
      handleAnalyze();
    }
  };
  const handleSample = (s) => {
    setText(s);
    setTimeout(() => taRef.current?.focus(), 0);
  };

  // Live mode — debounced auto-analyze
  useEffect(() => {
    if (!live) return;
    const trimmed = text.trim();
    if (!trimmed || trimmed.length < 6 || overLimit) return;
    if (trimmed === lastLiveText.current) return;
    clearTimeout(liveTimer.current);
    liveTimer.current = setTimeout(() => {
      lastLiveText.current = trimmed;
      onAnalyze(trimmed, { silent: true });
    }, 500);
    return () => clearTimeout(liveTimer.current);
  }, [text, live, overLimit, onAnalyze]);

  return (
    <div className="analyzer">
      {/* Composer */}
      <div className="composer card">
        <div className="composer-tabs">
          <button className={`composer-tab ${mode === 'paste' ? 'active' : ''}`} onClick={() => setMode('paste')}>Paste · 01</button>
          <button className={`composer-tab ${mode === 'guided' ? 'active' : ''}`} onClick={() => setMode('guided')}>Examples · 02</button>
        </div>

        <div className="composer-meta">
          <div className="left">
            <span>Input</span>
            <span className="seq">EN · UTF-8</span>
          </div>
          <div className="left">
            <span>Max 5,000</span>
          </div>
        </div>

        {mode === 'paste' ? (
          <textarea
            ref={taRef}
            className="composer-textarea"
            placeholder="Paste a review, tweet, message, comment…"
            value={text}
            onChange={(e) => setText(e.target.value)}
            onKeyDown={handleKey}
            autoFocus
          />
        ) : (
          <div style={{ padding: '8px var(--pad)', flex: 1, display: 'flex', flexDirection: 'column', gap: 8 }}>
            <div className="label-mono" style={{ marginBottom: 4 }}>Tap any line to load it</div>
            {SAMPLES.map((s, i) => (
              <button key={i}
                onClick={() => { handleSample(s); setMode('paste'); }}
                style={{
                  textAlign: 'left',
                  padding: '12px 14px',
                  border: '1px solid var(--border)',
                  borderRadius: 4,
                  background: 'var(--surface-2)',
                  fontSize: 13,
                  color: 'var(--text-dim)',
                  fontFamily: 'var(--font-sans)',
                  cursor: 'pointer',
                  lineHeight: 1.45,
                }}>
                <span style={{ color: 'var(--muted)', fontFamily: 'var(--font-mono)', fontSize: 10, marginRight: 8 }}>
                  {String(i + 1).padStart(2, '0')}
                </span>
                {s}
              </button>
            ))}
          </div>
        )}

        {mode === 'paste' && (
          <div className="samples">
            <span className="label-mono" style={{ marginRight: 4 }}>Try:</span>
            {SAMPLES.slice(0, 4).map((s, i) => (
              <button key={i} className="sample" onClick={() => handleSample(s)}>
                {s.slice(0, 36)}{s.length > 36 ? '…' : ''}
              </button>
            ))}
          </div>
        )}

        <div className="composer-footer">
          <div className="charcount">
            <span className={overLimit ? 'over' : ''}>{charCount.toLocaleString()}</span> / 5,000
            {text.trim() && <span style={{ marginLeft: 14, color: 'var(--muted)' }}>· {text.trim().split(/\s+/).length} words</span>}
          </div>
          <div style={{ display: 'flex', gap: 10, alignItems: 'center' }}>
            <div className="mode-switch">
              <button className={!live ? 'active' : ''} onClick={() => setLive(false)}>Manual</button>
              <button className={live ? 'active' : ''} onClick={() => setLive(true)}>Live</button>
            </div>
            {live && (
              <span className="live-badge"><span className="ld-dot"></span>LIVE</span>
            )}
            {!live && <span className="kbd-hint"><span className="kbd">⌘</span><span className="kbd">↵</span></span>}
            <button className="btn btn-primary" disabled={!canAnalyze} onClick={handleAnalyze}>
              {loading ? 'Analyzing…' : 'Analyze'}
              {!loading && <span style={{ fontSize: 12, opacity: 0.7 }}>→</span>}
            </button>
          </div>
        </div>
      </div>

      {/* Result */}
      <div className="result-card card">
        {loading && (
          <div className="loading">
            <div className="loading-bars">
              <span></span><span></span><span></span><span></span><span></span>
            </div>
            <div className="loading-text">Running ensemble · VADER · SVM · Transformer</div>
          </div>
        )}
        {!lastResult && !loading && (
          <div className="result-empty">
            <div className="empty-mark">∅</div>
            <div className="empty-hint">No analysis yet</div>
            <div style={{ maxWidth: 320, color: 'var(--text-dim)', fontSize: 13, lineHeight: 1.5 }}>
              Paste text and press <span className="kbd">⌘↵</span> to score sentiment across three backends, surface dominant emotions, flag sarcasm, and extract aspect-level signals.
            </div>
            <div className="empty-features">
              <span className="ef-row">3-class sentiment</span>
              <span className="ef-row">Emotion radar</span>
              <span className="ef-row">Sarcasm detection</span>
              <span className="ef-row">Aspect-based scores</span>
              <span className="ef-row">Ensemble fusion</span>
              <span className="ef-row">Batch + history</span>
            </div>
          </div>
        )}
        {lastResult && !loading && (
          <Result data={lastResult} history={history} tweaks={tweaks} />
        )}
      </div>
    </div>
  );
}

Object.assign(window, { Analyzer });
