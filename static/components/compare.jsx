/* global React */

function Compare({ onAnalyze }) {
  const [textA, setTextA] = useState('');
  const [textB, setTextB] = useState('');
  const [resA, setResA] = useState(null);
  const [resB, setResB] = useState(null);
  const [busy, setBusy] = useState(false);

  const canRun = textA.trim() && textB.trim() && !busy;

  const handleRun = async () => {
    setBusy(true);
    setResA(null); setResB(null);
    try {
      const [a, b] = await Promise.all([onAnalyze(textA), onAnalyze(textB)]);
      setResA({ ...a, text: textA });
      setResB({ ...b, text: textB });
    } finally {
      setBusy(false);
    }
  };

  const loadSample = () => {
    setTextA('The new espresso machine is genuinely transformative. Pulls beautiful shots every time. Best purchase of the year.');
    setTextB('The espresso machine is fine. Decent shots, nothing special. Cleaning is a pain.');
  };

  const delta = (resA && resB) ? round(resA.sentiment_score - resB.sentiment_score, 3) : null;
  const winner = delta === null ? null : delta > 0 ? 'A' : delta < 0 ? 'B' : 'tie';

  const Side = ({ label, text, setText, res }) => (
    <div className="card compare-side">
      <div className="composer-meta">
        <div className="left">
          <span className="seq">{label}</span>
          <span>{text.length} chars</span>
        </div>
        {res && <span className={`label-mono ${res.sentiment}`} style={{ color: `var(--${res.sentiment === 'positive' ? 'pos' : res.sentiment === 'negative' ? 'neg' : 'neu'})` }}>{res.sentiment}</span>}
      </div>
      <textarea
        placeholder={`Text ${label}…`}
        value={text}
        onChange={(e) => setText(e.target.value)}
      />
      {res && (
        <div className="compare-result">
          <div className={`sentiment-word ${res.sentiment}`} style={{ fontSize: 56 }}>
            <em>{res.sentiment}</em>.
          </div>
          <div className="result-track" style={{ marginTop: 4 }}>
            <div className="axis"></div>
            <div className="ticks">
              <span>-1</span><span>NEG</span><span>0</span><span>POS</span><span>+1</span>
            </div>
            <div className={`fill ${res.sentiment}`} style={{
              left: `${res.sentiment_score >= 0 ? 50 : 50 + res.sentiment_score * 50}%`,
              width: `${Math.abs(res.sentiment_score) * 50}%`,
            }}></div>
          </div>
          <div className="mini-stats">
            <div className="mini-stat">
              <div className="ms-label">Score</div>
              <div className={`ms-val ${res.sentiment}`}>{fmt(res.sentiment_score, 3)}</div>
            </div>
            <div className="mini-stat">
              <div className="ms-label">Conf</div>
              <div className="ms-val">{pct(res.confidence)}</div>
            </div>
            <div className="mini-stat">
              <div className="ms-label">Sarcasm</div>
              <div className="ms-val" style={{ color: res.sarcasm_score > 0.4 ? 'var(--sarc)' : 'var(--muted)' }}>{Math.round(res.sarcasm_score * 100)}%</div>
            </div>
          </div>
        </div>
      )}
    </div>
  );

  const absDelta = delta !== null ? Math.abs(delta) : 0;
  const magnitude = absDelta < 0.1 ? 'roughly equivalent' : absDelta < 0.4 ? 'meaningfully different' : absDelta < 0.8 ? 'sharply different' : 'polar opposites';

  return (
    <div>
      <div className="screen-head">
        <div>
          <h1 className="screen-title">Compare <em>two texts</em></h1>
          <div className="screen-sub">A/B copy testing · score delta · ensemble agreement</div>
        </div>
        <div className="screen-meta">
          <button className="btn btn-ghost" onClick={loadSample}>Load sample</button>
          <button className="btn btn-primary" disabled={!canRun} onClick={handleRun}>
            {busy ? 'Analyzing both…' : 'Compare'}
          </button>
        </div>
      </div>

      <div className="compare-grid">
        <Side label="A" text={textA} setText={setTextA} res={resA} />
        <div className="compare-vs">
          <span className="vs-mark">vs</span>
          {delta !== null && (
            <span className="vs-tag">Δ {fmt(delta, 2)}</span>
          )}
        </div>
        <Side label="B" text={textB} setText={setTextB} res={resB} />
      </div>

      {resA && resB && (
        <div className="card compare-delta" style={{ marginTop: 'var(--gap)' }}>
          <div>
            <div className="label-mono">Delta · score</div>
            <div className={`delta-num ${delta > 0 ? 'positive' : delta < 0 ? 'negative' : ''}`}>
              {fmt(delta, 2)}
            </div>
          </div>
          <div style={{ width: 1, alignSelf: 'stretch', background: 'var(--border)' }}></div>
          <div>
            <div className="label-mono">Verdict</div>
            <div className="summary">
              {winner === 'tie' ? <>The texts are <em>roughly equivalent</em> in sentiment.</> :
                <>Text <em>{winner}</em> is {magnitude}{winner === 'A' ? ' — and reads more positive' : ' — and reads more positive'} than the other.</>}
            </div>
            <div className="label-mono" style={{ marginTop: 12 }}>
              A · conf {pct(resA.confidence)} · sarcasm {Math.round(resA.sarcasm_score * 100)}%{'   '}
              B · conf {pct(resB.confidence)} · sarcasm {Math.round(resB.sarcasm_score * 100)}%
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

Object.assign(window, { Compare });
