/* global React, ReactDOM */

const TWEAK_DEFAULTS = /*EDITMODE-BEGIN*/{
  "theme": "dark",
  "accent": "green",
  "typeface": "sans",
  "density": "comfortable",
  "scoreVis": "bar",
  "showBackends": true
}/*EDITMODE-END*/;

const ACCENTS = {
  green:  'oklch(0.78 0.17 148)',
  amber:  'oklch(0.80 0.16 75)',
  azure:  'oklch(0.74 0.16 235)',
  coral:  'oklch(0.74 0.18 25)',
  violet: 'oklch(0.74 0.18 295)',
};

const TABS = [
  { id: 'analyze',  label: 'Analyze',  num: '01' },
  { id: 'compare',  label: 'Compare',  num: '02' },
  { id: 'batch',    label: 'Batch',    num: '03' },
  { id: 'history',  label: 'History',  num: '04' },
  { id: 'stats',    label: 'Stats',    num: '05' },
  { id: 'settings', label: 'Settings', num: '06' },
];

function App() {
  // Tweaks
  const [tweaks, setTweaks] = useTweaks(TWEAK_DEFAULTS);

  // App state
  const [tab, setTab] = useState('analyze');
  const [loading, setLoading] = useState(false);
  const [lastResult, setLastResult] = useState(null);
  const [history, setHistory] = useState(() => {
    try { return JSON.parse(localStorage.getItem('sl_history') || '[]'); }
    catch (e) { return []; }
  });
  const [batchLoading, setBatchLoading] = useState(false);
  const [batchResults, setBatchResults] = useState(null);

  const [backends, setBackends] = useState({ vader: true, svm: true, transformer: false });
  const [useTransformer, setUseTransformer] = useState(false);
  const [claudeEnabled, setClaudeEnabled] = useState(true);

  // Persist history
  useEffect(() => {
    try { localStorage.setItem('sl_history', JSON.stringify(history.slice(-100))); }
    catch (e) {}
  }, [history]);

  // Persist tweaks for end-users (localStorage). Reads on mount, writes on change.
  useEffect(() => {
    try {
      const stored = JSON.parse(localStorage.getItem('sl_tweaks') || 'null');
      if (stored && typeof stored === 'object') setTweaks(stored);
    } catch (e) {}
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);
  useEffect(() => {
    try { localStorage.setItem('sl_tweaks', JSON.stringify(tweaks)); } catch (e) {}
  }, [tweaks]);

  // Apply theme tokens
  useEffect(() => {
    const root = document.documentElement;
    root.setAttribute('data-theme', tweaks.theme);
    root.setAttribute('data-density', tweaks.density);
    root.setAttribute('data-typeface', tweaks.typeface);
    root.style.setProperty('--accent', ACCENTS[tweaks.accent] || ACCENTS.green);
  }, [tweaks]);

  // Analyze handler
  const handleAnalyze = useCallback(async (text, opts = {}) => {
    if (!opts.silent) setLoading(true);
    try {
      const result = claudeEnabled ? await analyzeWithClaude(text) : mockAnalyze(text);
      const record = { ...result, text, timestamp: new Date().toISOString() };
      setLastResult(record);
      if (!opts.silent) setHistory(h => [...h, record]);
      return record;
    } catch (e) {
      console.error(e);
      const fallback = mockAnalyze(text);
      const record = { ...fallback, text, timestamp: new Date().toISOString() };
      setLastResult(record);
      if (!opts.silent) setHistory(h => [...h, record]);
      return record;
    } finally {
      if (!opts.silent) setLoading(false);
    }
  }, [claudeEnabled]);

  // Pure analyze (no state side-effects) for Compare
  const pureAnalyze = useCallback(async (text) => {
    try {
      return claudeEnabled ? await analyzeWithClaude(text) : mockAnalyze(text);
    } catch (e) {
      return mockAnalyze(text);
    }
  }, [claudeEnabled]);

  // Batch handler — analyzes sequentially
  const handleBatchAnalyze = useCallback(async (texts) => {
    setBatchLoading(true);
    setBatchResults([]);
    const acc = [];
    for (let i = 0; i < texts.length; i++) {
      const text = texts[i];
      try {
        const result = claudeEnabled ? await analyzeWithClaude(text) : mockAnalyze(text);
        acc.push({ ...result, text, timestamp: new Date().toISOString() });
      } catch (e) {
        const fb = mockAnalyze(text);
        acc.push({ ...fb, text, timestamp: new Date().toISOString() });
      }
      setBatchResults([...acc]);
    }
    setBatchLoading(false);
  }, [claudeEnabled]);

  const handleClearHistory = () => {
    if (!confirm('Clear all analysis history?')) return;
    setHistory([]);
    setLastResult(null);
  };

  const handleOpenHistory = (h) => {
    setLastResult(h);
    setTab('analyze');
  };

  // Keyboard nav between tabs
  useEffect(() => {
    const onKey = (e) => {
      if (e.target.tagName === 'TEXTAREA' || e.target.tagName === 'INPUT') return;
      const i = TABS.findIndex(t => t.id === tab);
      if (e.key === '[' && i > 0) setTab(TABS[i - 1].id);
      if (e.key === ']' && i < TABS.length - 1) setTab(TABS[i + 1].id);
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [tab]);

  return (
    <div className="app" data-screen-label="SentimentLab">
      <header className="topbar">
        <div className="brand">
          <span className="brand-dot"></span>
          <span>SENTIMENT.LAB</span>
          <span className="brand-tag">v0.4</span>
        </div>

        <nav className="nav">
          {TABS.map(t => (
            <button key={t.id}
              className={tab === t.id ? 'active' : ''}
              onClick={() => setTab(t.id)}>
              <span className="nav-num">{t.num}</span>
              <span>{t.label}</span>
            </button>
          ))}
        </nav>

        <div className="status-row">
          <button className="theme-toggle"
                  onClick={() => setTweaks({ theme: tweaks.theme === 'dark' ? 'light' : 'dark' })}
                  title={`Switch to ${tweaks.theme === 'dark' ? 'light' : 'dark'} mode`}
                  aria-label="Toggle theme">
            <span className={`tt-track ${tweaks.theme === 'light' ? 'on' : ''}`}>
              <span className="tt-thumb">
                {tweaks.theme === 'dark'
                  ? <svg width="11" height="11" viewBox="0 0 24 24" fill="none"><path d="M21 12.79A9 9 0 1111.21 3 7 7 0 0021 12.79z" fill="currentColor"/></svg>
                  : <svg width="11" height="11" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.4" strokeLinecap="round"><circle cx="12" cy="12" r="4" fill="currentColor"/><path d="M12 2v2M12 20v2M4.93 4.93l1.41 1.41M17.66 17.66l1.41 1.41M2 12h2M20 12h2M4.93 19.07l1.41-1.41M17.66 6.34l1.41-1.41"/></svg>}
              </span>
            </span>
            <span className="tt-label">{tweaks.theme === 'dark' ? 'Dark' : 'Light'}</span>
          </button>
          <span className="status-pill">
            <span className={`dot ${claudeEnabled ? '' : 'off'}`}></span>
            <span>{claudeEnabled ? 'Claude · live' : 'Offline'}</span>
          </span>
          <span className="status-pill">
            <span>{Object.values(backends).filter(Boolean).length}/3 backends</span>
          </span>
        </div>
      </header>

      <main className="main">
        <div className="screen" data-screen-label={`${TABS.find(t => t.id === tab).num} ${TABS.find(t => t.id === tab).label}`}>
          {tab === 'analyze' && (
            <Analyzer
              onAnalyze={handleAnalyze}
              loading={loading}
              lastResult={lastResult}
              history={history}
              tweaks={tweaks}
            />
          )}
          {tab === 'batch' && (
            <Batch
              onBatchAnalyze={handleBatchAnalyze}
              results={batchResults}
              loading={batchLoading}
              tweaks={tweaks}
            />
          )}
          {tab === 'compare' && (
            <Compare onAnalyze={pureAnalyze} />
          )}
          {tab === 'history' && (
            <History
              history={history}
              onClear={handleClearHistory}
              onOpen={handleOpenHistory}
            />
          )}
          {tab === 'stats' && <Stats history={history} />}
          {tab === 'settings' && (
            <Settings
              backends={backends}
              setBackends={setBackends}
              useTransformer={useTransformer}
              setUseTransformer={setUseTransformer}
              claudeEnabled={claudeEnabled}
              setClaudeEnabled={setClaudeEnabled}
              tweaks={tweaks}
              setTweaks={setTweaks}
              accents={ACCENTS}
            />
          )}
        </div>
      </main>

      {/* Tweaks panel */}
      <TweaksPanel title="Tweaks">
        <TweakSection label="Theme">
          <TweakRadio label="Mode"
            options={['dark', 'light']}
            value={tweaks.theme}
            onChange={(v) => setTweaks({ theme: v })}
          />
          <TweakColor label="Accent"
            options={Object.values(ACCENTS)}
            value={ACCENTS[tweaks.accent]}
            onChange={(v) => {
              const key = Object.entries(ACCENTS).find(([k, c]) => c === v)?.[0] || 'green';
              setTweaks({ accent: key });
            }}
          />
        </TweakSection>

        <TweakSection label="Type & space">
          <TweakRadio label="Typeface"
            options={['sans', 'serif', 'mono']}
            value={tweaks.typeface}
            onChange={(v) => setTweaks({ typeface: v })}
          />
          <TweakRadio label="Density"
            options={['compact', 'comfortable', 'spacious']}
            value={tweaks.density}
            onChange={(v) => setTweaks({ density: v })}
          />
        </TweakSection>

        <TweakSection label="Result viz">
          <TweakSelect label="Score chart"
            options={['bar', 'donut', 'radial', 'sparkline']}
            value={tweaks.scoreVis}
            onChange={(v) => setTweaks({ scoreVis: v })}
          />
          <TweakToggle label="Show backend votes"
            value={tweaks.showBackends}
            onChange={(v) => setTweaks({ showBackends: v })}
          />
        </TweakSection>
      </TweaksPanel>
    </div>
  );
}

ReactDOM.createRoot(document.getElementById('root')).render(<App />);
