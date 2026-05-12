/* global React */

function Settings({
  backends, setBackends, useTransformer, setUseTransformer,
  claudeEnabled, setClaudeEnabled,
  tweaks, setTweaks, accents,
}) {
  const Row = ({ k, desc, on, onChange, disabled }) => (
    <div className="settings-row">
      <div className="key">{k}</div>
      <div className="desc">{desc}</div>
      <div className={`toggle ${on ? 'on' : ''} ${disabled ? 'disabled' : ''}`}
           style={{ opacity: disabled ? 0.4 : 1, pointerEvents: disabled ? 'none' : 'auto' }}
           onClick={() => !disabled && onChange(!on)}></div>
    </div>
  );

  // Segmented control (mirrors Tweaks panel)
  const Seg = ({ label, options, value, onChange }) => (
    <div className="ap-field">
      <div className="ap-label">{label}</div>
      <div className="ap-seg">
        {options.map(opt => (
          <button key={opt}
                  className={`ap-seg-btn ${value === opt ? 'on' : ''}`}
                  onClick={() => onChange(opt)}>{opt}</button>
        ))}
      </div>
    </div>
  );

  const Select = ({ label, options, value, onChange }) => (
    <div className="ap-field">
      <div className="ap-label">{label}</div>
      <div className="ap-select-wrap">
        <select className="ap-select" value={value} onChange={(e) => onChange(e.target.value)}>
          {options.map(o => <option key={o} value={o}>{o}</option>)}
        </select>
      </div>
    </div>
  );

  return (
    <div>
      <div className="screen-head">
        <div>
          <h1 className="screen-title">Settings <em>&amp; engine</em></h1>
          <div className="screen-sub">Appearance · backends · models · runtime config</div>
        </div>
      </div>

      {/* Appearance — same controls as Tweaks panel, exposed to end users */}
      <div className="card settings-section appearance">
        <div className="card-header">Appearance</div>

        <div className="ap-grid">
          <Seg label="Mode"
               options={['dark', 'light']}
               value={tweaks.theme}
               onChange={(v) => setTweaks({ theme: v })} />

          <div className="ap-field">
            <div className="ap-label">Accent</div>
            <div className="ap-swatches">
              {Object.entries(accents).map(([key, color]) => (
                <button key={key}
                        className={`ap-swatch ${tweaks.accent === key ? 'on' : ''}`}
                        style={{ '--sw': color }}
                        onClick={() => setTweaks({ accent: key })}
                        aria-label={key} title={key}>
                  {tweaks.accent === key && (
                    <svg width="14" height="14" viewBox="0 0 24 24" fill="none"
                         stroke="currentColor" strokeWidth="3" strokeLinecap="round" strokeLinejoin="round">
                      <polyline points="20 6 9 17 4 12"></polyline>
                    </svg>
                  )}
                </button>
              ))}
            </div>
          </div>

          <Seg label="Typeface"
               options={['sans', 'serif', 'mono']}
               value={tweaks.typeface}
               onChange={(v) => setTweaks({ typeface: v })} />

          <Select label="Density"
                  options={['compact', 'comfortable', 'spacious']}
                  value={tweaks.density}
                  onChange={(v) => setTweaks({ density: v })} />

          <Select label="Score chart"
                  options={['bar', 'donut', 'radial', 'sparkline']}
                  value={tweaks.scoreVis}
                  onChange={(v) => setTweaks({ scoreVis: v })} />

          <div className="ap-field">
            <div className="ap-label">Show backend votes</div>
            <div className={`toggle ${tweaks.showBackends ? 'on' : ''}`}
                 onClick={() => setTweaks({ showBackends: !tweaks.showBackends })}></div>
          </div>
        </div>

        <div className="ap-note">
          Preferences are saved to this browser. Clear site data to reset.
        </div>
      </div>

      <div className="card settings-section">
        <div className="card-header">Engine</div>
        <Row
          k="claude.fusion"
          desc="Use live Claude API as the analysis backbone. Falls back to local heuristic if unavailable."
          on={claudeEnabled}
          onChange={setClaudeEnabled}
        />
        <Row
          k="USE_TRANSFORMER"
          desc="Enable HuggingFace Transformer backend (~500MB download, higher accuracy)."
          on={useTransformer}
          onChange={setUseTransformer}
        />
      </div>

      <div className="card settings-section">
        <div className="card-header">Backends · ensemble</div>
        <Row k="VADER"       desc="Lexicon + rule-based scorer. Fast. Great for short, informal text." on={backends.vader}       onChange={(v) => setBackends({ ...backends, vader: v })} />
        <Row k="SVM"         desc="Trained on tweet_eval (~47k samples) with optional slang-aware retraining." on={backends.svm}         onChange={(v) => setBackends({ ...backends, svm: v })} />
        <Row k="Transformer" desc="HuggingFace transformer. Highest accuracy, slowest." on={backends.transformer} onChange={(v) => setBackends({ ...backends, transformer: v })} disabled={!useTransformer} />
      </div>

      <div className="card settings-section">
        <div className="card-header">Limits</div>
        <div className="settings-row">
          <div className="key">MAX_TEXT_LENGTH</div>
          <div className="desc">Maximum characters per single analysis.</div>
          <div className="seq" style={{ fontFamily: 'var(--font-mono)', fontSize: 12, color: 'var(--text)' }}>5,000</div>
        </div>
        <div className="settings-row">
          <div className="key">MAX_BATCH_SIZE</div>
          <div className="desc">Maximum lines per batch run.</div>
          <div className="seq" style={{ fontFamily: 'var(--font-mono)', fontSize: 12, color: 'var(--text)' }}>50</div>
        </div>
      </div>

      <div className="card settings-section">
        <div className="card-header">Storage</div>
        <div className="settings-row">
          <div className="key">history.store</div>
          <div className="desc">Local browser storage · 100 analyses retained. Postgres mirror in deployed build.</div>
          <div className="seq" style={{ fontFamily: 'var(--font-mono)', fontSize: 12, color: 'var(--pos)' }}>localStorage</div>
        </div>
      </div>

      <div style={{
        fontFamily: 'var(--font-mono)', fontSize: 10, color: 'var(--muted)',
        letterSpacing: '0.08em', padding: '20px 4px', textAlign: 'center',
      }}>
        SENTIMENT.LAB · v0.4.0
      </div>
    </div>
  );
}

Object.assign(window, { Settings });
