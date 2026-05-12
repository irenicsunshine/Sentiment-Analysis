/* global React */
const { useMemo: useChartMemo } = React;

// ====== Emotion radar chart ======
function EmotionRadar({ emotions = {}, size = 220 }) {
  const names = ['joy','surprise','anger','disgust','fear','sadness'];
  const cx = size / 2, cy = size / 2;
  const r = size / 2 - 28;
  const angles = names.map((_, i) => (Math.PI * 2 * i) / names.length - Math.PI / 2);
  const points = names.map((n, i) => {
    const v = clamp(emotions[n] || 0, 0, 1);
    return [cx + Math.cos(angles[i]) * r * v, cy + Math.sin(angles[i]) * r * v];
  });
  const ringR = [0.25, 0.5, 0.75, 1];
  return (
    <svg width={size} height={size} viewBox={`0 0 ${size} ${size}`}>
      {ringR.map((f, i) => (
        <polygon key={i}
          points={names.map((_, j) => `${cx + Math.cos(angles[j]) * r * f},${cy + Math.sin(angles[j]) * r * f}`).join(' ')}
          fill="none"
          stroke="var(--border)"
          strokeWidth={i === ringR.length - 1 ? 1 : 0.6}
          strokeDasharray={i === ringR.length - 1 ? 'none' : '2 3'}
        />
      ))}
      {names.map((n, i) => (
        <line key={n}
          x1={cx} y1={cy}
          x2={cx + Math.cos(angles[i]) * r}
          y2={cy + Math.sin(angles[i]) * r}
          stroke="var(--border)" strokeWidth="0.6" />
      ))}
      <polygon
        points={points.map(p => p.join(',')).join(' ')}
        fill="var(--accent)" fillOpacity="0.18"
        stroke="var(--accent)" strokeWidth="1.5"
      />
      {points.map((p, i) => (
        <circle key={i} cx={p[0]} cy={p[1]} r="3" fill="var(--accent)" />
      ))}
      {names.map((n, i) => {
        const lx = cx + Math.cos(angles[i]) * (r + 14);
        const ly = cy + Math.sin(angles[i]) * (r + 14);
        return (
          <text key={n} x={lx} y={ly}
            textAnchor="middle" dominantBaseline="middle"
            fontFamily="JetBrains Mono, monospace"
            fontSize="9" letterSpacing="0.08em"
            fill="var(--muted)"
            style={{ textTransform: 'uppercase' }}>{n}</text>
        );
      })}
    </svg>
  );
}

// ====== Donut chart for sentiment distribution ======
function SentimentDonut({ counts, size = 200 }) {
  const total = (counts.positive || 0) + (counts.negative || 0) + (counts.neutral || 0);
  const r = size / 2 - 18, cx = size / 2, cy = size / 2;
  const sw = 22;
  if (total === 0) return <svg width={size} height={size}><circle cx={cx} cy={cy} r={r} fill="none" stroke="var(--border)" strokeWidth={sw} /></svg>;
  const segs = [
    { v: counts.positive / total, color: 'var(--pos)' },
    { v: counts.neutral  / total, color: 'var(--neu)' },
    { v: counts.negative / total, color: 'var(--neg)' },
  ];
  let cum = 0;
  const C = 2 * Math.PI * r;
  return (
    <svg width={size} height={size} viewBox={`0 0 ${size} ${size}`}>
      <circle cx={cx} cy={cy} r={r} fill="none" stroke="var(--bg-soft)" strokeWidth={sw} />
      {segs.map((s, i) => {
        const dash = s.v * C;
        const gap = C - dash;
        const off = -cum * C;
        cum += s.v;
        return (
          <circle key={i}
            cx={cx} cy={cy} r={r}
            fill="none" stroke={s.color}
            strokeWidth={sw}
            strokeDasharray={`${dash} ${gap}`}
            strokeDashoffset={off}
            transform={`rotate(-90 ${cx} ${cy})`}
            style={{ transition: 'all 0.5s' }}
          />
        );
      })}
      <text x={cx} y={cy - 4} textAnchor="middle" fontFamily="Instrument Serif" fontStyle="italic" fontSize="34" fill="var(--text)">{total}</text>
      <text x={cx} y={cy + 14} textAnchor="middle" fontFamily="JetBrains Mono" fontSize="9" fill="var(--muted)" letterSpacing="0.12em" style={{ textTransform: 'uppercase' }}>ANALYSES</text>
    </svg>
  );
}

// ====== Trend sparkline / area ======
function TrendChart({ data, width = 600, height = 180 }) {
  if (!data || data.length < 2) {
    return (
      <div style={{ height, display: 'flex', alignItems: 'center', justifyContent: 'center', color: 'var(--muted)', fontFamily: 'var(--font-mono)', fontSize: 10, letterSpacing: '0.1em', textTransform: 'uppercase' }}>
        Need at least 2 analyses to plot trend
      </div>
    );
  }
  const pad = { l: 30, r: 12, t: 12, b: 24 };
  const W = width, H = height;
  const innerW = W - pad.l - pad.r, innerH = H - pad.t - pad.b;
  const xs = data.map((_, i) => pad.l + (i / (data.length - 1)) * innerW);
  const ys = data.map(d => pad.t + (1 - (d.score + 1) / 2) * innerH);
  const path = xs.map((x, i) => `${i === 0 ? 'M' : 'L'} ${x} ${ys[i]}`).join(' ');
  const area = `${path} L ${xs[xs.length - 1]} ${pad.t + innerH / 2} L ${xs[0]} ${pad.t + innerH / 2} Z`;
  const midY = pad.t + innerH / 2;
  return (
    <svg width="100%" height={H} viewBox={`0 0 ${W} ${H}`} preserveAspectRatio="none">
      {/* zero axis */}
      <line x1={pad.l} y1={midY} x2={W - pad.r} y2={midY} stroke="var(--border-strong)" strokeWidth="0.6" strokeDasharray="3 3" />
      <text x={pad.l - 6} y={pad.t + 4} textAnchor="end" fontFamily="JetBrains Mono" fontSize="9" fill="var(--muted)">+1</text>
      <text x={pad.l - 6} y={midY + 3} textAnchor="end" fontFamily="JetBrains Mono" fontSize="9" fill="var(--muted)">0</text>
      <text x={pad.l - 6} y={H - pad.b + 2} textAnchor="end" fontFamily="JetBrains Mono" fontSize="9" fill="var(--muted)">-1</text>
      <path d={area} fill="var(--accent)" fillOpacity="0.08" />
      <path d={path} fill="none" stroke="var(--accent)" strokeWidth="1.5" />
      {xs.map((x, i) => (
        <circle key={i} cx={x} cy={ys[i]} r="3" fill={data[i].score > 0.15 ? 'var(--pos)' : data[i].score < -0.15 ? 'var(--neg)' : 'var(--neu)'} stroke="var(--bg)" strokeWidth="1.5" />
      ))}
    </svg>
  );
}

// ====== Result vis variants: bar / donut / radial / sparkline ======
function ScoreBar({ score, sentiment }) {
  const v = clamp(score, -1, 1);
  const leftPct = v >= 0 ? 50 : 50 + v * 50;
  const widthPct = Math.abs(v) * 50;
  return (
    <div className="result-track">
      <div className="axis"></div>
      <div className="ticks">
        <span>-1.0</span><span>NEG</span><span>0</span><span>POS</span><span>+1.0</span>
      </div>
      <div className={`fill ${sentiment}`} style={{ left: `${leftPct}%`, width: `${widthPct}%` }}></div>
    </div>
  );
}

function ScoreDonut({ score, sentiment }) {
  const v = clamp(score, -1, 1);
  const abs = Math.abs(v);
  const C = 2 * Math.PI * 38;
  const dash = abs * C;
  return (
    <svg width="120" height="120" viewBox="0 0 120 120">
      <circle cx="60" cy="60" r="38" fill="none" stroke="var(--bg-soft)" strokeWidth="10" />
      <circle cx="60" cy="60" r="38" fill="none"
        stroke={sentiment === 'positive' ? 'var(--pos)' : sentiment === 'negative' ? 'var(--neg)' : 'var(--neu)'}
        strokeWidth="10"
        strokeDasharray={`${dash} ${C}`}
        strokeDashoffset="0"
        strokeLinecap="round"
        transform="rotate(-90 60 60)" />
      <text x="60" y="60" textAnchor="middle" fontFamily="Instrument Serif" fontStyle="italic" fontSize="26" fill="var(--text)">{fmt(v, 2)}</text>
      <text x="60" y="78" textAnchor="middle" fontFamily="JetBrains Mono" fontSize="8" letterSpacing="0.15em" fill="var(--muted)" style={{ textTransform: 'uppercase' }}>SCORE</text>
    </svg>
  );
}

function ScoreRadial({ score, sentiment }) {
  const v = clamp(score, -1, 1);
  const angle = ((v + 1) / 2) * 180 - 180; // -180 to 0
  const color = sentiment === 'positive' ? 'var(--pos)' : sentiment === 'negative' ? 'var(--neg)' : 'var(--neu)';
  const arcs = [];
  const segs = 24;
  for (let i = 0; i < segs; i++) {
    const a1 = (-180 + (i / segs) * 180) * Math.PI / 180;
    const a2 = (-180 + ((i + 0.7) / segs) * 180) * Math.PI / 180;
    const r1 = 50, r2 = 60;
    const cx = 70, cy = 70;
    const x1 = cx + Math.cos(a1) * r1, y1 = cy + Math.sin(a1) * r1;
    const x2 = cx + Math.cos(a1) * r2, y2 = cy + Math.sin(a1) * r2;
    const x3 = cx + Math.cos(a2) * r2, y3 = cy + Math.sin(a2) * r2;
    const x4 = cx + Math.cos(a2) * r1, y4 = cy + Math.sin(a2) * r1;
    const targetA = ((v + 1) / 2) * 180;
    const segA = (i / segs) * 180;
    const active = segA <= targetA;
    arcs.push(
      <polygon key={i} points={`${x1},${y1} ${x2},${y2} ${x3},${y3} ${x4},${y4}`}
        fill={active ? color : 'var(--border)'} opacity={active ? 1 : 0.5} />
    );
  }
  return (
    <svg width="140" height="100" viewBox="0 0 140 100">
      {arcs}
      <text x="70" y="78" textAnchor="middle" fontFamily="Instrument Serif" fontStyle="italic" fontSize="26" fill="var(--text)">{fmt(v, 2)}</text>
      <text x="70" y="94" textAnchor="middle" fontFamily="JetBrains Mono" fontSize="8" letterSpacing="0.15em" fill="var(--muted)" style={{ textTransform: 'uppercase' }}>POLARITY</text>
    </svg>
  );
}

function ScoreSparkline({ history = [], score, sentiment }) {
  const data = [...history.slice(-9).map(h => h.score), score];
  const W = 200, H = 60;
  if (data.length < 2) return <ScoreBar score={score} sentiment={sentiment} />;
  const xs = data.map((_, i) => 4 + (i / (data.length - 1)) * (W - 8));
  const ys = data.map(d => 4 + (1 - (d + 1) / 2) * (H - 8));
  const path = xs.map((x, i) => `${i === 0 ? 'M' : 'L'} ${x} ${ys[i]}`).join(' ');
  const color = sentiment === 'positive' ? 'var(--pos)' : sentiment === 'negative' ? 'var(--neg)' : 'var(--neu)';
  return (
    <svg width={W} height={H} viewBox={`0 0 ${W} ${H}`}>
      <line x1="0" y1={H/2} x2={W} y2={H/2} stroke="var(--border)" strokeDasharray="2 3" />
      <path d={path} fill="none" stroke={color} strokeWidth="2" />
      {xs.map((x, i) => i === xs.length - 1 ? <circle key={i} cx={x} cy={ys[i]} r="4" fill={color} stroke="var(--bg)" strokeWidth="2" /> : null)}
    </svg>
  );
}

Object.assign(window, {
  EmotionRadar, SentimentDonut, TrendChart,
  ScoreBar, ScoreDonut, ScoreRadial, ScoreSparkline,
});
