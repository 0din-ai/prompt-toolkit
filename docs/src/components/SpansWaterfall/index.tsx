import React, {useMemo, useState} from 'react';
import styles from './styles.module.css';

/** A single timed phase of a classify() call. */
export interface PhaseSpan {
  name: string;
  start_ms: number;
  duration_ms: number;
  chunk_index?: number;
  token_count?: number;
}

export interface SpansResult {
  total_timing_ms?: number;
  total_tokens?: number;
  spans: PhaseSpan[];
}

interface Preset {
  label: string;
  result: SpansResult;
}

/**
 * Real captures: OnnxSusFactor.classify() on an Apple M4 Pro CPU (release
 * build), one per prompt length. The longest prompt (1,348 tokens) is split
 * into three overlapping ~510-token chunks scored on a shared ONNX session.
 */
const PRESETS: Preset[] = [
  {
    label: 'Short · 15 tok',
    result: {
      total_timing_ms: 33.888,
      total_tokens: 15,
      spans: [
        {name: 'tokenize', start_ms: 0.0, duration_ms: 0.33},
        {name: 'chunk', start_ms: 0.33, duration_ms: 0.02},
        {
          name: 'inference',
          start_ms: 0.36,
          duration_ms: 33.46,
          chunk_index: 0,
          token_count: 15,
        },
        {name: 'reduce', start_ms: 33.89, duration_ms: 0.0},
      ],
    },
  },
  {
    label: 'Medium · 191 tok',
    result: {
      total_timing_ms: 144.354,
      total_tokens: 191,
      spans: [
        {name: 'tokenize', start_ms: 0.0, duration_ms: 0.53},
        {name: 'chunk', start_ms: 0.53, duration_ms: 0.0},
        {
          name: 'inference',
          start_ms: 0.54,
          duration_ms: 143.79,
          chunk_index: 0,
          token_count: 191,
        },
        {name: 'reduce', start_ms: 144.35, duration_ms: 0.0},
      ],
    },
  },
  {
    label: 'Long · 1,348 tok (3 chunks)',
    result: {
      total_timing_ms: 723.359,
      total_tokens: 1348,
      spans: [
        {name: 'tokenize', start_ms: 0.0, duration_ms: 1.56},
        {name: 'chunk', start_ms: 1.56, duration_ms: 0.02},
        {
          name: 'inference',
          start_ms: 1.6,
          duration_ms: 290.63,
          chunk_index: 0,
          token_count: 510,
        },
        {
          name: 'inference',
          start_ms: 1.65,
          duration_ms: 528.67,
          chunk_index: 1,
          token_count: 510,
        },
        {
          name: 'inference',
          start_ms: 1.67,
          duration_ms: 721.67,
          chunk_index: 2,
          token_count: 428,
        },
        {name: 'reduce', start_ms: 723.36, duration_ms: 0.0},
      ],
    },
  },
];

const PHASE_CLASS: Record<string, string> = {
  tokenize: styles.tokenize,
  chunk: styles.chunk,
  inference: styles.inference,
  reduce: styles.reduce,
};

const fmt = (n: number): string =>
  (Math.round(n * 1000) / 1000).toLocaleString();

function labelFor(s: PhaseSpan): string {
  return s.name === 'inference' && s.chunk_index != null
    ? `inference[${s.chunk_index}]`
    : s.name;
}

/** Wall time covered by the union of span intervals (overlap-aware). */
function mergedCoverage(spans: PhaseSpan[]): number {
  const iv = spans
    .map((s) => [s.start_ms, s.start_ms + s.duration_ms] as [number, number])
    .sort((a, b) => a[0] - b[0]);
  let total = 0;
  let curStart: number | null = null;
  let curEnd: number | null = null;
  for (const [s, e] of iv) {
    if (curEnd === null || s > curEnd) {
      if (curEnd !== null) total += curEnd - (curStart as number);
      curStart = s;
      curEnd = e;
    } else if (e > curEnd) {
      curEnd = e;
    }
  }
  if (curEnd !== null) total += curEnd - (curStart as number);
  return total;
}

function Chart({result}: {result: SpansResult}): JSX.Element {
  const spans = result.spans ?? [];
  const total =
    result.total_timing_ms ??
    Math.max(0, ...spans.map((s) => s.start_ms + s.duration_ms));

  const rowH = 34;
  const padTop = 14;
  const padBot = 34;
  const labelW = 118;
  const durW = 92;
  const W = 900;
  const plotL = labelW;
  const plotR = W - durW - 12;
  const plotW = plotR - plotL;
  const H = padTop + spans.length * rowH + padBot;
  const x = (ms: number): number => plotL + (total > 0 ? (ms / total) * plotW : 0);

  const ticks = 5;
  const gridlines = Array.from({length: ticks + 1}, (_, i) => {
    const ms = (total / ticks) * i;
    return {ms, gx: x(ms), i};
  });

  return (
    <svg
      className={styles.chart}
      viewBox={`0 0 ${W} ${H}`}
      role="img"
      aria-label="SusFactor phase span waterfall">
      {gridlines.map(({ms, gx, i}) => (
        <g key={`grid-${i}`}>
          <line
            className={styles.gridline}
            x1={gx}
            x2={gx}
            y1={padTop}
            y2={H - padBot}
          />
          <text
            className={styles.axis}
            x={gx}
            y={H - padBot + 16}
            textAnchor={i === 0 ? 'start' : i === ticks ? 'end' : 'middle'}>
            {fmt(ms)} ms
          </text>
        </g>
      ))}
      {spans.map((s, i) => {
        const y = padTop + i * rowH;
        const barY = y + 6;
        const barH = rowH - 14;
        const bx = x(s.start_ms);
        const bw = Math.max(2, x(s.start_ms + s.duration_ms) - bx);
        return (
          <g key={`span-${i}`}>
            <text
              className={styles.rowLabel}
              x={plotL - 12}
              y={barY + barH / 2 + 4}
              textAnchor="end">
              {labelFor(s)}
            </text>
            <rect
              className={PHASE_CLASS[s.name] ?? styles.reduce}
              x={bx}
              y={barY}
              width={bw}
              height={barH}
              rx={4}
            />
            <text
              className={styles.rowDur}
              x={plotR + 10}
              y={barY + barH / 2 + 4}>
              {fmt(s.duration_ms)} ms
              {s.name === 'inference' && s.token_count != null
                ? ` · ${s.token_count} tok`
                : ''}
            </text>
          </g>
        );
      })}
    </svg>
  );
}

export default function SpansWaterfall(): JSX.Element {
  const [activeIdx, setActiveIdx] = useState(0);
  const [custom, setCustom] = useState<SpansResult | null>(null);
  const [input, setInput] = useState<string>(
    JSON.stringify(PRESETS[0].result, null, 2),
  );
  const [error, setError] = useState<string>('');

  const result = custom ?? PRESETS[activeIdx].result;

  const {total, totalTokens, chunks, maxInf, overhead} = useMemo(() => {
    const spans = result.spans ?? [];
    const t =
      result.total_timing_ms ??
      Math.max(0, ...spans.map((s) => s.start_ms + s.duration_ms));
    const inf = spans.filter((s) => s.name === 'inference');
    return {
      total: t,
      totalTokens: result.total_tokens,
      chunks: inf.length,
      maxInf: Math.max(0, ...inf.map((s) => s.duration_ms)),
      overhead: Math.max(0, t - mergedCoverage(spans)),
    };
  }, [result]);

  const selectPreset = (i: number): void => {
    setActiveIdx(i);
    setCustom(null);
    setInput(JSON.stringify(PRESETS[i].result, null, 2));
    setError('');
  };

  const onRender = (): void => {
    setError('');
    try {
      const parsed = JSON.parse(input);
      if (!Array.isArray(parsed.spans)) {
        throw new Error('JSON must have a `spans` array.');
      }
      setCustom(parsed as SpansResult);
    } catch (e) {
      setError(String((e as Error).message ?? e));
    }
  };

  return (
    <div className={styles.card}>
      <div className={styles.tabs} role="tablist">
        {PRESETS.map((p, i) => (
          <button
            key={p.label}
            type="button"
            role="tab"
            aria-selected={!custom && i === activeIdx}
            className={`${styles.tab} ${
              !custom && i === activeIdx ? styles.tabActive : ''
            }`}
            onClick={() => selectPreset(i)}>
            {p.label}
          </button>
        ))}
        {custom ? <span className={styles.tabCustom}>custom</span> : null}
      </div>

      <div className={styles.meta}>
        <span>
          total <b>{fmt(total)} ms</b>
        </span>
        {totalTokens != null ? (
          <span>
            tokens <b>{totalTokens.toLocaleString()}</b>
          </span>
        ) : null}
        <span>
          chunks <b>{chunks}</b>
        </span>
        <span>
          slowest chunk <b>{fmt(maxInf)} ms</b>
        </span>
        <span>
          scheduling overhead <b>{fmt(overhead)} ms</b>
        </span>
      </div>

      <Chart result={result} />

      <div className={styles.legend}>
        <span>
          <span className={`${styles.swatch} ${styles.tokenize}`} />
          tokenize (serialize request)
        </span>
        <span>
          <span className={`${styles.swatch} ${styles.chunk}`} />
          chunk (batch)
        </span>
        <span>
          <span className={`${styles.swatch} ${styles.inference}`} />
          inference (model)
        </span>
        <span>
          <span className={`${styles.swatch} ${styles.reduce}`} />
          reduce (assemble response)
        </span>
      </div>

      <p className={styles.note}>
        {chunks > 1 ? (
          <>
            <b>Inference dominates; everything else stays sub-millisecond.</b>{' '}
            This {totalTokens?.toLocaleString()}-token prompt exceeded the model's
            512-token window, so it was split into {chunks} chunks scored on a
            shared ONNX session — later chunks' spans include time spent waiting
            for that session, so a bar's length reflects scheduling, not its
            token count.
          </>
        ) : (
          <>
            <b>Inference is essentially the entire call.</b> A{' '}
            {totalTokens?.toLocaleString()}-token prompt classified in{' '}
            {fmt(total)} ms — tokenizing, batching, and response assembly stay
            sub-millisecond. Latency scales with token count: compare the tabs
            above.
          </>
        )}
      </p>

      <details className={styles.details}>
        <summary>
          Render your own capture (paste the <code>spans</code> JSON)
        </summary>
        <textarea
          className={styles.textarea}
          spellCheck={false}
          value={input}
          onChange={(e) => setInput(e.target.value)}
        />
        <div>
          <button className={styles.button} onClick={onRender} type="button">
            Render
          </button>
        </div>
        {error ? <div className={styles.err}>{error}</div> : null}
      </details>
    </div>
  );
}
