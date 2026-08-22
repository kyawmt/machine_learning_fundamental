import { useMemo, useState } from "react";
import { VizFrame } from "./VizFrame";
import { gaussian, mulberry32, type Frame } from "./plot";

/** age (years) and income (thousands) with correlation ≈ 0.72. */
function buildData() {
  const rnd = mulberry32(90210);
  return Array.from({ length: 110 }, () => {
    const z1 = gaussian(rnd);
    const z2 = gaussian(rnd);
    return {
      age: 45 + 9 * z1,
      income: 90 + 28 * (0.72 * z1 + Math.sqrt(1 - 0.72 ** 2) * z2),
    };
  });
}

function eig2(a: number, b: number, c: number) {
  const tr = a + c;
  const det = a * c - b * b;
  const disc = Math.sqrt(Math.max(tr * tr - 4 * det, 0));
  const l1 = (tr + disc) / 2;
  const l2 = (tr - disc) / 2;
  const vec = (l: number): [number, number] => {
    if (Math.abs(b) > 1e-12) {
      const n = Math.hypot(b, l - a);
      return [b / n, (l - a) / n];
    }
    return a >= c ? [1, 0] : [0, 1];
  };
  return { l1, l2, v1: vec(l1), v2: vec(l2) };
}

const F: Frame = { w: 340, h: 320, pad: { t: 16, r: 16, b: 34, l: 40 } };

export function PCALab() {
  const raw = useMemo(buildData, []);
  const [standardized, setStandardized] = useState(false);
  const [project, setProject] = useState(false);

  const stats = useMemo(() => {
    const mAge = raw.reduce((s, p) => s + p.age, 0) / raw.length;
    const mInc = raw.reduce((s, p) => s + p.income, 0) / raw.length;
    const sdAge = Math.sqrt(raw.reduce((s, p) => s + (p.age - mAge) ** 2, 0) / raw.length);
    const sdInc = Math.sqrt(raw.reduce((s, p) => s + (p.income - mInc) ** 2, 0) / raw.length);
    return { mAge, mInc, sdAge, sdInc };
  }, [raw]);

  const pts = useMemo(
    () =>
      raw.map((p) =>
        standardized
          ? { x: (p.age - stats.mAge) / stats.sdAge, y: (p.income - stats.mInc) / stats.sdInc }
          : { x: p.age - stats.mAge, y: p.income - stats.mInc },
      ),
    [raw, standardized, stats],
  );

  const { l1, l2, v1, v2 } = useMemo(() => {
    const n = pts.length;
    const a = pts.reduce((s, p) => s + p.x * p.x, 0) / n;
    const b = pts.reduce((s, p) => s + p.x * p.y, 0) / n;
    const c = pts.reduce((s, p) => s + p.y * p.y, 0) / n;
    return eig2(a, b, c);
  }, [pts]);

  const ev1 = l1 / (l1 + l2);

  // equal aspect: one shared half-extent for both axes so angles are honest
  const half = Math.max(...pts.flatMap((p) => [Math.abs(p.x), Math.abs(p.y)])) * 1.12;
  const cx = (F.pad.l + (F.w - F.pad.r)) / 2;
  const cy = (F.pad.t + (F.h - F.pad.b)) / 2;
  const boxW = F.w - F.pad.l - F.pad.r;
  const boxH = F.h - F.pad.t - F.pad.b;
  const scale = Math.min(boxW, boxH) / (2 * half);
  const px = (v: number) => cx + v * scale;
  const py = (v: number) => cy - v * scale;

  const arrow = (v: [number, number], len: number) => ({
    x2: px(v[0] * len),
    y2: py(v[1] * len),
  });
  const a1 = arrow(v1, Math.sqrt(l1) * 2.1);
  const a2 = arrow(v2, Math.sqrt(l2) * 2.1);

  // PC1 loading composition (how much of PC1 is each original feature)
  const load1 = Math.abs(v1[0]);
  const load2 = Math.abs(v1[1]);
  const loadSum = load1 + load2;

  return (
    <VizFrame
      title="PCA Lab"
      hint="age (years) vs income (thousands)"
      footer={
        <>
          On raw units PC1 is essentially "income" — it inherits the biggest variance, which here just means the
          biggest measurement unit. Standardize first and PC1 becomes the genuine{" "}
          <em>shared direction</em> of the two features. For two standardized features with correlation r, PC1 is
          always the ±45° line and explains (1+r)/2 of the variance — which is where the numbers on the right come
          from once you standardize.
        </>
      }
    >
      <div className="grid gap-4 sm:grid-cols-[minmax(0,1fr)_minmax(0,0.85fr)]">
        <div>
          <svg viewBox={`0 0 ${F.w} ${F.h}`} className="w-full text-line" role="img"
            aria-label="Scatter of two features with the principal component directions drawn as arrows">
            <rect x={F.pad.l} y={F.pad.t} width={boxW} height={boxH} fill="none" stroke="currentColor" strokeWidth="1" rx="6" />
            <line x1={F.pad.l} x2={F.w - F.pad.r} y1={cy} y2={cy} stroke="currentColor" strokeWidth="0.8" opacity="0.55" />
            <line y1={F.pad.t} y2={F.h - F.pad.b} x1={cx} x2={cx} stroke="currentColor" strokeWidth="0.8" opacity="0.55" />

            {/* PC1 line */}
            <line
              x1={px(-v1[0] * half * 2)} y1={py(-v1[1] * half * 2)}
              x2={px(v1[0] * half * 2)} y2={py(v1[1] * half * 2)}
              stroke="#818cf8" strokeWidth="1.2" strokeDasharray="4 4" opacity="0.6"
            />

            {pts.map((p, i) => {
              const t = p.x * v1[0] + p.y * v1[1];
              const proj = { x: t * v1[0], y: t * v1[1] };
              return (
                <g key={i}>
                  {project && (
                    <line x1={px(p.x)} y1={py(p.y)} x2={px(proj.x)} y2={py(proj.y)}
                      stroke="#f43f5e" strokeWidth="0.8" opacity="0.4" />
                  )}
                  <circle cx={px(p.x)} cy={py(p.y)} r="3" fill="#94a3b8" opacity={project ? 0.35 : 0.8} />
                  {project && <circle cx={px(proj.x)} cy={py(proj.y)} r="3" fill="#818cf8" opacity="0.95" />}
                </g>
              );
            })}

            <line x1={px(0)} y1={py(0)} x2={a1.x2} y2={a1.y2} stroke="#818cf8" strokeWidth="3" markerEnd="url(#pcarrow1)" />
            <line x1={px(0)} y1={py(0)} x2={a2.x2} y2={a2.y2} stroke="#f59e0b" strokeWidth="2.4" markerEnd="url(#pcarrow2)" />
            <defs>
              <marker id="pcarrow1" markerWidth="7" markerHeight="7" refX="5" refY="3.5" orient="auto">
                <path d="M0,0 L7,3.5 L0,7 z" fill="#818cf8" />
              </marker>
              <marker id="pcarrow2" markerWidth="7" markerHeight="7" refX="5" refY="3.5" orient="auto">
                <path d="M0,0 L7,3.5 L0,7 z" fill="#f59e0b" />
              </marker>
            </defs>

            <text x={F.w / 2} y={F.h - 8} textAnchor="middle" className="fill-faint text-[9.5px] font-semibold" opacity="0.8">
              {standardized ? "age (z-score)" : "age − mean (years)"}
            </text>
            <text x={11} y={F.h / 2} textAnchor="middle" transform={`rotate(-90 11 ${F.h / 2})`}
              className="fill-faint text-[9.5px] font-semibold" opacity="0.8">
              {standardized ? "income (z-score)" : "income − mean (thousands)"}
            </text>
          </svg>
          <p className="mt-1 text-[10px] text-faint">Axes share one units-per-pixel scale, so the arrow angles are real.</p>
        </div>

        <div className="space-y-3">
          <div className="flex flex-wrap gap-2">
            <button type="button" onClick={() => setStandardized((s) => !s)}
              className={`rounded-lg border px-3 py-1.5 text-[12px] font-bold transition-colors ${
                standardized
                  ? "border-accent bg-accent text-white"
                  : "border-line bg-surface-2 text-muted hover:text-ink"
              }`}>
              {standardized ? "✓ Standardized" : "Standardize features"}
            </button>
            <button type="button" onClick={() => setProject((p) => !p)}
              className={`rounded-lg border px-3 py-1.5 text-[12px] font-bold transition-colors ${
                project ? "border-accent bg-accent text-white" : "border-line bg-surface-2 text-muted hover:text-ink"
              }`}>
              {project ? "✓ Projected to PC1" : "Project onto PC1"}
            </button>
          </div>

          <div className="rounded-xl border border-line bg-surface-2 p-3">
            <div className="mb-2 text-[10.5px] font-bold uppercase tracking-[0.08em] text-faint">
              Explained variance
            </div>
            <Bar label="PC1" pct={ev1} color="#818cf8" />
            <Bar label="PC2" pct={1 - ev1} color="#f59e0b" />
          </div>

          <div className="rounded-xl border border-line bg-surface-2 p-3">
            <div className="mb-2 text-[10.5px] font-bold uppercase tracking-[0.08em] text-faint">
              What PC1 is made of
            </div>
            <Bar label="age" pct={load1 / loadSum} color="#38bdf8" />
            <Bar label="income" pct={load2 / loadSum} color="#10b981" />
            <p className="mt-1.5 text-[10.5px] leading-relaxed text-faint">
              {standardized
                ? "Balanced — PC1 is a real combination of both features."
                : "Lopsided — income is measured in bigger numbers, so it hijacks PC1 before you scale."}
            </p>
          </div>

          {project && (
            <div className="rounded-lg border border-rose-200 bg-rose-50/70 px-3 py-2 text-[0.83rem] text-rose-700 dark:border-rose-400/25 dark:bg-rose-400/8 dark:text-rose-300">
              The red segments are the information you throw away by keeping only PC1. That discarded distance is
              the reconstruction error.
            </div>
          )}
        </div>
      </div>
    </VizFrame>
  );
}

function Bar({ label, pct, color }: { label: string; pct: number; color: string }) {
  return (
    <div className="mb-1.5 flex items-center gap-2">
      <span className="w-12 shrink-0 font-mono text-[10px] text-faint">{label}</span>
      <div className="h-3.5 flex-1 overflow-hidden rounded-sm bg-surface">
        <div className="h-full rounded-sm transition-[width] duration-200"
          style={{ width: `${pct * 100}%`, background: color }} />
      </div>
      <span className="w-11 shrink-0 text-right font-mono text-[10px] font-bold text-ink tabular-nums">
        {(pct * 100).toFixed(1)}%
      </span>
    </div>
  );
}
