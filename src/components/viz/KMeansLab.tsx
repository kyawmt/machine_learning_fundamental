import { useEffect, useMemo, useRef, useState } from "react";
import { VizFrame } from "./VizFrame";
import { gaussian, linePath, makeScales, mulberry32, type Frame } from "./plot";

type Pt = { x: number; y: number };

function buildPoints(): Pt[] {
  const rnd = mulberry32(4242);
  const blobs: [number, number, number][] = [
    [0.24, 0.32, 0.075],
    [0.74, 0.26, 0.085],
    [0.52, 0.76, 0.07],
  ];
  const pts: Pt[] = [];
  for (const [cx, cy, sd] of blobs) {
    for (let i = 0; i < 42; i++) {
      pts.push({
        x: Math.min(0.97, Math.max(0.03, gaussian(rnd, cx, sd))),
        y: Math.min(0.97, Math.max(0.03, gaussian(rnd, cy, sd))),
      });
    }
  }
  return pts;
}

const d2 = (a: Pt, b: Pt) => (a.x - b.x) ** 2 + (a.y - b.y) ** 2;

function assign(pts: Pt[], cents: Pt[]): number[] {
  return pts.map((p) => {
    let best = 0;
    let bd = Infinity;
    cents.forEach((c, i) => {
      const dd = d2(p, c);
      if (dd < bd) {
        bd = dd;
        best = i;
      }
    });
    return best;
  });
}

function recompute(pts: Pt[], labels: number[], k: number, prev: Pt[]): Pt[] {
  return Array.from({ length: k }, (_, i) => {
    const members = pts.filter((_, j) => labels[j] === i);
    if (members.length === 0) return prev[i]; // empty cluster keeps its position
    return {
      x: members.reduce((s, p) => s + p.x, 0) / members.length,
      y: members.reduce((s, p) => s + p.y, 0) / members.length,
    };
  });
}

const inertia = (pts: Pt[], labels: number[], cents: Pt[]) =>
  pts.reduce((s, p, i) => s + d2(p, cents[labels[i]]), 0);

/** Run to convergence from several seeds and keep the best — this is what `n_init` does. */
function bestRun(pts: Pt[], k: number, restarts = 8): number {
  let best = Infinity;
  for (let r = 0; r < restarts; r++) {
    const rnd = mulberry32(1000 + r * 97 + k * 13);
    let cents = Array.from({ length: k }, () => pts[Math.floor(rnd() * pts.length)]);
    let labels = assign(pts, cents);
    for (let it = 0; it < 40; it++) {
      const nc = recompute(pts, labels, k, cents);
      const nl = assign(pts, nc);
      const stable = nl.every((v, i) => v === labels[i]);
      cents = nc;
      labels = nl;
      if (stable) break;
    }
    best = Math.min(best, inertia(pts, labels, cents));
  }
  return best;
}

const PALETTE = ["#818cf8", "#10b981", "#f59e0b", "#f43f5e", "#38bdf8"];
const S: Frame = { w: 320, h: 300, pad: { t: 12, r: 12, b: 12, l: 12 } };
const E: Frame = { w: 320, h: 300, pad: { t: 18, r: 14, b: 36, l: 46 } };

export function KMeansLab() {
  const pts = useMemo(buildPoints, []);
  const [k, setK] = useState(3);
  const [seed, setSeed] = useState(1);
  const [phase, setPhase] = useState<"assign" | "update">("assign");
  const [iter, setIter] = useState(0);
  const [converged, setConverged] = useState(false);
  const [running, setRunning] = useState(false);

  const init = useMemo(() => {
    const rnd = mulberry32(seed * 7919 + k);
    const picked = new Set<number>();
    while (picked.size < k) picked.add(Math.floor(rnd() * pts.length));
    return [...picked].map((i) => ({ ...pts[i] }));
  }, [pts, k, seed]);

  const [cents, setCents] = useState<Pt[]>(init);
  const [labels, setLabels] = useState<number[]>(() => assign(pts, init));

  useEffect(() => {
    setCents(init);
    setLabels(assign(pts, init));
    setPhase("update");
    setIter(0);
    setConverged(false);
    setRunning(false);
  }, [init, pts]);

  const step = () => {
    if (converged) return;
    if (phase === "assign") {
      const nl = assign(pts, cents);
      const same = nl.every((v, i) => v === labels[i]);
      setLabels(nl);
      setPhase("update");
      if (same) setConverged(true);
    } else {
      setCents(recompute(pts, labels, k, cents));
      setPhase("assign");
      setIter((i) => i + 1);
    }
  };

  const timer = useRef<number | null>(null);
  useEffect(() => {
    if (!running || converged) return;
    timer.current = window.setInterval(step, 620);
    return () => {
      if (timer.current) window.clearInterval(timer.current);
    };
  });
  useEffect(() => {
    if (converged) setRunning(false);
  }, [converged]);

  const elbow = useMemo(
    () => Array.from({ length: 6 }, (_, i) => ({ k: i + 1, wcss: bestRun(pts, i + 1) })),
    [pts],
  );

  const s = makeScales(S, [0, 1], [0, 1]);
  const maxW = elbow[0].wcss;
  const es = makeScales(E, [1, 6], [0, maxW * 1.05]);
  const cur = inertia(pts, labels, cents);

  return (
    <VizFrame
      title="K-Means Lab"
      hint="126 points · 3 true blobs"
      footer={
        <>
          Re-seed a few times with K = 3: the algorithm usually finds the three blobs, but a bad initialisation can
          split one blob and merge two others. That is why libraries run it{" "}
          <span className="inline-code">n_init</span> times and keep the lowest inertia.
        </>
      }
    >
      <div className="grid gap-4 sm:grid-cols-2">
        <div>
          <div className="mb-1 flex items-baseline justify-between">
            <span className="text-[10.5px] font-bold uppercase tracking-[0.08em] text-faint">
              Iteration {iter} · next: {converged ? "converged" : phase}
            </span>
            <span className="font-mono text-[10.5px] font-bold text-accent">
              inertia {cur.toFixed(3)}
            </span>
          </div>
          <svg viewBox={`0 0 ${S.w} ${S.h}`} className="w-full text-line" role="img"
            aria-label="Scatter plot of points coloured by their current cluster assignment">
            <rect x={s.x0} y={s.y1} width={s.x1 - s.x0} height={s.y0 - s.y1} fill="none" stroke="currentColor" strokeWidth="1" rx="6" />
            {pts.map((p, i) => (
              <circle key={i} cx={s.sx(p.x)} cy={s.sy(p.y)} r="3.2"
                fill={PALETTE[labels[i] % PALETTE.length]} opacity="0.75" />
            ))}
            {cents.map((c, i) => (
              <g key={i}>
                <circle cx={s.sx(c.x)} cy={s.sy(c.y)} r="10" fill={PALETTE[i % PALETTE.length]} opacity="0.22" />
                <path
                  d={`M${s.sx(c.x) - 6},${s.sy(c.y) - 6} L${s.sx(c.x) + 6},${s.sy(c.y) + 6} M${s.sx(c.x) + 6},${s.sy(c.y) - 6} L${s.sx(c.x) - 6},${s.sy(c.y) + 6}`}
                  stroke={PALETTE[i % PALETTE.length]} strokeWidth="3" strokeLinecap="round"
                />
              </g>
            ))}
          </svg>
        </div>

        <div>
          <div className="mb-1 text-[10.5px] font-bold uppercase tracking-[0.08em] text-faint">
            Elbow plot — inertia vs K
          </div>
          <svg viewBox={`0 0 ${E.w} ${E.h}`} className="w-full text-line" role="img"
            aria-label="Within-cluster sum of squares against number of clusters, with a bend at K equals 3">
            <rect x={es.x0} y={es.y1} width={es.x1 - es.x0} height={es.y0 - es.y1} fill="none" stroke="currentColor" strokeWidth="1" />
            <line x1={es.sx(3)} x2={es.sx(3)} y1={es.y0} y2={es.y1} stroke="#10b981" strokeWidth="1.2" strokeDasharray="3 4" opacity="0.7" />
            <text x={es.sx(3)} y={es.y1 + 11} textAnchor="middle" className="text-[9px] font-bold" fill="#10b981">
              elbow
            </text>
            <path d={linePath(elbow.map((e) => [e.k, e.wcss]), es.sx, es.sy)} fill="none" stroke="#818cf8" strokeWidth="2.4" />
            {elbow.map((e) => (
              <circle key={e.k} cx={es.sx(e.k)} cy={es.sy(e.wcss)} r={e.k === k ? 5.5 : 3.5}
                fill={e.k === k ? "#818cf8" : "#a5b4fc"} stroke={e.k === k ? "var(--surface)" : "none"} strokeWidth="2" />
            ))}
            {elbow.map((e) => (
              <text key={e.k} x={es.sx(e.k)} y={E.h - 20} textAnchor="middle" className="fill-faint text-[9px]" opacity="0.75">
                {e.k}
              </text>
            ))}
            <text x={(es.x0 + es.x1) / 2} y={E.h - 6} textAnchor="middle" className="fill-faint text-[9.5px] font-semibold" opacity="0.8">
              K
            </text>
            <text x={12} y={E.h / 2} textAnchor="middle" transform={`rotate(-90 12 ${E.h / 2})`}
              className="fill-faint text-[9.5px] font-semibold" opacity="0.8">
              inertia (WCSS)
            </text>
          </svg>
          <p className="mt-1 text-[10.5px] leading-relaxed text-faint">
            Inertia always falls as K rises — at K = N it hits zero. You pick the bend, not the minimum.
          </p>
        </div>
      </div>

      <div className="mt-4 flex flex-wrap items-center gap-2">
        <div className="flex items-center gap-1.5">
          <span className="text-[11px] font-bold uppercase tracking-[0.07em] text-faint">K</span>
          {[2, 3, 4, 5].map((v) => (
            <button key={v} type="button" onClick={() => setK(v)}
              className={`size-7 rounded-md border text-[12px] font-bold transition-colors ${
                k === v
                  ? "border-accent bg-accent text-white"
                  : "border-line bg-surface-2 text-muted hover:text-ink"
              }`}>
              {v}
            </button>
          ))}
        </div>
        <button type="button" onClick={step} disabled={converged}
          className="rounded-lg border border-line bg-surface-2 px-3 py-1.5 text-[12.5px] font-bold text-muted transition-colors hover:text-ink disabled:opacity-40">
          Step ({converged ? "done" : phase})
        </button>
        <button type="button" onClick={() => setRunning((r) => !r)} disabled={converged}
          className="rounded-lg bg-accent px-3 py-1.5 text-[12.5px] font-bold text-white transition-opacity hover:opacity-90 disabled:opacity-40">
          {running ? "Pause" : "Run"}
        </button>
        <button type="button" onClick={() => setSeed((x) => x + 1)}
          className="rounded-lg border border-line bg-surface-2 px-3 py-1.5 text-[12.5px] font-bold text-muted transition-colors hover:text-ink">
          ↺ New initialisation
        </button>
        {converged && (
          <span className="rounded-md border border-emerald-300 bg-emerald-50 px-2 py-1 text-[11.5px] font-bold text-emerald-700 dark:border-emerald-400/35 dark:bg-emerald-500/10 dark:text-emerald-300">
            Converged in {iter} iterations
          </span>
        )}
      </div>
    </VizFrame>
  );
}
