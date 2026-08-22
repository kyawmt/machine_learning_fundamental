import { useMemo, useState } from "react";
import { VizFrame, Slider } from "./VizFrame";
import { gaussian, linePath, makeScales, mulberry32, range, type Frame } from "./plot";

/* ---------- a real (tiny) polynomial least-squares fit ---------- */

function solve(A: number[][], b: number[]): number[] {
  const n = b.length;
  const M = A.map((row, i) => [...row, b[i]]);
  for (let col = 0; col < n; col++) {
    let piv = col;
    for (let r = col + 1; r < n; r++) if (Math.abs(M[r][col]) > Math.abs(M[piv][col])) piv = r;
    [M[col], M[piv]] = [M[piv], M[col]];
    const d = M[col][col] || 1e-12;
    for (let c = col; c <= n; c++) M[col][c] /= d;
    for (let r = 0; r < n; r++) {
      if (r === col) continue;
      const f = M[r][col];
      if (f === 0) continue;
      for (let c = col; c <= n; c++) M[r][c] -= f * M[col][c];
    }
  }
  return M.map((row) => row[n]);
}

/** Least squares on a polynomial basis, with a whisper of ridge for conditioning. */
function polyFit(xs: number[], ys: number[], degree: number): number[] {
  const p = degree + 1;
  const X = xs.map((x) => Array.from({ length: p }, (_, j) => Math.pow(x, j)));
  const XtX = Array.from({ length: p }, (_, i) =>
    Array.from({ length: p }, (_, j) => X.reduce((s, row) => s + row[i] * row[j], 0) + (i === j ? 1e-10 : 0)),
  );
  const Xty = Array.from({ length: p }, (_, i) => X.reduce((s, row, k) => s + row[i] * ys[k], 0));
  return solve(XtX, Xty);
}

const evalPoly = (w: number[], x: number) => w.reduce((s, c, i) => s + c * Math.pow(x, i), 0);
const rmse = (a: number[], b: number[]) =>
  Math.sqrt(a.reduce((s, v, i) => s + (v - b[i]) ** 2, 0) / a.length);

/** x is mapped to [-1,1] before fitting — same fit, far better conditioning. */
const toUnit = (x: number) => x * 2 - 1;
const trueFn = (x: number) => Math.sin(x * Math.PI * 1.65) * 0.85 + x * 0.35;

const MAX_DEGREE = 12;

const N_POINTS = 14;
const NOISE = 0.16;

function buildSample() {
  const rnd = mulberry32(344);
  const xs = range(N_POINTS, 0.02, 0.98);
  const train = xs.map((x) => ({ x, y: trueFn(x) + gaussian(rnd, 0, NOISE) }));
  const val = range(N_POINTS, 0.05, 0.95).map((x) => ({ x, y: trueFn(x) + gaussian(rnd, 0, NOISE) }));
  return { train, val };
}

const FIT: Frame = { w: 330, h: 250, pad: { t: 16, r: 14, b: 34, l: 38 } };
const ERR: Frame = { w: 330, h: 250, pad: { t: 16, r: 14, b: 34, l: 44 } };

export function OverfittingLab() {
  const { train, val } = useMemo(buildSample, []);
  const [degree, setDegree] = useState(3);

  const fits = useMemo(() => {
    const out: { d: number; w: number[]; trainErr: number; valErr: number }[] = [];
    const tx = train.map((p) => toUnit(p.x));
    const ty = train.map((p) => p.y);
    const vx = val.map((p) => toUnit(p.x));
    const vy = val.map((p) => p.y);
    for (let d = 1; d <= MAX_DEGREE; d++) {
      const w = polyFit(tx, ty, d);
      out.push({
        d,
        w,
        trainErr: rmse(tx.map((x) => evalPoly(w, x)), ty),
        valErr: rmse(vx.map((x) => evalPoly(w, x)), vy),
      });
    }
    return out;
  }, [train, val]);

  const active = fits[degree - 1];
  const bestDegree = fits.reduce((a, b) => (b.valErr < a.valErr ? b : a)).d;

  const yDomain: [number, number] = [-1.9, 1.9];
  const fs = makeScales(FIT, [0, 1], yDomain);
  const maxErr = Math.min(1.4, Math.max(...fits.map((f) => f.valErr)) * 1.08);
  const es = makeScales(ERR, [1, MAX_DEGREE], [0, maxErr]);

  const curve = range(220, 0, 1).map(
    (x) => [x, Math.max(yDomain[0], Math.min(yDomain[1], evalPoly(active.w, toUnit(x))))] as [number, number],
  );

  const gap = active.valErr - active.trainErr;
  const diagnosis =
    degree <= 2
      ? { t: "Underfitting", c: "text-amber-600 dark:text-amber-400", d: "Both errors are high — the model is too rigid to capture the shape. This is high bias." }
      : degree >= bestDegree + 3
        ? { t: "Overfitting", c: "text-rose-600 dark:text-rose-400", d: "Training error keeps falling while validation error climbs. The extra capacity is being spent memorising noise. This is high variance." }
        : { t: "Well fit", c: "text-emerald-600 dark:text-emerald-400", d: "Validation error is near its minimum and the gap to training error is small. This is roughly where you want to stop." };

  return (
    <VizFrame
      title="Overfitting Lab"
      hint="real polynomial least squares · 14 train / 14 validation points"
      footer={
        <>
          Validation error bottoms out at <strong className="text-ink">degree {bestDegree}</strong>. Notice that
          training error <em>never</em> goes back up — which is exactly why you cannot detect overfitting from the
          training curve alone.
        </>
      }
    >
      <div className="grid gap-4 sm:grid-cols-2">
        {/* Fit plot */}
        <div>
          <div className="mb-1 text-[10.5px] font-bold uppercase tracking-[0.08em] text-faint">
            Fitted model — degree {degree}
          </div>
          <svg viewBox={`0 0 ${FIT.w} ${FIT.h}`} className="w-full text-line" role="img"
            aria-label={`Polynomial of degree ${degree} fitted to the training points`}>
            <rect x={fs.x0} y={fs.y1} width={fs.x1 - fs.x0} height={fs.y0 - fs.y1} fill="none" stroke="currentColor" strokeWidth="1" />
            <line x1={fs.x0} x2={fs.x1} y1={fs.sy(0)} y2={fs.sy(0)} stroke="currentColor" strokeWidth="0.8" opacity="0.6" />
            <path d={linePath(range(160, 0, 1).map((x) => [x, trueFn(x)]), fs.sx, fs.sy)}
              fill="none" stroke="currentColor" strokeWidth="1.6" strokeDasharray="5 4" opacity="0.65" />
            <path d={linePath(curve, fs.sx, fs.sy)} fill="none" stroke="#818cf8" strokeWidth="2.6" strokeLinejoin="round" />
            {train.map((p, i) => (
              <circle key={`t${i}`} cx={fs.sx(p.x)} cy={fs.sy(p.y)} r="3.4" fill="#10b981" opacity="0.95" />
            ))}
            {val.map((p, i) => (
              <rect key={`v${i}`} x={fs.sx(p.x) - 2.8} y={fs.sy(p.y) - 2.8} width="5.6" height="5.6"
                fill="none" stroke="#f43f5e" strokeWidth="1.6" />
            ))}
            <text x={(fs.x0 + fs.x1) / 2} y={FIT.h - 8} textAnchor="middle" className="fill-faint text-[9.5px] font-semibold" opacity="0.8">
              x
            </text>
          </svg>
          <div className="mt-1 flex flex-wrap gap-x-3 gap-y-0.5 text-[10px] text-faint">
            <LegendDot color="#10b981" label="train point" />
            <LegendDot color="#f43f5e" label="validation point" square />
            <LegendDot color="#818cf8" label="fitted model" />
            <span>┈ true function</span>
          </div>
        </div>

        {/* Error plot */}
        <div>
          <div className="mb-1 text-[10.5px] font-bold uppercase tracking-[0.08em] text-faint">
            Error vs complexity
          </div>
          <svg viewBox={`0 0 ${ERR.w} ${ERR.h}`} className="w-full text-line" role="img"
            aria-label="Training error falls monotonically while validation error is U-shaped">
            <rect x={es.x0} y={es.y1} width={es.x1 - es.x0} height={es.y0 - es.y1} fill="none" stroke="currentColor" strokeWidth="1" />
            {[0.25, 0.5, 0.75, 1].filter((g) => g <= maxErr).map((g) => (
              <g key={g}>
                <line x1={es.x0} x2={es.x1} y1={es.sy(g)} y2={es.sy(g)} stroke="currentColor" strokeWidth="0.8" opacity="0.5" />
                <text x={es.x0 - 6} y={es.sy(g) + 3} textAnchor="end" className="fill-faint text-[8.5px]" opacity="0.7">
                  {g.toFixed(2)}
                </text>
              </g>
            ))}
            <line x1={es.sx(bestDegree)} x2={es.sx(bestDegree)} y1={es.y0} y2={es.y1}
              stroke="#10b981" strokeWidth="1.2" strokeDasharray="3 4" opacity="0.7" />
            <path d={linePath(fits.map((f) => [f.d, Math.min(f.trainErr, maxErr)]), es.sx, es.sy)}
              fill="none" stroke="#10b981" strokeWidth="2.4" />
            <path d={linePath(fits.map((f) => [f.d, Math.min(f.valErr, maxErr)]), es.sx, es.sy)}
              fill="none" stroke="#f43f5e" strokeWidth="2.4" />
            <line x1={es.sx(degree)} x2={es.sx(degree)} y1={es.y0} y2={es.y1} stroke="currentColor" strokeWidth="1.4" opacity="0.55" />
            <circle cx={es.sx(degree)} cy={es.sy(Math.min(active.trainErr, maxErr))} r="4.5" fill="#10b981" stroke="var(--surface)" strokeWidth="1.6" />
            <circle cx={es.sx(degree)} cy={es.sy(Math.min(active.valErr, maxErr))} r="4.5" fill="#f43f5e" stroke="var(--surface)" strokeWidth="1.6" />
            <text x={(es.x0 + es.x1) / 2} y={ERR.h - 8} textAnchor="middle" className="fill-faint text-[9.5px] font-semibold" opacity="0.8">
              polynomial degree →
            </text>
            <text x={11} y={ERR.h / 2} textAnchor="middle" transform={`rotate(-90 11 ${ERR.h / 2})`}
              className="fill-faint text-[9.5px] font-semibold" opacity="0.8">
              RMSE
            </text>
          </svg>
          <div className="mt-1 flex flex-wrap gap-x-3 text-[10px] text-faint">
            <LegendDot color="#10b981" label="training error" />
            <LegendDot color="#f43f5e" label="validation error" />
          </div>
        </div>
      </div>

      <div className="mt-4 space-y-3">
        <Slider
          label="Model complexity (polynomial degree)"
          value={degree}
          min={1}
          max={MAX_DEGREE}
          step={1}
          onChange={setDegree}
          display={`degree ${degree}`}
          leftLabel="too simple"
          rightLabel="too flexible"
        />
        <div className="grid grid-cols-3 gap-2">
          <Stat label="Train RMSE" v={active.trainErr.toFixed(3)} color="#10b981" />
          <Stat label="Val RMSE" v={active.valErr.toFixed(3)} color="#f43f5e" />
          <Stat label="Gap" v={gap.toFixed(3)} color="#818cf8" />
        </div>
        <p className="text-[0.88rem] text-muted">
          <strong className={`font-bold ${diagnosis.c}`}>{diagnosis.t}.</strong> {diagnosis.d}
        </p>
      </div>
    </VizFrame>
  );
}

function LegendDot({ color, label, square }: { color: string; label: string; square?: boolean }) {
  return (
    <span className="inline-flex items-center gap-1">
      <span
        className={square ? "size-2 border-2" : "size-2 rounded-full"}
        style={square ? { borderColor: color } : { background: color }}
        aria-hidden
      />
      {label}
    </span>
  );
}

function Stat({ label, v, color }: { label: string; v: string; color: string }) {
  return (
    <div className="rounded-lg border border-line bg-surface-2 px-2.5 py-1.5">
      <div className="flex items-center gap-1.5">
        <span className="size-2 rounded-full" style={{ background: color }} aria-hidden />
        <span className="text-[10px] font-bold uppercase tracking-[0.06em] text-faint">{label}</span>
      </div>
      <div className="font-mono text-[0.95rem] font-bold text-ink tabular-nums">{v}</div>
    </div>
  );
}
