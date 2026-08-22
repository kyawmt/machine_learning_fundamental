import { useState } from "react";
import { VizFrame, Slider } from "./VizFrame";

/**
 * With orthonormal features the penalised solutions have exact closed forms,
 * which is where the standard L1/L2 intuition comes from:
 *   ridge:  w / (1 + λ)                       — proportional shrinkage, never exactly 0
 *   lasso:  sign(w) · max(|w| − λ, 0)         — soft threshold, snaps small weights to 0
 * Real designs are correlated, so the picture is fuzzier, but the behaviour is the same.
 */

const OLS = [
  { name: "income", w: 0.92 },
  { name: "tenure", w: -0.71 },
  { name: "n_logins", w: 0.48 },
  { name: "age", w: 0.31 },
  { name: "city_pop", w: -0.19 },
  { name: "device", w: 0.12 },
  { name: "hour", w: -0.07 },
  { name: "referrer", w: 0.04 },
];

const ridge = (w: number, l: number) => w / (1 + l);
const lasso = (w: number, l: number) => Math.sign(w) * Math.max(Math.abs(w) - l, 0);
const elastic = (w: number, l: number, a = 0.5) =>
  (Math.sign(w) * Math.max(Math.abs(w) - a * l, 0)) / (1 + (1 - a) * l);

export function RegularizationLab() {
  const [lambda, setLambda] = useState(0.2);

  const rows = OLS.map((f) => ({
    ...f,
    l2: ridge(f.w, lambda),
    l1: lasso(f.w, lambda),
    en: elastic(f.w, lambda),
  }));

  const zerosL1 = rows.filter((r) => r.l1 === 0).length;
  const zerosL2 = rows.filter((r) => r.l2 === 0).length;
  const zerosEN = rows.filter((r) => r.en === 0).length;

  return (
    <VizFrame
      title="Regularization Lab"
      hint="λ = 0 → plain least squares"
      footer={
        <>
          Push λ up and watch L1 zero out features one at a time while L2 only shrinks them toward zero. That
          single difference is the whole reason L1 is described as doing feature selection — and the reason L2 is
          the safer default when you believe every feature carries a little signal. (The bars use the exact
          closed-form solutions for uncorrelated features; with correlated features the picture is fuzzier, but
          the behaviour is the same.)
        </>
      }
    >
      <Slider
        label="Regularization strength λ"
        value={lambda}
        min={0}
        max={1}
        step={0.01}
        onChange={setLambda}
        display={`λ = ${lambda.toFixed(2)}`}
        leftLabel="no penalty · low bias, high variance"
        rightLabel="strong penalty · high bias, low variance"
      />

      <div className="mt-4 grid gap-3 sm:grid-cols-3">
        <Column title="L2 · Ridge" sub={`${zerosL2} coefficients at zero`} rows={rows} pick="l2" color="#38bdf8" />
        <Column title="L1 · Lasso" sub={`${zerosL1} coefficients at zero`} rows={rows} pick="l1" color="#f59e0b" />
        <Column title="Elastic Net (α=0.5)" sub={`${zerosEN} coefficients at zero`} rows={rows} pick="en" color="#a78bfa" />
      </div>
    </VizFrame>
  );
}

function Column({
  title,
  sub,
  rows,
  pick,
  color,
}: {
  title: string;
  sub: string;
  rows: { name: string; w: number; l1: number; l2: number; en: number }[];
  pick: "l1" | "l2" | "en";
  color: string;
}) {
  return (
    <div className="rounded-xl border border-line bg-surface-2 p-3">
      <div className="mb-0.5 flex items-center gap-1.5">
        <span className="size-2 rounded-full" style={{ background: color }} aria-hidden />
        <span className="text-[11.5px] font-bold text-ink">{title}</span>
      </div>
      <div className="mb-2 text-[10px] font-semibold uppercase tracking-[0.06em] text-faint">{sub}</div>
      <div className="space-y-1">
        {rows.map((r) => {
          const v = r[pick];
          const isZero = v === 0;
          return (
            <div key={r.name} className="flex items-center gap-1.5">
              <span className="w-[62px] shrink-0 truncate font-mono text-[9.5px] text-faint">{r.name}</span>
              <div className="relative h-3.5 flex-1 rounded-sm bg-surface">
                <div className="absolute inset-y-0 left-1/2 w-px bg-line-strong" aria-hidden />
                {/* ghost of the unpenalised coefficient */}
                <div
                  className="absolute inset-y-0 rounded-sm border border-dashed"
                  style={{
                    borderColor: color,
                    opacity: 0.35,
                    left: r.w >= 0 ? "50%" : `${50 - Math.abs(r.w) * 50}%`,
                    width: `${Math.abs(r.w) * 50}%`,
                  }}
                  aria-hidden
                />
                <div
                  className="absolute inset-y-0 rounded-sm transition-[width,left] duration-150"
                  style={{
                    background: color,
                    left: v >= 0 ? "50%" : `${50 - Math.abs(v) * 50}%`,
                    width: `${Math.abs(v) * 50}%`,
                  }}
                  aria-hidden
                />
              </div>
              <span
                className={`w-[42px] shrink-0 text-right font-mono text-[9.5px] tabular-nums ${
                  isZero ? "font-bold text-rose-500" : "text-muted"
                }`}
              >
                {isZero ? "0" : v.toFixed(2)}
              </span>
            </div>
          );
        })}
      </div>
    </div>
  );
}
