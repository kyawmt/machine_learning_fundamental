import { useMemo, useState } from "react";
import { VizFrame, Slider } from "./VizFrame";
import { linePath, makeScales, range, type Frame } from "./plot";

const F: Frame = { w: 560, h: 250, pad: { t: 16, r: 14, b: 34, l: 44 } };

const IRREDUCIBLE = 0.09;
const biasSq = (c: number) => 0.86 * Math.exp(-3.4 * c) + 0.015;
const variance = (c: number) => 0.035 * Math.exp(3.15 * c) - 0.03;
const total = (c: number) => biasSq(c) + Math.max(variance(c), 0) + IRREDUCIBLE;

export function BiasVarianceLab() {
  const [c, setC] = useState(0.5);

  const { curves, optimum } = useMemo(() => {
    const xs = range(121, 0, 1);
    const grid = xs.map((x) => ({ x, b: biasSq(x), v: Math.max(variance(x), 0), t: total(x) }));
    let best = grid[0];
    for (const g of grid) if (g.t < best.t) best = g;
    return { curves: grid, optimum: best };
  }, []);

  const yMax = 1.05;
  const { sx, sy, x0, x1, y0 } = makeScales(F, [0, 1], [0, yMax]);

  const b = biasSq(c);
  const v = Math.max(variance(c), 0);
  const t = b + v + IRREDUCIBLE;

  const zone =
    c < optimum.x - 0.12 ? "underfitting" : c > optimum.x + 0.12 ? "overfitting" : "sweet spot";
  const zoneStyle =
    zone === "underfitting"
      ? "text-amber-600 dark:text-amber-400"
      : zone === "overfitting"
        ? "text-sky-600 dark:text-sky-400"
        : "text-emerald-600 dark:text-emerald-400";

  const complexityLabel = ["Linear model", "Shallow tree", "Tuned ensemble", "Deep tree", "Memorizing model"][
    Math.min(4, Math.floor(c * 5))
  ];

  return (
    <VizFrame
      title="Bias–Variance Lab"
      hint="drag the slider"
      footer={
        <>
          Total expected error = <strong className="text-ink">bias² + variance + irreducible error</strong>.
          The irreducible part (noise in the data itself) is the flat floor at {IRREDUCIBLE.toFixed(2)} — no model
          gets below it. Tuning only moves you along the bias/variance trade.
        </>
      }
    >
      <svg viewBox={`0 0 ${F.w} ${F.h}`} className="w-full text-line" role="img"
        aria-label="Bias squared falls and variance rises as model complexity increases; total error is U-shaped.">
        {/* grid */}
        {[0, 0.25, 0.5, 0.75, 1].map((g) => (
          <line key={g} x1={x0} x2={x1} y1={sy(g)} y2={sy(g)} stroke="currentColor" strokeWidth="1" opacity="0.5" />
        ))}
        {/* irreducible floor */}
        <line x1={x0} x2={x1} y1={sy(IRREDUCIBLE)} y2={sy(IRREDUCIBLE)}
          stroke="currentColor" strokeWidth="1.5" strokeDasharray="4 4" opacity="0.9" />
        <text x={x1 - 4} y={sy(IRREDUCIBLE) - 5} textAnchor="end" className="fill-faint text-[9px]" opacity="0.75">
          irreducible error
        </text>

        {/* axes */}
        <line x1={x0} x2={x1} y1={y0} y2={y0} stroke="currentColor" strokeWidth="1.5" />
        <text x={(x0 + x1) / 2} y={F.h - 8} textAnchor="middle" className="fill-faint text-[10px] font-semibold" opacity="0.8">
          model complexity →
        </text>
        <text x={12} y={F.h / 2} textAnchor="middle" transform={`rotate(-90 12 ${F.h / 2})`}
          className="fill-faint text-[10px] font-semibold" opacity="0.8">
          error
        </text>
        {[0, 0.5, 1].map((g) => (
          <text key={g} x={x0 - 7} y={sy(g) + 3} textAnchor="end" className="fill-faint text-[9px]" opacity="0.7">
            {g.toFixed(1)}
          </text>
        ))}

        {/* optimum marker */}
        <line x1={sx(optimum.x)} x2={sx(optimum.x)} y1={sy(0)} y2={sy(yMax)}
          stroke="#10b981" strokeWidth="1.2" strokeDasharray="3 4" opacity="0.7" />
        <text x={sx(optimum.x)} y={sy(yMax) + 10} textAnchor="middle" className="text-[9px] font-bold" fill="#10b981">
          optimum
        </text>

        {/* curves */}
        <path d={linePath(curves.map((g) => [g.x, g.b]), sx, sy)} fill="none" stroke="#f59e0b" strokeWidth="2.4" />
        <path d={linePath(curves.map((g) => [g.x, g.v]), sx, sy)} fill="none" stroke="#38bdf8" strokeWidth="2.4" />
        <path d={linePath(curves.map((g) => [g.x, g.t]), sx, sy)} fill="none" stroke="#818cf8" strokeWidth="3" />

        {/* current position */}
        <line x1={sx(c)} x2={sx(c)} y1={sy(0)} y2={sy(yMax)} stroke="currentColor" strokeWidth="1.5" opacity="0.55" />
        <circle cx={sx(c)} cy={sy(b)} r="4.5" fill="#f59e0b" stroke="var(--surface)" strokeWidth="1.6" />
        <circle cx={sx(c)} cy={sy(v)} r="4.5" fill="#38bdf8" stroke="var(--surface)" strokeWidth="1.6" />
        <circle cx={sx(c)} cy={sy(t)} r="5.5" fill="#818cf8" stroke="var(--surface)" strokeWidth="1.8" />
      </svg>

      <div className="mt-3 space-y-3">
        <Slider
          label="Model complexity"
          value={c}
          min={0}
          max={1}
          step={0.01}
          onChange={setC}
          display={complexityLabel}
          leftLabel="simple"
          rightLabel="complex"
        />

        <div className="grid grid-cols-2 gap-2 sm:grid-cols-4">
          <Readout color="#f59e0b" label="Bias²" value={b} />
          <Readout color="#38bdf8" label="Variance" value={v} />
          <Readout color="#94a3b8" label="Irreducible" value={IRREDUCIBLE} />
          <Readout color="#818cf8" label="Total error" value={t} strong />
        </div>

        <p className="text-[0.88rem] text-muted">
          You are in the <strong className={`font-bold ${zoneStyle}`}>{zone}</strong> zone.{" "}
          {zone === "underfitting"
            ? "Bias dominates: the model can't represent the pattern. Train error and validation error are both high."
            : zone === "overfitting"
              ? "Variance dominates: the model tracks noise in this particular training sample. Train error keeps dropping while validation error climbs."
              : "Bias and variance are balanced — validation error is at its minimum. This is what tuning is trying to find."}
        </p>
      </div>
    </VizFrame>
  );
}

function Readout({
  color,
  label,
  value,
  strong,
}: {
  color: string;
  label: string;
  value: number;
  strong?: boolean;
}) {
  return (
    <div className={`rounded-lg border border-line bg-surface-2 px-2.5 py-1.5 ${strong ? "ring-1 ring-indigo-300 dark:ring-indigo-400/30" : ""}`}>
      <div className="flex items-center gap-1.5">
        <span className="size-2 rounded-full" style={{ background: color }} aria-hidden />
        <span className="text-[10px] font-bold uppercase tracking-[0.06em] text-faint">{label}</span>
      </div>
      <div className="font-mono text-[0.95rem] font-bold text-ink">{value.toFixed(2)}</div>
    </div>
  );
}
