import { useEffect, useRef, useState } from "react";
import { VizFrame, Slider } from "./VizFrame";
import { linePath, makeScales, range, type Frame } from "./plot";

/**
 * J(θ) = ½(θ − 2)²  ⇒  ∇J = (θ − 2)  ⇒  θ ← θ − η(θ − 2).
 * The error shrinks by a factor of (1 − η) each step, so the regimes are exact:
 * η<1 monotone, η=1 one step, 1<η<2 oscillating, η=2 stuck, η>2 divergent.
 */
const MIN = 2;
const J = (t: number) => 0.5 * (t - MIN) ** 2;
const dJ = (t: number) => t - MIN;
const START = -3.2;

const F: Frame = { w: 350, h: 250, pad: { t: 16, r: 14, b: 34, l: 40 } };
const L: Frame = { w: 350, h: 250, pad: { t: 16, r: 14, b: 34, l: 44 } };

export function GradientDescentLab() {
  const [lr, setLr] = useState(0.3);
  const [path, setPath] = useState<number[]>([START]);
  const [running, setRunning] = useState(false);
  const timer = useRef<number | null>(null);

  const theta = path[path.length - 1];
  const diverged = !Number.isFinite(theta) || Math.abs(theta) > 1e4;

  const step = () => {
    setPath((p) => {
      const last = p[p.length - 1];
      if (!Number.isFinite(last) || Math.abs(last) > 1e4) return p;
      const next = last - lr * dJ(last);
      return [...p, next].slice(-60);
    });
  };

  const reset = (nextLr?: number) => {
    setRunning(false);
    setPath([START]);
    if (nextLr !== undefined) setLr(nextLr);
  };

  useEffect(() => {
    if (!running) return;
    timer.current = window.setInterval(() => {
      setPath((p) => {
        const last = p[p.length - 1];
        if (!Number.isFinite(last) || Math.abs(last) > 1e4) return p;
        if (p.length > 55) return p;
        return [...p, last - lr * dJ(last)];
      });
    }, 220);
    return () => {
      if (timer.current) window.clearInterval(timer.current);
    };
  }, [running, lr]);

  useEffect(() => {
    if (path.length > 55 || diverged) setRunning(false);
  }, [path.length, diverged]);

  const xDom: [number, number] = [-4.5, 8.5];
  const yDom: [number, number] = [0, 22];
  const s = makeScales(F, xDom, yDom);
  const curve = range(160, xDom[0], xDom[1]).map((x) => [x, J(x)] as [number, number]);

  const clamp = (v: number) => Math.max(xDom[0], Math.min(xDom[1], v));
  const losses = path.map((t) => (Number.isFinite(t) ? Math.min(J(t), 1e6) : 1e6));
  const lossMax = Math.max(1, ...losses.filter((v) => v < 1e6).slice(0, 40));
  const ls = makeScales(L, [0, Math.max(9, path.length - 1)], [0, lossMax * 1.1]);

  const regime =
    lr < 0.02
      ? { t: "Far too small", c: "text-amber-600 dark:text-amber-400", d: "It converges, but you'd wait forever. In practice this looks like a loss curve that is almost flat — easy to mistake for a bug." }
      : lr < 0.9
        ? { t: "Healthy", c: "text-emerald-600 dark:text-emerald-400", d: "Each step moves smoothly downhill and the loss decreases monotonically. This is the shape you want to see." }
        : lr < 1.05
          ? { t: "Aggressive but fine", c: "text-emerald-600 dark:text-emerald-400", d: "At η = 1 this particular quadratic is solved in a single step. Real losses are not this forgiving." }
          : lr < 2
            ? { t: "Oscillating", c: "text-orange-600 dark:text-orange-400", d: "It overshoots the minimum every step and bounces across it, but the bounces shrink so it still converges — slowly and noisily." }
            : { t: "Diverging", c: "text-rose-600 dark:text-rose-400", d: "Each step overshoots by more than it started with. The loss explodes to NaN or Inf. This is the classic 'learning rate too high' failure." };

  return (
    <VizFrame
      title="Gradient Descent Lab"
      hint="θ ← θ − η∇J(θ)"
      footer={
        <>
          The gradient points in the direction of <em>steepest increase</em>, so we subtract it. The learning rate
          η controls how far along that direction we move — it does not change the direction, only the step size.
        </>
      }
    >
      <div className="grid gap-4 sm:grid-cols-2">
        <div>
          <div className="mb-1 text-[10.5px] font-bold uppercase tracking-[0.08em] text-faint">
            Parameter space
          </div>
          <svg viewBox={`0 0 ${F.w} ${F.h}`} className="w-full text-line" role="img"
            aria-label="A point descending a parabola toward its minimum">
            <rect x={s.x0} y={s.y1} width={s.x1 - s.x0} height={s.y0 - s.y1} fill="none" stroke="currentColor" strokeWidth="1" />
            <line x1={s.sx(MIN)} x2={s.sx(MIN)} y1={s.y0} y2={s.y1} stroke="#10b981" strokeWidth="1.2" strokeDasharray="3 4" opacity="0.75" />
            <text x={s.sx(MIN)} y={s.y1 + 11} textAnchor="middle" className="text-[9px] font-bold" fill="#10b981">
              minimum
            </text>
            <path d={linePath(curve, s.sx, s.sy)} fill="none" stroke="currentColor" strokeWidth="2" opacity="0.85" />

            {/* trajectory */}
            {path.slice(0, -1).map((t, i) => {
              const a = clamp(t);
              const b = clamp(path[i + 1]);
              return (
                <line key={i} x1={s.sx(a)} y1={s.sy(Math.min(J(a), yDom[1]))} x2={s.sx(b)} y2={s.sy(Math.min(J(b), yDom[1]))}
                  stroke="#818cf8" strokeWidth="1.5" opacity={0.25 + (0.7 * i) / Math.max(1, path.length)} />
              );
            })}
            {path.map((t, i) => (
              <circle key={i} cx={s.sx(clamp(t))} cy={s.sy(Math.min(J(clamp(t)), yDom[1]))}
                r={i === path.length - 1 ? 6 : 3}
                fill={i === path.length - 1 ? "#818cf8" : "#a5b4fc"}
                stroke={i === path.length - 1 ? "var(--surface)" : "none"} strokeWidth="2"
                opacity={i === path.length - 1 ? 1 : 0.55} />
            ))}
            <text x={(s.x0 + s.x1) / 2} y={F.h - 8} textAnchor="middle" className="fill-faint text-[9.5px] font-semibold" opacity="0.8">
              θ (parameter)
            </text>
            <text x={11} y={F.h / 2} textAnchor="middle" transform={`rotate(-90 11 ${F.h / 2})`}
              className="fill-faint text-[9.5px] font-semibold" opacity="0.8">
              J(θ) — loss
            </text>
          </svg>
        </div>

        <div>
          <div className="mb-1 text-[10.5px] font-bold uppercase tracking-[0.08em] text-faint">
            Loss curve (what you actually see while training)
          </div>
          <svg viewBox={`0 0 ${L.w} ${L.h}`} className="w-full text-line" role="img" aria-label="Loss against iteration">
            <rect x={ls.x0} y={ls.y1} width={ls.x1 - ls.x0} height={ls.y0 - ls.y1} fill="none" stroke="currentColor" strokeWidth="1" />
            <path d={linePath(losses.map((v, i) => [i, Math.min(v, lossMax * 1.1)] as [number, number]), ls.sx, ls.sy)}
              fill="none" stroke="#f43f5e" strokeWidth="2.4" strokeLinejoin="round" />
            {losses.map((v, i) => (
              <circle key={i} cx={ls.sx(i)} cy={ls.sy(Math.min(v, lossMax * 1.1))} r="2.6" fill="#f43f5e" opacity="0.85" />
            ))}
            <text x={(ls.x0 + ls.x1) / 2} y={L.h - 8} textAnchor="middle" className="fill-faint text-[9.5px] font-semibold" opacity="0.8">
              iteration →
            </text>
            <text x={11} y={L.h / 2} textAnchor="middle" transform={`rotate(-90 11 ${L.h / 2})`}
              className="fill-faint text-[9.5px] font-semibold" opacity="0.8">
              loss
            </text>
          </svg>
        </div>
      </div>

      <div className="mt-4 space-y-3">
        <Slider
          label="Learning rate η"
          value={lr}
          min={0.01}
          max={2.4}
          step={0.01}
          onChange={(v) => {
            setRunning(false);
            setLr(v);
            setPath([START]);
          }}
          display={`η = ${lr.toFixed(2)}`}
          leftLabel="too slow"
          rightLabel="diverges"
        />

        <div className="flex flex-wrap items-center gap-2">
          <button type="button" onClick={step} disabled={diverged}
            className="rounded-lg border border-line bg-surface-2 px-3 py-1.5 text-[12.5px] font-bold text-muted transition-colors hover:text-ink disabled:opacity-40">
            Step
          </button>
          <button type="button" onClick={() => setRunning((r) => !r)} disabled={diverged}
            className="rounded-lg bg-accent px-3 py-1.5 text-[12.5px] font-bold text-white transition-opacity hover:opacity-90 disabled:opacity-40">
            {running ? "Pause" : "Run"}
          </button>
          <button type="button" onClick={() => reset()}
            className="rounded-lg border border-line bg-surface-2 px-3 py-1.5 text-[12.5px] font-bold text-muted transition-colors hover:text-ink">
            Reset
          </button>
          <div className="ml-auto flex gap-1.5">
            {[
              { l: "0.05", v: 0.05 },
              { l: "0.3", v: 0.3 },
              { l: "1.6", v: 1.6 },
              { l: "2.2", v: 2.2 },
            ].map((p) => (
              <button key={p.l} type="button" onClick={() => reset(p.v)}
                className="rounded-md border border-line bg-surface px-2 py-1 font-mono text-[11px] font-semibold text-faint hover:text-ink">
                η={p.l}
              </button>
            ))}
          </div>
        </div>

        <div className="grid grid-cols-3 gap-2 font-mono text-[0.9rem]">
          <Box label="step" v={String(path.length - 1)} />
          <Box label="θ" v={diverged ? "±∞" : theta.toFixed(4)} />
          <Box label="loss" v={diverged ? "NaN" : J(theta).toFixed(4)} />
        </div>

        <p className="text-[0.88rem] text-muted">
          <strong className={`font-bold ${regime.c}`}>{regime.t}.</strong> {regime.d}
        </p>
      </div>
    </VizFrame>
  );
}

function Box({ label, v }: { label: string; v: string }) {
  return (
    <div className="rounded-lg border border-line bg-surface-2 px-2.5 py-1.5">
      <div className="font-sans text-[10px] font-bold uppercase tracking-[0.06em] text-faint">{label}</div>
      <div className="font-bold text-ink tabular-nums">{v}</div>
    </div>
  );
}
