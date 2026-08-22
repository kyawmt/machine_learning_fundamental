import { useState } from "react";
import { VizFrame } from "./VizFrame";
import { linePath, makeScales, range, type Frame } from "./plot";

/* ------------------------------------------------------------------ */
/* Train / validation / test split                                      */
/* ------------------------------------------------------------------ */

interface Preset {
  name: string;
  parts: { label: string; pct: number; tone: "train" | "val" | "test" }[];
  when: string;
}

const PRESETS: Preset[] = [
  {
    name: "80 / 10 / 10",
    parts: [
      { label: "Train", pct: 80, tone: "train" },
      { label: "Val", pct: 10, tone: "val" },
      { label: "Test", pct: 10, tone: "test" },
    ],
    when: "The default for a reasonably large dataset (tens of thousands of rows and up). Enough validation data for the score to be stable, most of the data still goes to fitting.",
  },
  {
    name: "70 / 15 / 15",
    parts: [
      { label: "Train", pct: 70, tone: "train" },
      { label: "Val", pct: 15, tone: "val" },
      { label: "Test", pct: 15, tone: "test" },
    ],
    when: "Smaller datasets, or when you need tighter confidence on the held-out scores. You trade training data for a less noisy estimate.",
  },
  {
    name: "80 / 20 + CV",
    parts: [
      { label: "Train + CV folds", pct: 80, tone: "train" },
      { label: "Test", pct: 20, tone: "test" },
    ],
    when: "The usual setup on small data: no fixed validation split — cross-validation carves folds out of the 80% for every tuning decision, and the 20% test set is opened once at the end.",
  },
  {
    name: "Time-ordered",
    parts: [
      { label: "Train (past)", pct: 70, tone: "train" },
      { label: "Val (later)", pct: 15, tone: "val" },
      { label: "Test (latest)", pct: 15, tone: "test" },
    ],
    when: "Anything with a time dimension: forecasting, churn, fraud, recommendations. Splits must respect chronology — train on the past, evaluate on the future, never the reverse.",
  },
];

const TONES = {
  train: "bg-indigo-500/85 text-white",
  val: "bg-amber-500/85 text-white",
  test: "bg-rose-500/85 text-white",
} as const;

export function SplitDiagram() {
  const [i, setI] = useState(0);
  const p = PRESETS[i];

  return (
    <VizFrame title="Split Ratios" hint="pick a strategy">
      <div className="mb-3 flex flex-wrap gap-1.5">
        {PRESETS.map((preset, idx) => (
          <button
            key={preset.name}
            type="button"
            onClick={() => setI(idx)}
            className={`rounded-lg border px-2.5 py-1 text-[12px] font-bold transition-colors ${
              i === idx
                ? "border-accent bg-accent text-white"
                : "border-line bg-surface-2 text-muted hover:text-ink"
            }`}
          >
            {preset.name}
          </button>
        ))}
      </div>

      <div className="mb-1 text-[10.5px] font-bold uppercase tracking-[0.08em] text-faint">
        Full dataset (100%)
      </div>
      <div className="flex h-11 gap-1 overflow-hidden rounded-lg">
        {p.parts.map((part) => (
          <div
            key={part.label}
            className={`flex flex-col items-center justify-center rounded-md transition-[width] duration-300 ${TONES[part.tone]}`}
            style={{ width: `${part.pct}%` }}
          >
            <span className="text-[11px] font-bold leading-tight">{part.pct}%</span>
            <span className="truncate px-1 text-[9.5px] font-semibold opacity-90">{part.label}</span>
          </div>
        ))}
      </div>

      <p className="mt-3 text-[0.88rem] leading-relaxed text-muted">{p.when}</p>

      <div className="mt-3 grid gap-2 sm:grid-cols-3">
        <Role tone="train" title="Training set" body="The model fits its parameters here. This is the only data the learning algorithm gets to see." />
        <Role tone="val" title="Validation set" body="You choose between models and hyperparameters here. Looked at many times, so its score is optimistically biased." />
        <Role tone="test" title="Test set" body="Untouched until the very end. Its only job is to give one honest estimate of production performance." />
      </div>
    </VizFrame>
  );
}

function Role({ tone, title, body }: { tone: "train" | "val" | "test"; title: string; body: string }) {
  const dot = { train: "bg-indigo-500", val: "bg-amber-500", test: "bg-rose-500" }[tone];
  return (
    <div className="rounded-lg border border-line bg-surface-2 p-2.5">
      <div className="mb-1 flex items-center gap-1.5">
        <span className={`size-2 rounded-full ${dot}`} aria-hidden />
        <span className="text-[11.5px] font-bold text-ink">{title}</span>
      </div>
      <p className="text-[0.8rem] leading-relaxed text-faint">{body}</p>
    </div>
  );
}

/* ------------------------------------------------------------------ */
/* Sigmoid                                                              */
/* ------------------------------------------------------------------ */

const SIG: Frame = { w: 420, h: 200, pad: { t: 14, r: 14, b: 30, l: 40 } };

export function SigmoidCurve() {
  const s = makeScales(SIG, [-6, 6], [0, 1]);
  const pts = range(160, -6, 6).map((z) => [z, 1 / (1 + Math.exp(-z))] as [number, number]);
  return (
    <div>
      <svg viewBox={`0 0 ${SIG.w} ${SIG.h}`} className="w-full text-line" role="img"
        aria-label="The sigmoid function maps any real number to a value between 0 and 1">
        <rect x={s.x0} y={s.y1} width={s.x1 - s.x0} height={s.y0 - s.y1} fill="none" stroke="currentColor" strokeWidth="1" />
        <line x1={s.x0} x2={s.x1} y1={s.sy(0.5)} y2={s.sy(0.5)} stroke="#f43f5e" strokeWidth="1.2" strokeDasharray="4 4" />
        <text x={s.x1 - 4} y={s.sy(0.5) - 5} textAnchor="end" className="text-[9px] font-bold" fill="#f43f5e">
          default threshold 0.5
        </text>
        <line x1={s.sx(0)} x2={s.sx(0)} y1={s.y0} y2={s.y1} stroke="currentColor" strokeWidth="0.8" opacity="0.5" />
        <path d={linePath(pts, s.sx, s.sy)} fill="none" stroke="#818cf8" strokeWidth="2.6" />
        {[0, 0.5, 1].map((g) => (
          <text key={g} x={s.x0 - 6} y={s.sy(g) + 3} textAnchor="end" className="fill-faint text-[9px]" opacity="0.7">
            {g}
          </text>
        ))}
        {[-6, -3, 0, 3, 6].map((g) => (
          <text key={g} x={s.sx(g)} y={s.y0 + 13} textAnchor="middle" className="fill-faint text-[9px]" opacity="0.7">
            {g}
          </text>
        ))}
        <text x={(s.x0 + s.x1) / 2} y={SIG.h - 4} textAnchor="middle" className="fill-faint text-[9.5px] font-semibold" opacity="0.8">
          z = w·x + b (the log-odds)
        </text>
        <text x={11} y={SIG.h / 2} textAnchor="middle" transform={`rotate(-90 11 ${SIG.h / 2})`}
          className="fill-faint text-[9.5px] font-semibold" opacity="0.8">
          p(y=1)
        </text>
      </svg>
    </div>
  );
}

/* ------------------------------------------------------------------ */
/* Ensembles                                                            */
/* ------------------------------------------------------------------ */

export function EnsembleDiagram() {
  return (
    <div className="grid gap-3 lg:grid-cols-3">
      <EnsemblePanel
        title="Bagging"
        color="#10b981"
        tagline="parallel · independent"
        attacks="Reduces variance"
        body="Each model sees a different bootstrap sample; predictions are averaged (or voted). Errors that come from the quirks of one sample cancel out."
      >
        <svg viewBox="0 0 240 110" className="w-full text-faint" role="img" aria-label="Three independent models trained in parallel and averaged">
          <Box x={6} y={44} w={44} h={22} label="Data" />
          {[10, 44, 78].map((y, i) => (
            <g key={i}>
              <Arrow x1={52} y1={55} x2={78} y2={y + 11} />
              <Box x={80} y={y} w={62} h={22} label={`Tree ${i + 1}`} tone="#10b981" />
              <Arrow x1={144} y1={y + 11} x2={172} y2={55} />
            </g>
          ))}
          <Box x={174} y={44} w={60} h={22} label="Average" tone="#10b981" />
        </svg>
      </EnsemblePanel>

      <EnsemblePanel
        title="Boosting"
        color="#f59e0b"
        tagline="sequential · corrective"
        attacks="Reduces bias (and often variance too)"
        body="Each model is fit to what the previous ones got wrong — the residuals, or reweighted examples. Predictions are summed, scaled by a learning rate."
      >
        <svg viewBox="0 0 240 110" className="w-full text-faint" role="img" aria-label="Models trained one after another, each fitting the previous model's residuals">
          <Box x={4} y={44} w={40} h={22} label="Data" />
          <Arrow x1={46} y1={55} x2={62} y2={55} />
          <Box x={64} y={44} w={44} h={22} label="Tree 1" tone="#f59e0b" />
          <Arrow x1={110} y1={55} x2={126} y2={55} />
          <Box x={128} y={44} w={44} h={22} label="Tree 2" tone="#f59e0b" />
          <Arrow x1={174} y1={55} x2={190} y2={55} />
          <Box x={192} y={44} w={44} h={22} label="Tree 3" tone="#f59e0b" />
          <text x={120} y={26} textAnchor="middle" className="fill-faint text-[8.5px] font-semibold" opacity="0.75">
            each one fits the residual errors of the sum so far
          </text>
          <path d="M150,38 C150,30 96,30 96,38" stroke="currentColor" strokeWidth="1" fill="none" strokeDasharray="3 3" opacity="0.6" />
          <text x={120} y={96} textAnchor="middle" className="fill-faint text-[8.5px] font-semibold" opacity="0.75">
            prediction = η · (tree₁ + tree₂ + tree₃ + …)
          </text>
        </svg>
      </EnsemblePanel>

      <EnsemblePanel
        title="Stacking"
        color="#818cf8"
        tagline="layered · learned blend"
        attacks="Exploits complementary errors"
        body="Different model families predict, and a small meta-model learns how to weight them. Base predictions must come from out-of-fold data or the meta-model learns from leakage."
      >
        <svg viewBox="0 0 240 110" className="w-full text-faint" role="img" aria-label="Several different models feeding a meta model">
          <Box x={4} y={44} w={36} h={22} label="Data" />
          {[8, 44, 80].map((y, i) => (
            <g key={i}>
              <Arrow x1={42} y1={55} x2={62} y2={y + 11} />
              <Box x={64} y={y} w={60} h={22} label={["GBM", "Linear", "KNN"][i]} tone="#818cf8" />
              <Arrow x1={126} y1={y + 11} x2={150} y2={55} />
            </g>
          ))}
          <Box x={152} y={40} w={82} h={30} label="Meta-model" tone="#818cf8" />
        </svg>
      </EnsemblePanel>
    </div>
  );
}

function EnsemblePanel({
  title,
  color,
  tagline,
  attacks,
  body,
  children,
}: {
  title: string;
  color: string;
  tagline: string;
  attacks: string;
  body: string;
  children: React.ReactNode;
}) {
  return (
    <div className="card overflow-hidden">
      <div className="flex items-center gap-2 border-b border-line bg-surface-2 px-3 py-2">
        <span className="size-2.5 rounded-full" style={{ background: color }} aria-hidden />
        <span className="text-[0.92rem] font-bold text-ink">{title}</span>
        <span className="ml-auto text-[10px] font-semibold uppercase tracking-[0.06em] text-faint">
          {tagline}
        </span>
      </div>
      <div className="p-3">
        {children}
        <div className="mt-2 rounded-md px-2 py-1 text-[11px] font-bold" style={{ background: `${color}1a`, color }}>
          {attacks}
        </div>
        <p className="mt-2 text-[0.84rem] leading-relaxed text-muted">{body}</p>
      </div>
    </div>
  );
}

function Box({
  x,
  y,
  w,
  h,
  label,
  tone,
}: {
  x: number;
  y: number;
  w: number;
  h: number;
  label: string;
  tone?: string;
}) {
  return (
    <g>
      <rect x={x} y={y} width={w} height={h} rx="5"
        fill={tone ? `${tone}22` : "var(--surface-2)"}
        stroke={tone ?? "currentColor"} strokeWidth="1.2" />
      <text x={x + w / 2} y={y + h / 2 + 3.2} textAnchor="middle"
        className="fill-ink text-[9px] font-bold">
        {label}
      </text>
    </g>
  );
}

function Arrow({ x1, y1, x2, y2 }: { x1: number; y1: number; x2: number; y2: number }) {
  return (
    <g>
      <line x1={x1} y1={y1} x2={x2 - 4} y2={y2} stroke="currentColor" strokeWidth="1.1" opacity="0.65" />
      <path d={`M${x2 - 5},${y2 - 3} L${x2},${y2} L${x2 - 5},${y2 + 3}`} fill="none"
        stroke="currentColor" strokeWidth="1.1" opacity="0.65" />
    </g>
  );
}
