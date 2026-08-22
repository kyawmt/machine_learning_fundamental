import { useMemo, useState } from "react";
import { VizFrame, Slider } from "./VizFrame";
import { gaussian, linePath, makeScales, mulberry32, type Frame } from "./plot";

/**
 * One fixed, seeded synthetic scoring problem (imbalanced, like fraud):
 * 970 negatives and 30 positives with overlapping score distributions.
 * Every number in this demo is computed from that dataset — nothing is faked,
 * so the metric behaviour (including the noisy tail of the precision curve) is real.
 */
function buildData() {
  const rnd = mulberry32(20260822);
  const scores: { s: number; y: 0 | 1 }[] = [];
  const squash = (z: number) => 1 / (1 + Math.exp(-z));
  for (let i = 0; i < 970; i++) scores.push({ s: squash(gaussian(rnd, -3.0, 1.6)), y: 0 });
  for (let i = 0; i < 30; i++) scores.push({ s: squash(gaussian(rnd, -0.1, 1.35)), y: 1 });
  return scores;
}

interface Point {
  t: number;
  tp: number;
  fp: number;
  tn: number;
  fn: number;
  tpr: number;
  fpr: number;
  precision: number;
}

function confusion(data: { s: number; y: 0 | 1 }[], t: number) {
  let tp = 0,
    fp = 0,
    tn = 0,
    fn = 0;
  for (const d of data) {
    const pred = d.s >= t;
    if (d.y === 1) pred ? tp++ : fn++;
    else pred ? fp++ : tn++;
  }
  return { tp, fp, tn, fn };
}

const F: Frame = { w: 300, h: 240, pad: { t: 14, r: 14, b: 32, l: 40 } };

export function ThresholdLab() {
  const data = useMemo(buildData, []);
  const [t, setT] = useState(0.5);

  const sweep = useMemo<Point[]>(() => {
    const pts: Point[] = [];
    for (let i = 0; i <= 200; i++) {
      const th = i / 200;
      const { tp, fp, tn, fn } = confusion(data, th);
      pts.push({
        t: th,
        tp,
        fp,
        tn,
        fn,
        tpr: tp + fn === 0 ? 0 : tp / (tp + fn),
        fpr: fp + tn === 0 ? 0 : fp / (fp + tn),
        precision: tp + fp === 0 ? 1 : tp / (tp + fp),
      });
    }
    return pts;
  }, [data]);

  const rocAuc = useMemo(() => {
    // Trapezoidal integration over the ROC sweep (FPR descending as t rises).
    const ordered = [...sweep].sort((a, b) => a.fpr - b.fpr);
    let area = 0;
    for (let i = 1; i < ordered.length; i++) {
      area += ((ordered[i].fpr - ordered[i - 1].fpr) * (ordered[i].tpr + ordered[i - 1].tpr)) / 2;
    }
    return area;
  }, [sweep]);

  const prAuc = useMemo(() => {
    const ordered = [...sweep].sort((a, b) => a.tpr - b.tpr);
    let area = 0;
    for (let i = 1; i < ordered.length; i++) {
      area +=
        ((ordered[i].tpr - ordered[i - 1].tpr) * (ordered[i].precision + ordered[i - 1].precision)) / 2;
    }
    return area;
  }, [sweep]);

  const { tp, fp, tn, fn } = confusion(data, t);
  const total = data.length;
  const precision = tp + fp === 0 ? 0 : tp / (tp + fp);
  const recall = tp + fn === 0 ? 0 : tp / (tp + fn);
  const specificity = tn + fp === 0 ? 0 : tn / (tn + fp);
  const accuracy = (tp + tn) / total;
  const f1 = precision + recall === 0 ? 0 : (2 * precision * recall) / (precision + recall);
  const baseRate = data.filter((d) => d.y === 1).length / total;

  const roc = makeScales(F, [0, 1], [0, 1]);
  const pr = makeScales(F, [0, 1], [0, 1]);
  const cur = sweep.reduce((a, b) => (Math.abs(b.t - t) < Math.abs(a.t - t) ? b : a));

  return (
    <VizFrame
      title="Threshold / Confusion-Matrix Lab"
      hint="1,000 cases · 3% positive · one fixed model"
      footer={
        <>
          The model is fixed — only the decision threshold moves. Ranking quality
          (<strong className="text-ink">ROC-AUC {rocAuc.toFixed(3)}</strong>,{" "}
          <strong className="text-ink">PR-AUC {prAuc.toFixed(3)}</strong>) is threshold-independent; precision,
          recall, F1 and accuracy all are not. Note the gap between those two AUCs — the ROC number sounds like a
          strong model, and the PR number is what the alert queue will actually feel like. That gap is the whole
          argument for PR-AUC on imbalanced data.
        </>
      }
    >
      <div className="mb-4">
        <Slider
          label="Decision threshold"
          value={t}
          min={0.02}
          max={0.98}
          step={0.01}
          onChange={setT}
          display={t.toFixed(2)}
          leftLabel="0.02 — flag almost everything"
          rightLabel="0.98 — flag almost nothing"
        />
      </div>

      {/* Confusion matrix */}
      <div className="grid gap-4 sm:grid-cols-[minmax(0,1fr)_minmax(0,1fr)]">
        <div>
          <div className="mb-1.5 text-[10.5px] font-bold uppercase tracking-[0.08em] text-faint">
            Confusion matrix @ {t.toFixed(2)}
          </div>
          <div className="grid grid-cols-[auto_1fr_1fr] gap-1 text-center text-[11px]">
            <div />
            <div className="pb-0.5 font-semibold text-faint">Pred +</div>
            <div className="pb-0.5 font-semibold text-faint">Pred −</div>

            <div className="flex items-center pr-1 text-right font-semibold text-faint">
              Actual&nbsp;+
            </div>
            <Cell v={tp} label="TP" tone="good" />
            <Cell v={fn} label="FN" tone="bad" />

            <div className="flex items-center pr-1 text-right font-semibold text-faint">
              Actual&nbsp;−
            </div>
            <Cell v={fp} label="FP" tone="bad" />
            <Cell v={tn} label="TN" tone="good" />
          </div>

        </div>

        <div>
          <div className="mb-1.5 text-[10.5px] font-bold uppercase tracking-[0.08em] text-faint">
            Metrics at this threshold
          </div>
          <div className="grid grid-cols-2 gap-1.5">
            <Metric label="Precision" v={precision} formula="TP / (TP+FP)" />
            <Metric label="Recall (TPR)" v={recall} formula="TP / (TP+FN)" />
            <Metric label="F1" v={f1} formula="harmonic mean" />
            <Metric label="Specificity" v={specificity} formula="TN / (TN+FP)" />
            <Metric label="Accuracy" v={accuracy} formula="(TP+TN) / N" wide />
          </div>
          <p className="mt-2 text-[11.5px] leading-relaxed text-faint">
            A model that predicts "negative" for everything scores{" "}
            <strong className="text-ink">{((1 - baseRate) * 100).toFixed(1)}% accuracy</strong> here with{" "}
            <strong className="text-ink">0% recall</strong>. That is the whole argument against accuracy on
            imbalanced data.
          </p>
        </div>
      </div>

      {/* Curves */}
      <div className="mt-4 grid gap-4 sm:grid-cols-2">
          <MiniPlot
            title={`ROC · AUC ${rocAuc.toFixed(3)}`}
            xLabel="FPR"
            yLabel="TPR"
            path={linePath(
              [...sweep].sort((a, b) => a.fpr - b.fpr).map((p) => [p.fpr, p.tpr]),
              roc.sx,
              roc.sy,
            )}
            diagonal={`M${roc.sx(0)},${roc.sy(0)} L${roc.sx(1)},${roc.sy(1)}`}
            marker={[roc.sx(cur.fpr), roc.sy(cur.tpr)]}
            color="#818cf8"
            scales={roc}
          />
          <MiniPlot
            title={`Precision–Recall · AUC ${prAuc.toFixed(3)}`}
            xLabel="Recall"
            yLabel="Precision"
            path={linePath(
              [...sweep].sort((a, b) => a.tpr - b.tpr).map((p) => [p.tpr, p.precision]),
              pr.sx,
              pr.sy,
            )}
            diagonal={`M${pr.sx(0)},${pr.sy(baseRate)} L${pr.sx(1)},${pr.sy(baseRate)}`}
            marker={[pr.sx(cur.tpr), pr.sy(cur.precision)]}
            color="#10b981"
            scales={pr}
            baselineNote={`no-skill = ${baseRate.toFixed(2)}`}
          />
      </div>
    </VizFrame>
  );
}

function Cell({ v, label, tone }: { v: number; label: string; tone: "good" | "bad" }) {
  const cls =
    tone === "good"
      ? "border-emerald-300 bg-emerald-50 text-emerald-700 dark:border-emerald-400/30 dark:bg-emerald-500/10 dark:text-emerald-300"
      : "border-rose-300 bg-rose-50 text-rose-700 dark:border-rose-400/30 dark:bg-rose-500/10 dark:text-rose-300";
  return (
    <div className={`rounded-lg border px-2 py-2 ${cls}`}>
      <div className="text-[9.5px] font-bold uppercase tracking-[0.08em] opacity-75">{label}</div>
      <div className="font-mono text-lg font-extrabold tabular-nums">{v}</div>
    </div>
  );
}

function Metric({
  label,
  v,
  formula,
  wide,
}: {
  label: string;
  v: number;
  formula: string;
  wide?: boolean;
}) {
  return (
    <div
      className={`rounded-lg border border-line bg-surface-2 px-2.5 py-1.5 ${wide ? "col-span-2" : ""}`}
    >
      <div className="flex items-baseline justify-between gap-2">
        <span className="text-[10px] font-bold uppercase tracking-[0.06em] text-faint">{label}</span>
        <span className="font-mono text-[0.95rem] font-bold text-ink tabular-nums">
          {(v * 100).toFixed(1)}%
        </span>
      </div>
      <div className="font-mono text-[9.5px] text-faint">{formula}</div>
    </div>
  );
}

function MiniPlot({
  title,
  xLabel,
  yLabel,
  path,
  diagonal,
  marker,
  color,
  scales,
  baselineNote,
}: {
  title: string;
  xLabel: string;
  yLabel: string;
  path: string;
  diagonal: string;
  marker: [number, number];
  color: string;
  scales: ReturnType<typeof makeScales>;
  baselineNote?: string;
}) {
  const { x0, x1, y0, y1 } = scales;
  return (
    <div>
      <div className="mb-1 text-[10.5px] font-bold uppercase tracking-[0.08em] text-faint">
        {title}
      </div>
      <svg viewBox={`0 0 ${F.w} ${F.h}`} className="w-full text-line" role="img" aria-label={title}>
        <rect x={x0} y={y1} width={x1 - x0} height={y0 - y1} fill="none" stroke="currentColor" strokeWidth="1" />
        {[0.25, 0.5, 0.75].map((g) => (
          <g key={g}>
            <line x1={x0} x2={x1} y1={scales.sy(g)} y2={scales.sy(g)} stroke="currentColor" strokeWidth="0.8" opacity="0.5" />
            <line y1={y0} y2={y1} x1={scales.sx(g)} x2={scales.sx(g)} stroke="currentColor" strokeWidth="0.8" opacity="0.5" />
          </g>
        ))}
        <path d={diagonal} stroke="currentColor" strokeWidth="1.3" strokeDasharray="4 4" fill="none" opacity="0.9" />
        <path d={path} fill="none" stroke={color} strokeWidth="2.4" strokeLinejoin="round" />
        <circle cx={marker[0]} cy={marker[1]} r="5" fill={color} stroke="var(--surface)" strokeWidth="2" />
        <text x={(x0 + x1) / 2} y={F.h - 8} textAnchor="middle" className="fill-faint text-[9.5px] font-semibold" opacity="0.8">
          {xLabel}
        </text>
        <text x={11} y={(y0 + y1) / 2} textAnchor="middle" transform={`rotate(-90 11 ${(y0 + y1) / 2})`}
          className="fill-faint text-[9.5px] font-semibold" opacity="0.8">
          {yLabel}
        </text>
        {[0, 1].map((g) => (
          <text key={g} x={x0 - 6} y={scales.sy(g) + 3} textAnchor="end" className="fill-faint text-[8.5px]" opacity="0.7">
            {g}
          </text>
        ))}
      </svg>
      {baselineNote && <div className="-mt-1 text-[10px] text-faint">dashed line: {baselineNote}</div>}
    </div>
  );
}
