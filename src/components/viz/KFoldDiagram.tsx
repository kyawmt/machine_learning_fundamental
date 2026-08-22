import { useState } from "react";
import { VizFrame } from "./VizFrame";

export function KFoldDiagram() {
  const [k, setK] = useState(5);
  const [hover, setHover] = useState<number | null>(null);

  const scores = [0.842, 0.871, 0.826, 0.859, 0.848, 0.863, 0.834, 0.855, 0.869, 0.841];
  const used = scores.slice(0, k);
  const mean = used.reduce((a, b) => a + b, 0) / k;
  const sd = Math.sqrt(used.reduce((s, v) => s + (v - mean) ** 2, 0) / k);

  return (
    <VizFrame
      title={`${k}-Fold Cross-Validation`}
      hint="each row is one complete train + evaluate run"
      footer={
        <>
          Reported CV score = <strong className="text-ink">{mean.toFixed(3)} ± {sd.toFixed(3)}</strong>. The spread
          matters as much as the mean: a wide spread says your estimate is unstable, usually because the dataset is
          small or the folds are not comparable.
        </>
      }
    >
      <div className="mb-3 flex items-center gap-1.5">
        <span className="text-[11px] font-bold uppercase tracking-[0.07em] text-faint">K</span>
        {[3, 5, 10].map((v) => (
          <button
            key={v}
            type="button"
            onClick={() => setK(v)}
            className={`rounded-md border px-2.5 py-1 text-[12px] font-bold transition-colors ${
              k === v ? "border-accent bg-accent text-white" : "border-line bg-surface-2 text-muted hover:text-ink"
            }`}
          >
            {v}
          </button>
        ))}
      </div>

      <div className="space-y-1.5">
        {Array.from({ length: k }, (_, row) => (
          <div
            key={row}
            className="flex items-center gap-2"
            onMouseEnter={() => setHover(row)}
            onMouseLeave={() => setHover(null)}
          >
            <span className="w-12 shrink-0 text-right font-mono text-[10.5px] font-semibold text-faint">
              run {row + 1}
            </span>
            <div className="flex flex-1 gap-1">
              {Array.from({ length: k }, (_, col) => {
                const isVal = col === row;
                return (
                  <div
                    key={col}
                    className={`h-6 flex-1 rounded-[5px] border text-center text-[9.5px] font-bold leading-6 transition-colors ${
                      isVal
                        ? "border-rose-400 bg-rose-100 text-rose-700 dark:border-rose-400/50 dark:bg-rose-500/20 dark:text-rose-300"
                        : "border-emerald-300 bg-emerald-50 text-emerald-700 dark:border-emerald-400/30 dark:bg-emerald-500/12 dark:text-emerald-300"
                    } ${hover === row ? "ring-1 ring-accent" : ""}`}
                  >
                    {k <= 5 ? (isVal ? "val" : "train") : ""}
                  </div>
                );
              })}
            </div>
            <span className="w-11 shrink-0 text-right font-mono text-[10.5px] font-semibold text-ink tabular-nums">
              {used[row].toFixed(3)}
            </span>
          </div>
        ))}
      </div>

      <div className="mt-3 flex items-center gap-2 border-t border-line pt-3">
        <span className="w-12 shrink-0" />
        <div className="flex-1 rounded-lg border border-dashed border-line-strong bg-surface-2 px-3 py-1.5 text-[11.5px] font-semibold text-muted">
          Every point is used for validation exactly once, and for training K−1 times.
        </div>
        <span className="w-11 shrink-0 text-right font-mono text-[11px] font-extrabold text-accent tabular-nums">
          {mean.toFixed(3)}
        </span>
      </div>

      <div className="mt-3 rounded-lg border border-amber-300 bg-amber-50/80 px-3 py-2 text-[0.85rem] text-amber-800 dark:border-amber-400/30 dark:bg-amber-400/10 dark:text-amber-200">
        <strong>The test set is not on this diagram.</strong> All K folds come out of the training portion. The
        held-out test set sits outside the whole loop and is touched once, at the end.
      </div>
    </VizFrame>
  );
}
