import { useState } from "react";
import { RichText } from "./RichText";

export interface Scenario {
  id: string;
  title: string;
  /** The situation, as an interviewer would present it. */
  setup: string;
  /** Optional numeric readout shown as a small stat strip. */
  stats?: { label: string; value: string; tone?: "good" | "bad" | "warn" | "neutral" }[];
  question: string;
  answer: string;
  /** What you'd do next / what to investigate. */
  nextSteps?: string;
}

const STAT_TONE = {
  good: "border-emerald-300 text-emerald-700 dark:border-emerald-400/35 dark:text-emerald-300",
  bad: "border-rose-300 text-rose-700 dark:border-rose-400/35 dark:text-rose-300",
  warn: "border-amber-300 text-amber-700 dark:border-amber-400/35 dark:text-amber-300",
  neutral: "border-line text-muted",
} as const;

export function ScenarioCard({ s, n }: { s: Scenario; n: number }) {
  const [open, setOpen] = useState(false);

  return (
    <div className="card overflow-hidden">
      <div className="border-b border-line bg-surface-2 px-4 py-2.5">
        <div className="flex items-center gap-2">
          <span className="rounded-md bg-accent px-1.5 py-0.5 font-mono text-[10.5px] font-bold text-white">
            S{n}
          </span>
          <h4 className="text-[0.95rem] font-bold text-ink">{s.title}</h4>
        </div>
      </div>

      <div className="space-y-3 p-4">
        <RichText text={s.setup} className="text-[0.9rem] text-muted" />

        {s.stats && (
          <div className="flex flex-wrap gap-2">
            {s.stats.map((st) => (
              <div
                key={st.label}
                className={`rounded-lg border bg-surface px-3 py-1.5 ${STAT_TONE[st.tone ?? "neutral"]}`}
              >
                <div className="text-[10px] font-semibold uppercase tracking-[0.07em] opacity-70">
                  {st.label}
                </div>
                <div className="font-mono text-[0.95rem] font-bold">{st.value}</div>
              </div>
            ))}
          </div>
        )}

        <p className="text-[0.92rem] font-semibold text-ink">{s.question}</p>

        {!open ? (
          <button
            type="button"
            onClick={() => setOpen(true)}
            className="no-print rounded-lg border border-line bg-surface-2 px-3 py-1.5 text-[12.5px] font-semibold text-muted transition-colors hover:border-line-strong hover:text-ink"
          >
            Think first, then reveal →
          </button>
        ) : (
          <div className="space-y-3">
            <div className="rounded-lg border border-violet-200 bg-violet-50/70 px-3 py-2.5 dark:border-violet-400/25 dark:bg-violet-400/8">
              <div className="mb-1 text-[10.5px] font-bold uppercase tracking-[0.09em] text-violet-700 dark:text-violet-300">
                How to answer
              </div>
              <RichText text={s.answer} className="text-[0.89rem] text-muted" />
            </div>
            {s.nextSteps && (
              <div className="rounded-lg border border-teal-200 bg-teal-50/70 px-3 py-2.5 dark:border-teal-400/25 dark:bg-teal-400/8">
                <div className="mb-1 text-[10.5px] font-bold uppercase tracking-[0.09em] text-teal-700 dark:text-teal-300">
                  What you'd do next
                </div>
                <RichText text={s.nextSteps} className="text-[0.89rem] text-muted" />
              </div>
            )}
            <button
              type="button"
              onClick={() => setOpen(false)}
              className="no-print text-[12px] font-semibold text-faint hover:text-ink"
            >
              Hide answer
            </button>
          </div>
        )}
      </div>
    </div>
  );
}
