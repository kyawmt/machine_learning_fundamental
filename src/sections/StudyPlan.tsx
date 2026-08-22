import {
  HOURS,
  HOUR_TITLES,
  SECTIONS,
  TOTAL_MINUTES,
  minutesForHour,
  sectionsForHour,
} from "../data/sections";
import { formatMinutes } from "../hooks/useProgress";
import { Callout } from "../components/Callout";
import { PriorityBadge } from "../components/ui";

/* ------------------------------------------------------------------ */
/* Hero                                                                */
/* ------------------------------------------------------------------ */

export function Hero({
  percent,
  minutesLeft,
  completedCount,
  totalCount,
  nextSectionId,
  nextSectionTitle,
}: {
  percent: number;
  minutesLeft: number;
  completedCount: number;
  totalCount: number;
  nextSectionId: string;
  nextSectionTitle: string;
}) {
  const r = 38;
  const circumference = 2 * Math.PI * r;

  return (
    <header className="relative mb-10 overflow-hidden rounded-2xl border border-line bg-surface p-6 shadow-[var(--shadow-card)] sm:p-8">
      <div className="grid-bg pointer-events-none absolute inset-0 opacity-40" aria-hidden />
      <div
        className="pointer-events-none absolute -right-24 -top-24 size-72 rounded-full opacity-25 blur-3xl"
        style={{ background: "radial-gradient(circle, var(--accent), transparent 70%)" }}
        aria-hidden
      />

      <div className="relative">
        <div className="mb-3 flex flex-wrap items-center gap-2">
          <span className="rounded-full border border-accent/30 bg-accent-soft px-2.5 py-0.5 text-[10.5px] font-bold uppercase tracking-[0.09em] text-accent">
            AI / ML Engineer Interview Prep
          </span>
          <span className="text-[11px] font-semibold text-faint">
            {SECTIONS.length} sections · {formatMinutes(TOTAL_MINUTES)} of focused review
          </span>
        </div>

        <h1 className="max-w-3xl text-[2rem] font-extrabold leading-[1.1] tracking-tight text-ink sm:text-[2.6rem]">
          Machine Learning Fundamentals
        </h1>
        <p className="mt-1.5 max-w-2xl text-[1.05rem] font-semibold text-accent sm:text-[1.2rem]">
          4-Hour AI Engineer Interview Review
        </p>
        <p className="mt-3 max-w-2xl text-[0.95rem] leading-relaxed text-muted">
          Not a textbook. Every section is built around what actually gets asked: the intuition, the formula worth
          memorising, the sentence you say out loud, and the mistake that gives you away. Work top to bottom and
          you'll finish in four hours.
        </p>

        <div className="mt-6 flex flex-col gap-6 sm:flex-row sm:items-center">
          {/* Progress ring */}
          <div className="flex items-center gap-4">
            <div className="relative shrink-0">
              <svg width="92" height="92" viewBox="0 0 92 92" role="img" aria-label={`${percent} percent complete`}>
                <circle cx="46" cy="46" r={r} fill="none" stroke="var(--surface-3)" strokeWidth="8" />
                <circle
                  cx="46" cy="46" r={r} fill="none" stroke="url(#heroGrad)" strokeWidth="8" strokeLinecap="round"
                  strokeDasharray={circumference}
                  strokeDashoffset={circumference * (1 - percent / 100)}
                  transform="rotate(-90 46 46)"
                  style={{ transition: "stroke-dashoffset 600ms ease" }}
                />
                <defs>
                  <linearGradient id="heroGrad" x1="0" y1="0" x2="1" y2="1">
                    <stop offset="0%" stopColor="#6366f1" />
                    <stop offset="100%" stopColor="#a855f7" />
                  </linearGradient>
                </defs>
              </svg>
              <div className="absolute inset-0 grid place-items-center">
                <span className="font-mono text-[1.15rem] font-extrabold text-ink">{percent}%</span>
              </div>
            </div>
            <div>
              <div className="text-[11px] font-bold uppercase tracking-[0.08em] text-faint">
                Overall progress
              </div>
              <div className="mt-0.5 text-[1.05rem] font-bold text-ink">
                {completedCount} of {totalCount} sections
              </div>
              <div className="mt-0.5 text-[0.88rem] text-muted">
                {minutesLeft > 0 ? (
                  <>
                    <strong className="font-bold text-ink">{formatMinutes(minutesLeft)}</strong> of review
                    remaining
                  </>
                ) : (
                  "All sections complete — run the quiz and the readiness check."
                )}
              </div>
            </div>
          </div>

          <div className="flex flex-wrap gap-2 sm:ml-auto">
            <a
              href={`#${nextSectionId}`}
              className="inline-flex items-center gap-2 rounded-xl bg-gradient-to-r from-indigo-600 to-violet-600 px-5 py-2.5 text-[13.5px] font-bold text-white shadow-lg shadow-indigo-500/20 transition-transform hover:-translate-y-px"
            >
              {completedCount === 0 ? "Start Review" : "Continue"}
              <span className="opacity-80">→</span>
            </a>
            <a
              href="#cheat-sheet"
              className="inline-flex items-center gap-2 rounded-xl border border-line bg-surface-2 px-4 py-2.5 text-[13.5px] font-bold text-muted transition-colors hover:text-ink"
            >
              Cheat sheet
            </a>
          </div>
        </div>

        {completedCount > 0 && completedCount < totalCount && (
          <p className="mt-3 text-[12px] text-faint">
            Next up: <span className="font-semibold text-muted">{nextSectionTitle}</span>
          </p>
        )}
      </div>
    </header>
  );
}

/* ------------------------------------------------------------------ */
/* Study plan                                                          */
/* ------------------------------------------------------------------ */

const HOUR_ACCENT: Record<number, string> = {
  1: "from-indigo-500 to-indigo-400",
  2: "from-violet-500 to-violet-400",
  3: "from-fuchsia-500 to-fuchsia-400",
  4: "from-rose-500 to-rose-400",
};

export function StudyPlan({
  isDone,
  toggleDone,
}: {
  isDone: (id: string) => boolean;
  toggleDone: (id: string) => void;
}) {
  return (
    <div className="space-y-5">
      <div className="grid gap-3 lg:grid-cols-2">
        {HOURS.map((h) => {
          const items = sectionsForHour(h);
          const doneCount = items.filter((s) => isDone(s.id)).length;
          const pct = Math.round((doneCount / items.length) * 100);
          return (
            <div key={h} className="card overflow-hidden">
              <div className={`h-1 w-full bg-gradient-to-r ${HOUR_ACCENT[h]}`} />
              <div className="flex items-start justify-between gap-2 border-b border-line bg-surface-2 px-4 py-2.5">
                <div>
                  <h3 className="text-[0.98rem] font-bold text-ink">{HOUR_TITLES[h].title}</h3>
                  <p className="text-[11.5px] text-faint">{HOUR_TITLES[h].subtitle}</p>
                </div>
                <div className="shrink-0 text-right">
                  <div className="font-mono text-[13px] font-extrabold text-ink">
                    {minutesForHour(h)}m
                  </div>
                  <div className="text-[10px] font-semibold text-faint">
                    {doneCount}/{items.length} done
                  </div>
                </div>
              </div>

              <div className="h-0.5 w-full bg-surface-3">
                <div
                  className={`h-full bg-gradient-to-r ${HOUR_ACCENT[h]} transition-[width] duration-500`}
                  style={{ width: `${pct}%` }}
                />
              </div>

              <ul className="divide-y divide-[var(--border)]">
                {items.map((s) => {
                  const done = isDone(s.id);
                  return (
                    <li key={s.id} className="flex items-center gap-2.5 px-3 py-2">
                      <button
                        type="button"
                        onClick={() => toggleDone(s.id)}
                        aria-label={`Mark ${s.nav} ${done ? "incomplete" : "complete"}`}
                        aria-pressed={done}
                        className={`grid size-4 shrink-0 place-items-center rounded-[5px] border transition-colors ${
                          done
                            ? "border-emerald-500 bg-emerald-500 text-white dark:border-emerald-400 dark:bg-emerald-400 dark:text-slate-900"
                            : "border-line-strong text-transparent hover:border-accent"
                        }`}
                      >
                        <svg width="9" height="9" viewBox="0 0 24 24" fill="none" aria-hidden>
                          <path d="M4 12.5l5.2 5L20 6.5" stroke="currentColor" strokeWidth="3.4" strokeLinecap="round" strokeLinejoin="round" />
                        </svg>
                      </button>
                      <PriorityBadge priority={s.priority} compact />
                      <a
                        href={`#${s.id}`}
                        className={`min-w-0 flex-1 truncate text-[13px] font-medium hover:text-accent ${
                          done ? "text-faint line-through decoration-1" : "text-muted"
                        }`}
                      >
                        {s.nav}
                      </a>
                      <span className="shrink-0 font-mono text-[10.5px] text-faint">{s.minutes}m</span>
                    </li>
                  );
                })}
              </ul>
            </div>
          );
        })}
      </div>

      <div className="grid gap-3 lg:grid-cols-2">
        <Callout variant="tip" title="How to actually use this">
          <ul className="bullet-list">
            <li>
              Read a section, then <strong>close your eyes and say the interview answer out loud</strong>. Reading
              feels like learning; speaking is the thing you're being tested on.
            </li>
            <li>Tick the box only after you can explain it without looking. The progress bar should mean something.</li>
            <li>
              Play with the interactive labs — the threshold and bias–variance ones especially. Two minutes of
              dragging a slider beats ten minutes of re-reading.
            </li>
            <li>Leave the Rapid-Fire Quiz for the last 10 minutes. It's a diagnostic, not a study method.</li>
          </ul>
        </Callout>

        <Callout variant="warning" title="Short on time? Triage like this">
          <ul className="bullet-list">
            <li>
              <strong>60 minutes:</strong> Overfitting/Underfitting → Bias–Variance → Evaluation Metrics →
              Data Leakage → Cheat Sheet.
            </li>
            <li>
              <strong>2 hours:</strong> all of Hour 1 + Evaluation Metrics + Regularization + the Random Forest vs
              Gradient Boosting card + Cheat Sheet.
            </li>
            <li>
              <strong>The morning of:</strong> Cheat Sheet, then the Rapid-Fire Quiz, then re-read any red-priority
              section you missed.
            </li>
            <li>
              Skip green-priority material entirely if you're squeezed. It is the 5% that almost never decides an
              interview.
            </li>
          </ul>
        </Callout>
      </div>
    </div>
  );
}
