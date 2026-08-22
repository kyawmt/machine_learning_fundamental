import type { ReactNode } from "react";
import type { SectionMeta } from "../data/sections";
import { PriorityBadge } from "./ui";

interface TopicSectionProps {
  meta: SectionMeta;
  index: number;
  done: boolean;
  onToggleDone: () => void;
  children: ReactNode;
}

export function TopicSection({
  meta,
  index,
  done,
  onToggleDone,
  children,
}: TopicSectionProps) {
  return (
    <section id={meta.id} className="anchor-offset scroll-mt-24 pt-2">
      <header className="mb-5 border-b border-line pb-4">
        <div className="mb-2 flex flex-wrap items-center gap-2">
          <span className="font-mono text-[11px] font-semibold tracking-wider text-faint">
            {String(index).padStart(2, "0")}
          </span>
          <PriorityBadge priority={meta.priority} />
          {meta.hour > 0 && (
            <span className="rounded-md border border-line bg-surface-2 px-2 py-0.5 text-[11px] font-semibold text-muted">
              Hour {meta.hour}
            </span>
          )}
          {meta.minutes > 0 && (
            <span className="inline-flex items-center gap-1 text-[11px] font-semibold text-faint">
              <svg width="11" height="11" viewBox="0 0 24 24" fill="none" aria-hidden>
                <circle cx="12" cy="12" r="9" stroke="currentColor" strokeWidth="2.2" />
                <path d="M12 7v5l3.2 2" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" />
              </svg>
              {meta.minutes} min
            </span>
          )}
        </div>

        <div className="flex flex-wrap items-start justify-between gap-3">
          <div className="min-w-0">
            <h2 className="text-[1.6rem] font-extrabold leading-tight text-ink sm:text-[1.85rem]">
              {meta.title}
            </h2>
            <p className="mt-1 max-w-2xl text-[0.92rem] leading-relaxed text-faint">
              {meta.blurb}
            </p>
          </div>

          <button
            type="button"
            onClick={onToggleDone}
            aria-pressed={done}
            className={`no-print inline-flex shrink-0 items-center gap-2 rounded-lg border px-3 py-1.5 text-[12.5px] font-semibold transition-colors ${
              done
                ? "border-emerald-300 bg-emerald-50 text-emerald-700 dark:border-emerald-400/35 dark:bg-emerald-500/12 dark:text-emerald-300"
                : "border-line bg-surface text-muted hover:border-line-strong hover:text-ink"
            }`}
          >
            <span
              className={`grid size-4 place-items-center rounded-[5px] border ${
                done
                  ? "border-emerald-500 bg-emerald-500 text-white dark:border-emerald-400 dark:bg-emerald-400 dark:text-slate-900"
                  : "border-line-strong"
              }`}
              aria-hidden
            >
              {done && (
                <svg width="10" height="10" viewBox="0 0 24 24" fill="none">
                  <path
                    d="M4 12.5l5.2 5L20 6.5"
                    stroke="currentColor"
                    strokeWidth="3.2"
                    strokeLinecap="round"
                    strokeLinejoin="round"
                  />
                </svg>
              )}
            </span>
            {done ? "Completed" : "Mark complete"}
          </button>
        </div>
      </header>

      <div className="space-y-5">{children}</div>
    </section>
  );
}
