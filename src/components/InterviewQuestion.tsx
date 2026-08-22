import { useState } from "react";
import { RichText } from "./RichText";
import { Chip } from "./ui";

export interface QA {
  id: string;
  q: string;
  /** One-line answer — the thing to say first. */
  short: string;
  /** The fuller answer you'd actually give in an interview. */
  strong: string;
  /** Optional extra depth for follow-up questions. */
  deeper?: string;
  /** Section id this question maps back to. */
  ref?: string;
}

export function InterviewQuestion({
  qa,
  n,
  defaultOpen = false,
}: {
  qa: QA;
  n?: number;
  defaultOpen?: boolean;
}) {
  const [open, setOpen] = useState(defaultOpen);
  const [deep, setDeep] = useState(false);

  return (
    <div
      className={`card overflow-hidden transition-colors ${
        open ? "ring-1 ring-indigo-200 dark:ring-indigo-400/25" : ""
      }`}
    >
      <button
        type="button"
        onClick={() => setOpen((o) => !o)}
        aria-expanded={open}
        className="flex w-full items-start gap-3 px-4 py-3 text-left hover:bg-surface-2"
      >
        {n !== undefined && (
          <span className="mt-0.5 font-mono text-[11px] font-bold text-faint">
            {String(n).padStart(2, "0")}
          </span>
        )}
        <span className="flex-1 text-[0.94rem] font-semibold leading-snug text-ink">{qa.q}</span>
        <span
          className={`mt-0.5 shrink-0 text-faint transition-transform ${open ? "rotate-180" : ""}`}
          aria-hidden
        >
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none">
            <path
              d="M5 9l7 7 7-7"
              stroke="currentColor"
              strokeWidth="2.4"
              strokeLinecap="round"
              strokeLinejoin="round"
            />
          </svg>
        </span>
      </button>

      {open && (
        <div className="space-y-3 border-t border-line px-4 py-3.5">
          <div className="rounded-lg border border-emerald-200 bg-emerald-50/70 px-3 py-2 dark:border-emerald-400/25 dark:bg-emerald-400/8">
            <div className="mb-1 text-[10.5px] font-bold uppercase tracking-[0.09em] text-emerald-700 dark:text-emerald-300">
              Short answer
            </div>
            <RichText text={qa.short} className="text-[0.89rem] text-muted" />
          </div>

          <div className="rounded-lg border border-violet-200 bg-violet-50/70 px-3 py-2 dark:border-violet-400/25 dark:bg-violet-400/8">
            <div className="mb-1 text-[10.5px] font-bold uppercase tracking-[0.09em] text-violet-700 dark:text-violet-300">
              Say this in the interview
            </div>
            <RichText text={qa.strong} className="text-[0.89rem] text-muted" />
          </div>

          {qa.deeper && (
            <div>
              <button
                type="button"
                onClick={() => setDeep((d) => !d)}
                aria-expanded={deep}
                className="text-[12px] font-semibold text-accent hover:underline"
              >
                {deep ? "− Hide deeper explanation" : "+ Deeper explanation"}
              </button>
              {deep && (
                <div className="mt-2 rounded-lg border border-line bg-surface-2 px-3 py-2">
                  <RichText text={qa.deeper} className="text-[0.87rem] text-muted" />
                </div>
              )}
            </div>
          )}

          {qa.ref && (
            <a
              href={`#${qa.ref}`}
              className="inline-block no-print"
              onClick={(e) => e.stopPropagation()}
            >
              <Chip tone="accent">↑ Review the section</Chip>
            </a>
          )}
        </div>
      )}
    </div>
  );
}
