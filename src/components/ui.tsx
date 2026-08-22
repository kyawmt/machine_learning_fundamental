import type { ReactNode } from "react";
import type { Priority } from "../data/sections";
import { PRIORITY_LABEL } from "../data/sections";

/* ------------------------------------------------------------------ */
/* Priority                                                            */
/* ------------------------------------------------------------------ */

const PRIORITY_STYLE: Record<Priority, { dot: string; chip: string }> = {
  must: {
    dot: "bg-red-500",
    chip: "border-red-200 bg-red-50 text-red-700 dark:border-red-400/30 dark:bg-red-500/10 dark:text-red-300",
  },
  important: {
    dot: "bg-orange-500",
    chip: "border-orange-200 bg-orange-50 text-orange-700 dark:border-orange-400/30 dark:bg-orange-500/10 dark:text-orange-300",
  },
  good: {
    dot: "bg-green-500",
    chip: "border-green-200 bg-green-50 text-green-700 dark:border-green-400/30 dark:bg-green-500/10 dark:text-green-300",
  },
};

export function PriorityBadge({
  priority,
  compact = false,
}: {
  priority: Priority;
  compact?: boolean;
}) {
  const s = PRIORITY_STYLE[priority];
  if (compact) {
    return (
      <span
        title={PRIORITY_LABEL[priority]}
        aria-label={PRIORITY_LABEL[priority]}
        className={`inline-block size-2 shrink-0 rounded-full ${s.dot}`}
      />
    );
  }
  return (
    <span
      className={`inline-flex items-center gap-1.5 whitespace-nowrap rounded-full border px-2 py-0.5 text-[10.5px] font-bold uppercase tracking-[0.07em] ${s.chip}`}
    >
      <span className={`size-1.5 rounded-full ${s.dot}`} aria-hidden />
      {PRIORITY_LABEL[priority]}
    </span>
  );
}

/* ------------------------------------------------------------------ */
/* Small pieces                                                        */
/* ------------------------------------------------------------------ */

export function Chip({
  children,
  tone = "neutral",
  className = "",
}: {
  children: ReactNode;
  tone?: "neutral" | "accent" | "green" | "red" | "amber" | "blue";
  className?: string;
}) {
  const tones = {
    neutral: "border-line bg-surface-2 text-muted",
    accent:
      "border-indigo-200 bg-indigo-50 text-indigo-700 dark:border-indigo-400/30 dark:bg-indigo-500/10 dark:text-indigo-300",
    green:
      "border-emerald-200 bg-emerald-50 text-emerald-700 dark:border-emerald-400/30 dark:bg-emerald-500/10 dark:text-emerald-300",
    red: "border-rose-200 bg-rose-50 text-rose-700 dark:border-rose-400/30 dark:bg-rose-500/10 dark:text-rose-300",
    amber:
      "border-amber-200 bg-amber-50 text-amber-700 dark:border-amber-400/30 dark:bg-amber-500/10 dark:text-amber-300",
    blue: "border-sky-200 bg-sky-50 text-sky-700 dark:border-sky-400/30 dark:bg-sky-500/10 dark:text-sky-300",
  } as const;
  return (
    <span
      className={`inline-flex items-center gap-1 whitespace-nowrap rounded-md border px-2 py-0.5 text-[11px] font-semibold ${tones[tone]} ${className}`}
    >
      {children}
    </span>
  );
}

export function Code({ children }: { children: ReactNode }) {
  return <code className="inline-code">{children}</code>;
}

export function Formula({
  children,
  caption,
  className = "",
}: {
  children: ReactNode;
  caption?: ReactNode;
  className?: string;
}) {
  return (
    <div className={className}>
      <div className="formula thin-scroll">{children}</div>
      {caption && <p className="mt-1.5 text-xs text-faint">{caption}</p>}
    </div>
  );
}

/** A named formula in a bordered card — used heavily in the metrics section. */
export function FormulaCard({
  name,
  formula,
  reads,
  children,
  tone = "neutral",
}: {
  name: ReactNode;
  formula: string;
  /** Plain-English "how to read it" line. */
  reads?: ReactNode;
  children?: ReactNode;
  tone?: "neutral" | "accent";
}) {
  return (
    <div
      className={`card overflow-hidden ${
        tone === "accent" ? "ring-1 ring-indigo-200 dark:ring-indigo-400/25" : ""
      }`}
    >
      <div className="flex flex-wrap items-center justify-between gap-2 border-b border-line bg-surface-2 px-4 py-2">
        <h4 className="text-sm font-bold text-ink">{name}</h4>
      </div>
      <div className="space-y-2.5 p-4">
        <div className="formula thin-scroll !bg-transparent !border-dashed text-center font-semibold">
          {formula}
        </div>
        {reads && (
          <p className="text-[0.86rem] italic leading-relaxed text-faint">“{reads}”</p>
        )}
        {children && (
          <div className="space-y-2 text-[0.9rem] leading-relaxed text-muted">{children}</div>
        )}
      </div>
    </div>
  );
}

/* ------------------------------------------------------------------ */
/* Tables                                                              */
/* ------------------------------------------------------------------ */

export function ComparisonTable({
  headers,
  rows,
  caption,
  dense = false,
}: {
  headers: ReactNode[];
  rows: ReactNode[][];
  caption?: ReactNode;
  dense?: boolean;
}) {
  return (
    <figure className="my-1">
      <div className="thin-scroll overflow-x-auto rounded-xl border border-line bg-surface shadow-[var(--shadow-card)]">
        <table className={`data-table ${dense ? "[&_td]:!py-1.5 [&_td]:!px-2.5 [&_th]:!px-2.5" : ""}`}>
          <thead>
            <tr>
              {headers.map((h, i) => (
                <th key={i}>{h}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {rows.map((row, i) => (
              <tr key={i}>
                {row.map((cell, j) => (
                  <td key={j}>{cell}</td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      {caption && <figcaption className="mt-1.5 text-xs text-faint">{caption}</figcaption>}
    </figure>
  );
}

/** Compact yes/no marker for comparison grids. */
export function YesNo({ v }: { v: "yes" | "no" | "some" | "n/a" }) {
  const map = {
    yes: { t: "Yes", c: "text-emerald-600 dark:text-emerald-400" },
    no: { t: "No", c: "text-rose-600 dark:text-rose-400" },
    some: { t: "Partly", c: "text-amber-600 dark:text-amber-400" },
    "n/a": { t: "—", c: "text-faint" },
  } as const;
  return <span className={`font-semibold ${map[v].c}`}>{map[v].t}</span>;
}

/* ------------------------------------------------------------------ */
/* Layout helpers                                                      */
/* ------------------------------------------------------------------ */

export function Grid({
  cols = 2,
  children,
  className = "",
}: {
  cols?: 2 | 3 | 4;
  children: ReactNode;
  className?: string;
}) {
  const map = {
    2: "sm:grid-cols-2",
    3: "sm:grid-cols-2 lg:grid-cols-3",
    4: "sm:grid-cols-2 lg:grid-cols-4",
  } as const;
  return <div className={`grid grid-cols-1 gap-3 ${map[cols]} ${className}`}>{children}</div>;
}

export function SubHeading({
  children,
  id,
  priority,
  right,
}: {
  children: ReactNode;
  id?: string;
  priority?: Priority;
  right?: ReactNode;
}) {
  return (
    <div id={id} className="anchor-offset mt-2 flex flex-wrap items-center gap-2.5">
      <h3 className="text-[1.05rem] font-bold text-ink">{children}</h3>
      {priority && <PriorityBadge priority={priority} />}
      {right}
    </div>
  );
}

export function Panel({
  title,
  tone = "neutral",
  children,
  className = "",
}: {
  title?: ReactNode;
  tone?: "neutral" | "green" | "red";
  children: ReactNode;
  className?: string;
}) {
  const tones = {
    neutral: "border-line",
    green: "border-emerald-300 dark:border-emerald-400/30",
    red: "border-rose-300 dark:border-rose-400/30",
  } as const;
  const heads = {
    neutral: "text-muted",
    green: "text-emerald-700 dark:text-emerald-300",
    red: "text-rose-700 dark:text-rose-300",
  } as const;
  return (
    <div className={`card border ${tones[tone]} p-4 ${className}`}>
      {title && (
        <div className={`mb-2 text-[11px] font-bold uppercase tracking-[0.09em] ${heads[tone]}`}>
          {title}
        </div>
      )}
      <div className="text-[0.9rem] leading-relaxed text-muted">{children}</div>
    </div>
  );
}

/** Vertical arrow-linked flow diagram. */
export function FlowDiagram({
  steps,
  compact = false,
}: {
  steps: { label: string; note?: string; tone?: "neutral" | "accent" | "green" | "red" }[];
  compact?: boolean;
}) {
  const tones = {
    neutral: "border-line bg-surface-2 text-ink",
    accent:
      "border-indigo-300 bg-indigo-50 text-indigo-800 dark:border-indigo-400/40 dark:bg-indigo-500/12 dark:text-indigo-200",
    green:
      "border-emerald-300 bg-emerald-50 text-emerald-800 dark:border-emerald-400/40 dark:bg-emerald-500/12 dark:text-emerald-200",
    red: "border-rose-300 bg-rose-50 text-rose-800 dark:border-rose-400/40 dark:bg-rose-500/12 dark:text-rose-200",
  } as const;
  return (
    <ol className="flex flex-col items-stretch">
      {steps.map((s, i) => (
        <li key={s.label}>
          <div
            className={`rounded-lg border px-3 ${compact ? "py-1.5" : "py-2"} ${
              tones[s.tone ?? "neutral"]
            }`}
          >
            <div className="text-[0.86rem] font-semibold">{s.label}</div>
            {s.note && <div className="mt-0.5 text-xs font-normal opacity-80">{s.note}</div>}
          </div>
          {i < steps.length - 1 && (
            <div className="flex h-4 items-center justify-center text-faint" aria-hidden>
              <svg width="10" height="16" viewBox="0 0 10 16" fill="none">
                <path
                  d="M5 0v11M1.5 8L5 12l3.5-4"
                  stroke="currentColor"
                  strokeWidth="1.4"
                  strokeLinecap="round"
                  strokeLinejoin="round"
                />
              </svg>
            </div>
          )}
        </li>
      ))}
    </ol>
  );
}
