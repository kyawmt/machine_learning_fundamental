import type { ReactNode } from "react";

export type CalloutVariant =
  | "definition"
  | "intuition"
  | "example"
  | "interview"
  | "mistake"
  | "memorize"
  | "tip"
  | "warning"
  | "good"
  | "bad";

interface VariantStyle {
  label: string;
  icon: string;
  wrap: string;
  head: string;
}

const VARIANTS: Record<CalloutVariant, VariantStyle> = {
  definition: {
    label: "Definition",
    icon: "◆",
    wrap: "border-indigo-200 bg-indigo-50/70 dark:border-indigo-400/25 dark:bg-indigo-400/8",
    head: "text-indigo-700 dark:text-indigo-300",
  },
  intuition: {
    label: "Intuition",
    icon: "◒",
    wrap: "border-sky-200 bg-sky-50/70 dark:border-sky-400/25 dark:bg-sky-400/8",
    head: "text-sky-700 dark:text-sky-300",
  },
  example: {
    label: "Example",
    icon: "▸",
    wrap: "border-emerald-200 bg-emerald-50/70 dark:border-emerald-400/25 dark:bg-emerald-400/8",
    head: "text-emerald-700 dark:text-emerald-300",
  },
  interview: {
    label: "Interview Answer",
    icon: "🎙",
    wrap: "border-violet-200 bg-violet-50/70 dark:border-violet-400/25 dark:bg-violet-400/8",
    head: "text-violet-700 dark:text-violet-300",
  },
  mistake: {
    label: "Common Mistake",
    icon: "⚠",
    wrap: "border-rose-200 bg-rose-50/70 dark:border-rose-400/25 dark:bg-rose-400/8",
    head: "text-rose-700 dark:text-rose-300",
  },
  memorize: {
    label: "Memorize This",
    icon: "★",
    wrap: "border-amber-300 bg-amber-50/80 dark:border-amber-400/30 dark:bg-amber-400/10",
    head: "text-amber-700 dark:text-amber-300",
  },
  tip: {
    label: "Interview Tip",
    icon: "✎",
    wrap: "border-teal-200 bg-teal-50/70 dark:border-teal-400/25 dark:bg-teal-400/8",
    head: "text-teal-700 dark:text-teal-300",
  },
  warning: {
    label: "Watch Out",
    icon: "!",
    wrap: "border-orange-200 bg-orange-50/70 dark:border-orange-400/25 dark:bg-orange-400/8",
    head: "text-orange-700 dark:text-orange-300",
  },
  good: {
    label: "Correct",
    icon: "✓",
    wrap: "border-emerald-300 bg-emerald-50/80 dark:border-emerald-400/30 dark:bg-emerald-400/10",
    head: "text-emerald-700 dark:text-emerald-300",
  },
  bad: {
    label: "Wrong",
    icon: "✕",
    wrap: "border-rose-300 bg-rose-50/80 dark:border-rose-400/30 dark:bg-rose-400/10",
    head: "text-rose-700 dark:text-rose-300",
  },
};

interface CalloutProps {
  variant: CalloutVariant;
  /** Overrides the default variant label. Pass `null` to hide the header. */
  title?: string | null;
  children: ReactNode;
  className?: string;
}

export function Callout({ variant, title, children, className = "" }: CalloutProps) {
  const v = VARIANTS[variant];
  return (
    <div className={`rounded-xl border px-4 py-3 ${v.wrap} ${className}`}>
      {title !== null && (
        <div
          className={`mb-1.5 flex items-center gap-1.5 text-[11px] font-bold uppercase tracking-[0.09em] ${v.head}`}
        >
          <span aria-hidden className="text-[12px] leading-none">
            {v.icon}
          </span>
          {title ?? v.label}
        </div>
      )}
      <div className="text-[0.9rem] leading-relaxed text-muted [&_strong]:text-ink [&_strong]:font-semibold">
        {children}
      </div>
    </div>
  );
}
