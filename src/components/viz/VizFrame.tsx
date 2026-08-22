import type { ReactNode } from "react";

/** Consistent chrome around every interactive demo. */
export function VizFrame({
  title,
  hint,
  children,
  footer,
}: {
  title: string;
  hint?: ReactNode;
  children: ReactNode;
  footer?: ReactNode;
}) {
  return (
    <div className="card overflow-hidden">
      <div className="flex flex-wrap items-center justify-between gap-2 border-b border-line bg-surface-2 px-4 py-2">
        <div className="flex items-center gap-2">
          <span className="text-[11px]" aria-hidden>
            🎛
          </span>
          <h4 className="text-[0.85rem] font-bold uppercase tracking-[0.07em] text-muted">
            {title}
          </h4>
        </div>
        {hint && <span className="text-[11px] text-faint">{hint}</span>}
      </div>
      <div className="p-4">{children}</div>
      {footer && (
        <div className="border-t border-line bg-surface-2 px-4 py-2.5 text-[0.84rem] leading-relaxed text-muted">
          {footer}
        </div>
      )}
    </div>
  );
}

export function Slider({
  label,
  value,
  min,
  max,
  step,
  onChange,
  display,
  leftLabel,
  rightLabel,
}: {
  label: string;
  value: number;
  min: number;
  max: number;
  step: number;
  onChange: (v: number) => void;
  display?: ReactNode;
  leftLabel?: string;
  rightLabel?: string;
}) {
  return (
    <div>
      <div className="mb-1 flex items-baseline justify-between gap-2">
        <label className="text-[11.5px] font-bold uppercase tracking-[0.07em] text-faint">
          {label}
        </label>
        {display !== undefined && (
          <span className="font-mono text-[12.5px] font-bold text-accent">{display}</span>
        )}
      </div>
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(e) => onChange(Number(e.target.value))}
        aria-label={label}
        className="h-1.5 w-full cursor-pointer appearance-none rounded-full bg-surface-3 accent-indigo-500"
      />
      {(leftLabel || rightLabel) && (
        <div className="mt-1 flex justify-between text-[10.5px] font-medium text-faint">
          <span>{leftLabel}</span>
          <span>{rightLabel}</span>
        </div>
      )}
    </div>
  );
}
