import { formatMinutes } from "../hooks/useProgress";

/**
 * Overall completion summary: percentage bar, sections done, time remaining.
 * Rendered at the top of the sidebar; the same numbers appear in the hero.
 */
export function ProgressTracker({
  percent,
  completedCount,
  totalCount,
  minutesLeft,
}: {
  percent: number;
  completedCount: number;
  totalCount: number;
  minutesLeft: number;
}) {
  return (
    <div className="px-4 py-3.5">
      <div className="mb-1.5 flex items-baseline justify-between">
        <span className="text-[10.5px] font-bold uppercase tracking-[0.09em] text-faint">
          Progress
        </span>
        <span className="font-mono text-[11.5px] font-bold text-ink">{percent}%</span>
      </div>
      <div
        className="h-1.5 w-full overflow-hidden rounded-full bg-surface-3"
        role="progressbar"
        aria-valuenow={percent}
        aria-valuemin={0}
        aria-valuemax={100}
        aria-label="Review progress"
      >
        <div
          className="h-full rounded-full bg-gradient-to-r from-indigo-500 to-violet-500 transition-[width] duration-500"
          style={{ width: `${percent}%` }}
        />
      </div>
      <div className="mt-1.5 flex items-center justify-between text-[11px] text-faint">
        <span>
          {completedCount}/{totalCount} sections
        </span>
        <span>{formatMinutes(minutesLeft)} left</span>
      </div>
    </div>
  );
}
