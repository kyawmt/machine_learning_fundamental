import { useMemo } from "react";
import { TRACKED_SECTIONS, TOTAL_MINUTES } from "../data/sections";
import { useCheckedSet } from "./useLocalStorage";

export function useProgress() {
  const done = useCheckedSet("progress.sections");

  const stats = useMemo(() => {
    const completed = TRACKED_SECTIONS.filter((s) => done.map[s.id]);
    const minutesDone = completed.reduce((sum, s) => sum + s.minutes, 0);
    return {
      completedCount: completed.length,
      totalCount: TRACKED_SECTIONS.length,
      minutesDone,
      minutesLeft: TOTAL_MINUTES - minutesDone,
      percent: TOTAL_MINUTES === 0 ? 0 : Math.round((minutesDone / TOTAL_MINUTES) * 100),
    };
  }, [done.map]);

  return { ...done, stats };
}

export function formatMinutes(mins: number): string {
  if (mins <= 0) return "0m";
  const h = Math.floor(mins / 60);
  const m = mins % 60;
  if (h === 0) return `${m}m`;
  if (m === 0) return `${h}h`;
  return `${h}h ${m}m`;
}
