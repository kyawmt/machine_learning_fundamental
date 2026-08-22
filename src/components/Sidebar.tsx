import { useMemo, useState } from "react";
import {
  HOURS,
  HOUR_TITLES,
  SECTIONS,
  minutesForHour,
  sectionsForHour,
  type SectionMeta,
} from "../data/sections";
import { ProgressTracker } from "./ProgressTracker";
import { PriorityBadge } from "./ui";

interface SidebarProps {
  active: string;
  isDone: (id: string) => boolean;
  toggleDone: (id: string) => void;
  completedCount: number;
  totalCount: number;
  percent: number;
  minutesLeft: number;
  onNavigate: () => void;
  onReset: () => void;
}

function matches(s: SectionMeta, q: string) {
  const hay = [s.title, s.nav, s.blurb, ...(s.keywords ?? [])].join(" ").toLowerCase();
  return hay.includes(q);
}

export function Sidebar({
  active,
  isDone,
  toggleDone,
  completedCount,
  totalCount,
  percent,
  minutesLeft,
  onNavigate,
  onReset,
}: SidebarProps) {
  const [query, setQuery] = useState("");
  const q = query.trim().toLowerCase();

  const results = useMemo(() => (q ? SECTIONS.filter((s) => matches(s, q)) : null), [q]);

  return (
    <div className="flex h-full flex-col">
      {/* Progress summary */}
      <div className="shrink-0 border-b border-line">
        <ProgressTracker
          percent={percent}
          completedCount={completedCount}
          totalCount={totalCount}
          minutesLeft={minutesLeft}
        />
      </div>

      {/* Search */}
      <div className="shrink-0 border-b border-line px-3 py-2.5">
        <div className="relative">
          <svg
            className="pointer-events-none absolute left-2.5 top-1/2 -translate-y-1/2 text-faint"
            width="13" height="13" viewBox="0 0 24 24" fill="none" aria-hidden
          >
            <circle cx="11" cy="11" r="7" stroke="currentColor" strokeWidth="2.4" />
            <path d="M16.5 16.5L21 21" stroke="currentColor" strokeWidth="2.4" strokeLinecap="round" />
          </svg>
          <input
            type="search"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="Search topics…"
            aria-label="Search topics"
            className="w-full rounded-lg border border-line bg-surface-2 py-1.5 pl-8 pr-2 text-[12.5px] text-ink placeholder:text-faint focus:border-accent focus:outline-none"
          />
        </div>
      </div>

      {/* Nav */}
      <nav className="thin-scroll flex-1 overflow-y-auto px-2 py-3" aria-label="Sections">
        {results ? (
          <div className="space-y-1">
            <div className="px-2 pb-1 text-[10px] font-bold uppercase tracking-[0.09em] text-faint">
              {results.length} match{results.length === 1 ? "" : "es"}
            </div>
            {results.map((s) => (
              <NavItem
                key={s.id}
                s={s}
                active={active === s.id}
                done={isDone(s.id)}
                onToggle={() => toggleDone(s.id)}
                onNavigate={onNavigate}
              />
            ))}
            {results.length === 0 && (
              <p className="px-2 py-4 text-[12px] text-faint">
                Nothing matched. Try “recall”, “leakage”, “boosting”…
              </p>
            )}
          </div>
        ) : (
          <div className="space-y-4">
            <div className="space-y-1">
              <NavItem
                s={SECTIONS[0]}
                active={active === SECTIONS[0].id}
                done={isDone(SECTIONS[0].id)}
                onToggle={() => toggleDone(SECTIONS[0].id)}
                onNavigate={onNavigate}
                hideCheck
              />
            </div>

            {HOURS.map((h) => {
              const items = sectionsForHour(h);
              const doneCount = items.filter((s) => isDone(s.id)).length;
              return (
                <div key={h}>
                  <div className="flex items-baseline justify-between gap-2 px-2 pb-1.5">
                    <span className="min-w-0 flex-1 text-[10px] font-bold uppercase leading-tight tracking-[0.09em] text-accent">
                      {HOUR_TITLES[h].title}
                    </span>
                    <span className="shrink-0 whitespace-nowrap font-mono text-[9.5px] text-faint">
                      {doneCount}/{items.length} · {minutesForHour(h)}m
                    </span>
                  </div>
                  <div className="space-y-0.5">
                    {items.map((s) => (
                      <NavItem
                        key={s.id}
                        s={s}
                        active={active === s.id}
                        done={isDone(s.id)}
                        onToggle={() => toggleDone(s.id)}
                        onNavigate={onNavigate}
                      />
                    ))}
                  </div>
                </div>
              );
            })}
          </div>
        )}
      </nav>

      {/* Legend + reset */}
      <div className="shrink-0 space-y-2 border-t border-line px-3 py-3">
        <div className="flex flex-wrap gap-1.5">
          <PriorityBadge priority="must" />
          <PriorityBadge priority="important" />
          <PriorityBadge priority="good" />
        </div>
        <button
          type="button"
          onClick={() => {
            if (window.confirm("Clear all progress, quiz score and checklists?")) onReset();
          }}
          className="w-full rounded-lg border border-line px-2 py-1.5 text-[11px] font-semibold text-faint transition-colors hover:border-rose-300 hover:text-rose-600 dark:hover:border-rose-400/40 dark:hover:text-rose-400"
        >
          Reset all progress
        </button>
      </div>
    </div>
  );
}

function NavItem({
  s,
  active,
  done,
  onToggle,
  onNavigate,
  hideCheck,
}: {
  s: SectionMeta;
  active: boolean;
  done: boolean;
  onToggle: () => void;
  onNavigate: () => void;
  hideCheck?: boolean;
}) {
  return (
    <div
      className={`group flex items-center gap-1.5 rounded-lg pr-1 transition-colors ${
        active ? "bg-accent-soft" : "hover:bg-surface-2"
      }`}
    >
      <a
        href={`#${s.id}`}
        onClick={onNavigate}
        aria-current={active ? "true" : undefined}
        className="flex min-w-0 flex-1 items-center gap-2 rounded-lg py-1.5 pl-2"
      >
        <PriorityBadge priority={s.priority} compact />
        <span
          className={`min-w-0 flex-1 truncate text-[12.5px] leading-tight ${
            active ? "font-bold text-accent" : done ? "font-medium text-faint" : "font-medium text-muted"
          }`}
        >
          {s.nav}
        </span>
        {s.minutes > 0 && (
          <span className="shrink-0 font-mono text-[9.5px] text-faint opacity-70">{s.minutes}m</span>
        )}
      </a>
      {!hideCheck && (
        <button
          type="button"
          onClick={onToggle}
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
      )}
    </div>
  );
}
