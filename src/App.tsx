import { useCallback, useEffect, useMemo, useState } from "react";
import { SECTIONS, TRACKED_SECTIONS } from "./data/sections";
import { useActiveSection } from "./hooks/useActiveSection";
import { useProgress } from "./hooks/useProgress";
import { useTheme } from "./hooks/useTheme";
import { Sidebar } from "./components/Sidebar";
import { TopicSection } from "./components/TopicSection";
import { Hero, StudyPlan } from "./sections/StudyPlan";
import {
  BiasVariance,
  CrossValidation,
  DataLeakage,
  OverUnderfitting,
  ProblemTypes,
  Splits,
} from "./sections/hour1";
import { Metrics } from "./sections/metrics";
import {
  Categorical,
  ClassImbalance,
  FeatureEngineering,
  FeatureScaling,
  MissingData,
} from "./sections/hour2";
import { GradientDescent, ParamsHyperparams, Regularization } from "./sections/hour3";
import { Algorithms } from "./sections/algorithms";
import {
  Clustering,
  DimensionalityReduction,
  Ensembles,
  ModelSelection,
  Pipeline,
} from "./sections/hour4";
import { InterviewQuestions } from "./sections/InterviewQuestions";
import { Quiz } from "./sections/Quiz";
import { CheatSheet, ReadinessCheck } from "./sections/CheatSheet";

const SECTION_BODIES: Record<string, () => React.ReactElement> = {
  "problem-types": ProblemTypes,
  splits: Splits,
  "data-leakage": DataLeakage,
  "over-underfitting": OverUnderfitting,
  "bias-variance": BiasVariance,
  metrics: Metrics,
  "cross-validation": CrossValidation,
  "feature-engineering": FeatureEngineering,
  "feature-scaling": FeatureScaling,
  "missing-data": MissingData,
  categorical: Categorical,
  "class-imbalance": ClassImbalance,
  regularization: Regularization,
  "gradient-descent": GradientDescent,
  "params-hyperparams": ParamsHyperparams,
  algorithms: Algorithms,
  "model-selection": ModelSelection,
  ensembles: Ensembles,
  "dimensionality-reduction": DimensionalityReduction,
  clustering: Clustering,
  pipeline: Pipeline,
  "interview-questions": InterviewQuestions,
  quiz: Quiz,
  "cheat-sheet": CheatSheet,
};

const SECTION_IDS = SECTIONS.map((s) => s.id);

export default function App() {
  const { theme, toggle } = useTheme();
  const progress = useProgress();
  const active = useActiveSection(SECTION_IDS);
  const [drawerOpen, setDrawerOpen] = useState(false);

  // Close the mobile drawer on Escape.
  useEffect(() => {
    if (!drawerOpen) return;
    const onKey = (e: KeyboardEvent) => e.key === "Escape" && setDrawerOpen(false);
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [drawerOpen]);

  const nextSection = useMemo(
    () => TRACKED_SECTIONS.find((s) => !progress.map[s.id]) ?? TRACKED_SECTIONS[0],
    [progress.map],
  );

  const resetEverything = useCallback(() => {
    for (const key of Object.keys(window.localStorage)) {
      if (key.startsWith("mlprep.") && key !== "mlprep.theme") window.localStorage.removeItem(key);
    }
    window.location.reload();
  }, []);

  const activeMeta = SECTIONS.find((s) => s.id === active);

  return (
    <div className="min-h-screen">
      {/* ---------------- top bar ---------------- */}
      <header className="no-print sticky top-0 z-40 border-b border-line bg-surface/88 backdrop-blur-md">
        <div className="h-0.5 w-full bg-surface-3">
          <div
            className="h-full bg-gradient-to-r from-indigo-500 to-violet-500 transition-[width] duration-500"
            style={{ width: `${progress.stats.percent}%` }}
          />
        </div>
        <div className="flex h-[54px] items-center gap-3 px-3 sm:px-5">
          <button
            type="button"
            onClick={() => setDrawerOpen((o) => !o)}
            aria-label="Toggle navigation"
            aria-expanded={drawerOpen}
            className="grid size-8 shrink-0 place-items-center rounded-lg border border-line text-muted hover:text-ink lg:hidden"
          >
            <svg width="15" height="15" viewBox="0 0 24 24" fill="none" aria-hidden>
              <path d="M4 6h16M4 12h16M4 18h16" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" />
            </svg>
          </button>

          <a href="#top" className="flex min-w-0 items-center gap-2.5">
            <span
              className="grid size-7 shrink-0 place-items-center rounded-lg bg-gradient-to-br from-indigo-500 to-violet-600 text-[13px] font-black text-white"
              aria-hidden
            >
              ML
            </span>
            <span className="min-w-0">
              <span className="block truncate text-[13.5px] font-bold leading-tight text-ink">
                ML Fundamentals
              </span>
              <span className="block truncate text-[10.5px] leading-tight text-faint">
                4-Hour AI Engineer Interview Review
              </span>
            </span>
          </a>

          {activeMeta && (
            <span className="ml-2 hidden min-w-0 items-center gap-2 border-l border-line pl-3 text-[12px] text-faint xl:flex">
              <span className="truncate font-semibold text-muted">{activeMeta.nav}</span>
            </span>
          )}

          <div className="ml-auto flex items-center gap-2">
            <span className="hidden text-[11.5px] font-semibold text-faint sm:inline">
              {progress.stats.completedCount}/{progress.stats.totalCount} · {progress.stats.percent}%
            </span>
            <button
              type="button"
              onClick={toggle}
              aria-label={`Switch to ${theme === "dark" ? "light" : "dark"} mode`}
              className="grid size-8 place-items-center rounded-lg border border-line text-muted transition-colors hover:text-ink"
            >
              {theme === "dark" ? (
                <svg width="15" height="15" viewBox="0 0 24 24" fill="none" aria-hidden>
                  <circle cx="12" cy="12" r="4.2" stroke="currentColor" strokeWidth="2" />
                  <path
                    d="M12 2v2.5M12 19.5V22M2 12h2.5M19.5 12H22M4.9 4.9l1.8 1.8M17.3 17.3l1.8 1.8M19.1 4.9l-1.8 1.8M6.7 17.3l-1.8 1.8"
                    stroke="currentColor"
                    strokeWidth="2"
                    strokeLinecap="round"
                  />
                </svg>
              ) : (
                <svg width="15" height="15" viewBox="0 0 24 24" fill="none" aria-hidden>
                  <path
                    d="M20 14.5A8.5 8.5 0 019.5 4a8.5 8.5 0 1010.5 10.5z"
                    stroke="currentColor"
                    strokeWidth="2"
                    strokeLinejoin="round"
                  />
                </svg>
              )}
            </button>
          </div>
        </div>
      </header>

      <div className="mx-auto flex max-w-[1500px]">
        {/* ---------------- sidebar ---------------- */}
        <aside className="no-print sticky top-[54px] hidden h-[calc(100vh-54px)] w-[268px] shrink-0 border-r border-line bg-surface lg:block">
          <Sidebar
            active={active}
            isDone={progress.has}
            toggleDone={progress.toggle}
            completedCount={progress.stats.completedCount}
            totalCount={progress.stats.totalCount}
            percent={progress.stats.percent}
            minutesLeft={progress.stats.minutesLeft}
            onNavigate={() => undefined}
            onReset={resetEverything}
          />
        </aside>

        {/* mobile drawer */}
        {drawerOpen && (
          <div className="no-print fixed inset-0 z-50 lg:hidden">
            <button
              type="button"
              aria-label="Close navigation"
              onClick={() => setDrawerOpen(false)}
              className="absolute inset-0 bg-slate-900/50 backdrop-blur-sm"
            />
            <div className="absolute inset-y-0 left-0 w-[290px] max-w-[85vw] border-r border-line bg-surface shadow-2xl">
              <Sidebar
                active={active}
                isDone={progress.has}
                toggleDone={progress.toggle}
                completedCount={progress.stats.completedCount}
                totalCount={progress.stats.totalCount}
                percent={progress.stats.percent}
                minutesLeft={progress.stats.minutesLeft}
                onNavigate={() => setDrawerOpen(false)}
                onReset={resetEverything}
              />
            </div>
          </div>
        )}

        {/* ---------------- main ---------------- */}
        <main id="top" className="min-w-0 flex-1 px-4 pb-24 pt-6 sm:px-8 lg:px-10">
          <div className="mx-auto max-w-[900px]">
            <Hero
              percent={progress.stats.percent}
              minutesLeft={progress.stats.minutesLeft}
              completedCount={progress.stats.completedCount}
              totalCount={progress.stats.totalCount}
              nextSectionId={nextSection.id}
              nextSectionTitle={nextSection.title}
            />

            <div className="space-y-14">
              {SECTIONS.map((meta, i) => {
                const Body = SECTION_BODIES[meta.id];
                return (
                  <TopicSection
                    key={meta.id}
                    meta={meta}
                    index={i + 1}
                    done={progress.has(meta.id)}
                    onToggleDone={() => progress.toggle(meta.id)}
                  >
                    {meta.id === "study-plan" ? (
                      <StudyPlan isDone={progress.has} toggleDone={progress.toggle} />
                    ) : Body ? (
                      <Body />
                    ) : null}
                    {meta.id === "cheat-sheet" && <ReadinessCheck />}
                  </TopicSection>
                );
              })}
            </div>

            <footer className="mt-16 border-t border-line pt-6 text-center text-[12px] leading-relaxed text-faint">
              <p>
                Progress, quiz scores and checklists are stored in your browser's localStorage. Nothing is sent
                anywhere.
              </p>
              <p className="mt-1">
                <a href="#top" className="font-semibold text-accent hover:underline">
                  ↑ Back to top
                </a>
              </p>
            </footer>
          </div>
        </main>
      </div>
    </div>
  );
}
