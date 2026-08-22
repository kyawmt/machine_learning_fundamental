import { useMemo, useState } from "react";
import { Callout } from "../components/Callout";
import { InterviewQuestion } from "../components/InterviewQuestion";
import { ScenarioCard } from "../components/ScenarioCard";
import { SubHeading } from "../components/ui";
import { QA_GROUPS, TOTAL_QUESTIONS } from "../data/interviewQuestions";
import { SCENARIOS } from "../data/scenarios";

export function InterviewQuestions() {
  const [filter, setFilter] = useState<string>("all");
  const [query, setQuery] = useState("");

  const q = query.trim().toLowerCase();

  const groups = useMemo(
    () =>
      QA_GROUPS.map((g) => ({
        ...g,
        questions: q
          ? g.questions.filter((item) =>
              `${item.q} ${item.short} ${item.strong}`.toLowerCase().includes(q),
            )
          : g.questions,
      })).filter((g) => (filter === "all" || filter === g.id) && g.questions.length > 0),
    [filter, q],
  );

  const visible = groups.reduce((n, g) => n + g.questions.length, 0);

  return (
    <>
      <Callout variant="tip" title="Use these out loud">
        Reading an answer and being able to say it are different skills, and only one of them is being tested.
        Cover the answer, say yours out loud, then compare. The "say this in the interview" version is written the
        way you'd actually speak — roughly 30–60 seconds each, with a concrete detail so it doesn't sound
        rehearsed.
      </Callout>

      <div className="no-print sticky top-[68px] z-20 -mx-2 rounded-xl border border-line bg-surface/92 px-2 py-2 backdrop-blur">
        <div className="flex flex-wrap items-center gap-1.5">
          <button
            type="button"
            onClick={() => setFilter("all")}
            className={`rounded-lg border px-2.5 py-1 text-[12px] font-bold transition-colors ${
              filter === "all"
                ? "border-accent bg-accent text-white"
                : "border-line bg-surface-2 text-muted hover:text-ink"
            }`}
          >
            All {TOTAL_QUESTIONS}
          </button>
          {QA_GROUPS.map((g) => (
            <button
              key={g.id}
              type="button"
              onClick={() => setFilter(g.id)}
              className={`rounded-lg border px-2.5 py-1 text-[12px] font-bold transition-colors ${
                filter === g.id
                  ? "border-accent bg-accent text-white"
                  : "border-line bg-surface-2 text-muted hover:text-ink"
              }`}
            >
              {g.title}
              <span className="ml-1 opacity-60">{g.questions.length}</span>
            </button>
          ))}
          <input
            type="search"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="Filter questions…"
            aria-label="Filter interview questions"
            className="ml-auto w-40 rounded-lg border border-line bg-surface-2 px-2.5 py-1 text-[12px] text-ink placeholder:text-faint focus:border-accent focus:outline-none sm:w-52"
          />
        </div>
      </div>

      {q && (
        <p className="text-[12px] text-faint">
          {visible} question{visible === 1 ? "" : "s"} matching “{query}”.
        </p>
      )}

      {groups.map((g) => (
        <div key={g.id} className="space-y-2.5">
          <SubHeading id={g.id} priority="must">
            {g.title}
          </SubHeading>
          <p className="-mt-1 text-[0.88rem] text-faint">{g.blurb}</p>
          {g.questions.map((item, i) => (
            <InterviewQuestion key={item.id} qa={item} n={i + 1} />
          ))}
        </div>
      ))}

      {groups.length === 0 && (
        <p className="rounded-xl border border-dashed border-line px-4 py-8 text-center text-sm text-faint">
          No questions matched “{query}”.
        </p>
      )}

      {/* ---------------- Scenarios ---------------- */}
      <SubHeading id="scenarios" priority="must">
        Scenario drills — {SCENARIOS.length} situations
      </SubHeading>
      <p className="-mt-1 text-[0.9rem] text-muted">
        These are the questions that separate candidates. Each one is a situation you'd genuinely meet, with
        numbers that point somewhere specific. <strong>Work out your answer before you reveal</strong> — the value
        is entirely in the attempt.
      </p>

      <div className="space-y-3">
        {SCENARIOS.map((s, i) => (
          <ScenarioCard key={s.id} s={s} n={i + 1} />
        ))}
      </div>

      <Callout variant="memorize" title="The shape of a good scenario answer">
        <ol className="space-y-1 pl-5 [&>li]:list-decimal [&>li]:text-muted">
          <li>
            <strong>Name the phenomenon.</strong> "That's overfitting", "that looks like leakage", "the metric is
            hiding the imbalance."
          </li>
          <li>
            <strong>Say what you'd check to confirm it</strong> — before proposing fixes. This is what distinguishes
            a diagnosis from a guess.
          </li>
          <li>
            <strong>Give two or three fixes, ordered</strong>, with a reason for the order.
          </li>
          <li>
            <strong>Name the tradeoff or the thing you'd still be unsure about.</strong> Certainty about an
            underspecified problem is a bad signal; interviewers are listening for judgement.
          </li>
        </ol>
      </Callout>
    </>
  );
}
