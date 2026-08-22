import { useCallback, useMemo, useState } from "react";
import { RichText } from "./RichText";
import { useLocalStorage } from "../hooks/useLocalStorage";

export interface QuizItem {
  id: string;
  q: string;
  a: string;
  topic: string;
}

function shuffled<T>(arr: T[]): T[] {
  const a = [...arr];
  for (let i = a.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [a[i], a[j]] = [a[j], a[i]];
  }
  return a;
}

export function QuizEngine({ items }: { items: QuizItem[] }) {
  const topics = useMemo(
    () => ["All topics", ...Array.from(new Set(items.map((i) => i.topic)))],
    [items],
  );

  const [topic, setTopic] = useState("All topics");
  const [shuffle, setShuffle] = useState(true);
  const [best, setBest] = useLocalStorage<number | null>("quiz.best", null);

  const pool = useMemo(
    () => (topic === "All topics" ? items : items.filter((i) => i.topic === topic)),
    [items, topic],
  );

  const [order, setOrder] = useState<QuizItem[]>(() => shuffled(items));
  const [pos, setPos] = useState(0);
  const [revealed, setRevealed] = useState(false);
  const [correct, setCorrect] = useState<string[]>([]);
  const [missed, setMissed] = useState<string[]>([]);
  const [finished, setFinished] = useState(false);

  const start = useCallback(
    (nextPool: QuizItem[], doShuffle: boolean) => {
      setOrder(doShuffle ? shuffled(nextPool) : nextPool);
      setPos(0);
      setRevealed(false);
      setCorrect([]);
      setMissed([]);
      setFinished(false);
    },
    [],
  );

  const answered = correct.length + missed.length;
  const total = order.length;
  const current = order[pos];

  const grade = (ok: boolean) => {
    if (!current) return;
    if (ok) setCorrect((c) => [...c, current.id]);
    else setMissed((m) => [...m, current.id]);

    if (pos + 1 >= total) {
      const finalCorrect = ok ? correct.length + 1 : correct.length;
      const pct = total === 0 ? 0 : Math.round((finalCorrect / total) * 100);
      setBest((b) => (b === null || pct > b ? pct : b));
      setFinished(true);
    } else {
      setPos((p) => p + 1);
      setRevealed(false);
    }
  };

  const missedItems = order.filter((i) => missed.includes(i.id));
  const pct = total === 0 ? 0 : Math.round((correct.length / total) * 100);

  return (
    <div className="card overflow-hidden">
      {/* Controls */}
      <div className="flex flex-wrap items-center gap-2 border-b border-line bg-surface-2 px-4 py-2.5">
        <label className="flex items-center gap-1.5 text-[11.5px] font-semibold text-muted">
          Topic
          <select
            value={topic}
            onChange={(e) => {
              const t = e.target.value;
              setTopic(t);
              const nextPool = t === "All topics" ? items : items.filter((i) => i.topic === t);
              start(nextPool, shuffle);
            }}
            className="rounded-md border border-line bg-surface px-2 py-1 text-[11.5px] font-semibold text-ink"
          >
            {topics.map((t) => (
              <option key={t} value={t}>
                {t}
              </option>
            ))}
          </select>
        </label>

        <label className="flex cursor-pointer items-center gap-1.5 text-[11.5px] font-semibold text-muted">
          <input
            type="checkbox"
            checked={shuffle}
            onChange={(e) => {
              setShuffle(e.target.checked);
              start(pool, e.target.checked);
            }}
            className="size-3.5 accent-indigo-500"
          />
          Shuffle
        </label>

        <button
          type="button"
          onClick={() => start(pool, shuffle)}
          className="rounded-md border border-line bg-surface px-2.5 py-1 text-[11.5px] font-semibold text-muted transition-colors hover:text-ink"
        >
          ↺ Restart
        </button>

        <div className="ml-auto flex items-center gap-3 text-[11.5px] font-semibold">
          <span className="text-emerald-600 dark:text-emerald-400">✓ {correct.length}</span>
          <span className="text-rose-600 dark:text-rose-400">✕ {missed.length}</span>
          {best !== null && <span className="text-faint">best {best}%</span>}
        </div>
      </div>

      {/* Progress */}
      <div className="h-1 w-full bg-surface-3">
        <div
          className="h-full bg-gradient-to-r from-indigo-500 to-violet-500 transition-[width] duration-300"
          style={{ width: `${total === 0 ? 0 : (answered / total) * 100}%` }}
        />
      </div>

      {finished ? (
        <div className="space-y-4 p-5">
          <div className="text-center">
            <div className="text-[11px] font-bold uppercase tracking-[0.09em] text-faint">
              Round complete
            </div>
            <div className="mt-1 text-4xl font-extrabold text-ink">{pct}%</div>
            <div className="mt-1 text-sm text-muted">
              {correct.length} of {total} correct
              {best !== null && best > pct ? ` · personal best ${best}%` : ""}
            </div>
            <p className="mx-auto mt-3 max-w-md text-[0.88rem] text-faint">
              {pct >= 90
                ? "Recall is solid. Move to the cheat sheet and the readiness check."
                : pct >= 70
                  ? "Good. Re-read the sections behind the misses below, then run the quiz again."
                  : "Go back through the missed topics before re-running — this is exactly what the round is for."}
            </p>
          </div>

          {missedItems.length > 0 && (
            <div>
              <div className="mb-2 text-[11px] font-bold uppercase tracking-[0.09em] text-rose-600 dark:text-rose-400">
                Review these {missedItems.length}
              </div>
              <ul className="space-y-1.5">
                {missedItems.map((i) => (
                  <li
                    key={i.id}
                    className="rounded-lg border border-line bg-surface-2 px-3 py-2 text-[0.86rem]"
                  >
                    <div className="font-semibold text-ink">{i.q}</div>
                    <div className="mt-0.5 text-muted">
                      <RichText text={i.a} />
                    </div>
                  </li>
                ))}
              </ul>
            </div>
          )}

          <div className="flex flex-wrap justify-center gap-2">
            <button
              type="button"
              onClick={() => start(pool, shuffle)}
              className="rounded-lg bg-accent px-4 py-2 text-[13px] font-bold text-white hover:opacity-90"
            >
              Run again
            </button>
            {missedItems.length > 0 && (
              <button
                type="button"
                onClick={() => start(missedItems, shuffle)}
                className="rounded-lg border border-line bg-surface px-4 py-2 text-[13px] font-bold text-muted hover:text-ink"
              >
                Drill the {missedItems.length} misses
              </button>
            )}
          </div>
        </div>
      ) : current ? (
        <div className="p-5">
          <div className="mb-3 flex items-center justify-between text-[11px] font-semibold text-faint">
            <span>
              Question {pos + 1} / {total}
            </span>
            <span className="rounded-md border border-line bg-surface-2 px-2 py-0.5">
              {current.topic}
            </span>
          </div>

          <p className="min-h-14 text-[1.08rem] font-semibold leading-snug text-ink">{current.q}</p>

          <div className="mt-4">
            {!revealed ? (
              <button
                type="button"
                onClick={() => setRevealed(true)}
                className="w-full rounded-lg border border-dashed border-line-strong bg-surface-2 py-3 text-[13px] font-bold text-muted transition-colors hover:border-accent hover:text-accent"
              >
                Reveal answer
              </button>
            ) : (
              <div className="space-y-3">
                <div className="rounded-lg border border-emerald-200 bg-emerald-50/70 px-3.5 py-2.5 dark:border-emerald-400/25 dark:bg-emerald-400/8">
                  <RichText text={current.a} className="text-[0.94rem] text-ink" />
                </div>
                <div className="flex gap-2">
                  <button
                    type="button"
                    onClick={() => grade(true)}
                    className="flex-1 rounded-lg border border-emerald-300 bg-emerald-50 py-2 text-[13px] font-bold text-emerald-700 hover:bg-emerald-100 dark:border-emerald-400/35 dark:bg-emerald-500/12 dark:text-emerald-300 dark:hover:bg-emerald-500/20"
                  >
                    ✓ I knew it
                  </button>
                  <button
                    type="button"
                    onClick={() => grade(false)}
                    className="flex-1 rounded-lg border border-rose-300 bg-rose-50 py-2 text-[13px] font-bold text-rose-700 hover:bg-rose-100 dark:border-rose-400/35 dark:bg-rose-500/12 dark:text-rose-300 dark:hover:bg-rose-500/20"
                  >
                    ✕ Missed it
                  </button>
                </div>
              </div>
            )}
          </div>
        </div>
      ) : (
        <div className="p-5 text-center text-sm text-faint">No questions in this topic.</div>
      )}
    </div>
  );
}
