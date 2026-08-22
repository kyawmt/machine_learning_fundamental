import type { ReactNode } from "react";
import type { Priority } from "../data/sections";
import { Callout } from "./Callout";
import { InterviewQuestion, type QA } from "./InterviewQuestion";
import { Chip, Formula, PriorityBadge } from "./ui";

export interface AlgorithmSpec {
  id: string;
  name: string;
  priority: Priority;
  /** classification / regression / both */
  task: string;
  whatItDoes: ReactNode;
  intuition: ReactNode;
  formula?: string;
  formulaCaption?: ReactNode;
  strengths: string[];
  weaknesses: string[];
  hyperparams: string[];
  scaling: "required" | "not-required" | "helpful";
  nonlinear: "yes" | "no" | "with-kernel" | "with-features";
  interpretable: "high" | "medium" | "low";
  extra?: ReactNode;
  questions: QA[];
}

const SCALING_CHIP = {
  required: { tone: "red" as const, label: "Scaling required" },
  helpful: { tone: "amber" as const, label: "Scaling helps" },
  "not-required": { tone: "green" as const, label: "No scaling needed" },
};

const NONLINEAR_CHIP = {
  yes: { tone: "green" as const, label: "Nonlinear natively" },
  no: { tone: "neutral" as const, label: "Linear only" },
  "with-kernel": { tone: "blue" as const, label: "Nonlinear via kernel" },
  "with-features": { tone: "amber" as const, label: "Nonlinear only via features" },
};

const INTERP_CHIP = {
  high: { tone: "green" as const, label: "Highly interpretable" },
  medium: { tone: "amber" as const, label: "Somewhat interpretable" },
  low: { tone: "neutral" as const, label: "Low interpretability" },
};

export function AlgorithmCard({ a }: { a: AlgorithmSpec }) {
  const s = SCALING_CHIP[a.scaling];
  const n = NONLINEAR_CHIP[a.nonlinear];
  const i = INTERP_CHIP[a.interpretable];

  return (
    <article id={a.id} className="card anchor-offset overflow-hidden">
      <div className="border-b border-line bg-surface-2 px-4 py-3 sm:px-5">
        <div className="flex flex-wrap items-center gap-2.5">
          <h3 className="text-[1.08rem] font-extrabold text-ink">{a.name}</h3>
          <PriorityBadge priority={a.priority} />
          <span className="text-[11px] font-semibold text-faint">{a.task}</span>
        </div>
        <div className="mt-2 flex flex-wrap gap-1.5">
          <Chip tone={s.tone}>{s.label}</Chip>
          <Chip tone={n.tone}>{n.label}</Chip>
          <Chip tone={i.tone}>{i.label}</Chip>
        </div>
      </div>

      <div className="space-y-3 p-4 sm:p-5">
        <Callout variant="definition" title="What it does">
          {a.whatItDoes}
        </Callout>

        {a.formula && <Formula caption={a.formulaCaption}>{a.formula}</Formula>}

        <Callout variant="intuition">{a.intuition}</Callout>

        {a.extra}

        <div className="grid grid-cols-1 gap-3 lg:grid-cols-2">
          <div className="rounded-xl border border-emerald-200 bg-emerald-50/60 p-3 dark:border-emerald-400/25 dark:bg-emerald-400/8">
            <div className="mb-1.5 text-[10.5px] font-bold uppercase tracking-[0.09em] text-emerald-700 dark:text-emerald-300">
              ✓ Strengths
            </div>
            <ul className="bullet-list text-[0.87rem]">
              {a.strengths.map((x) => (
                <li key={x}>{x}</li>
              ))}
            </ul>
          </div>
          <div className="rounded-xl border border-rose-200 bg-rose-50/60 p-3 dark:border-rose-400/25 dark:bg-rose-400/8">
            <div className="mb-1.5 text-[10.5px] font-bold uppercase tracking-[0.09em] text-rose-700 dark:text-rose-300">
              ✕ Weaknesses
            </div>
            <ul className="bullet-list text-[0.87rem]">
              {a.weaknesses.map((x) => (
                <li key={x}>{x}</li>
              ))}
            </ul>
          </div>
        </div>

        <div>
          <div className="mb-1.5 text-[10.5px] font-bold uppercase tracking-[0.09em] text-faint">
            Hyperparameters worth knowing
          </div>
          <div className="flex flex-wrap gap-1.5">
            {a.hyperparams.map((h) => (
              <span
                key={h}
                className="rounded-md border border-line bg-surface-2 px-2 py-0.5 font-mono text-[11px] font-semibold text-muted"
              >
                {h}
              </span>
            ))}
          </div>
        </div>

        {a.questions.length > 0 && (
          <div>
            <div className="mb-1.5 text-[10.5px] font-bold uppercase tracking-[0.09em] text-faint">
              Interview questions
            </div>
            <div className="space-y-2">
              {a.questions.map((q) => (
                <InterviewQuestion key={q.id} qa={q} />
              ))}
            </div>
          </div>
        )}
      </div>
    </article>
  );
}
