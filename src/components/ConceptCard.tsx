import type { ReactNode } from "react";
import type { Priority } from "../data/sections";
import { Callout } from "./Callout";
import { DefinitionCard } from "./DefinitionCard";
import { PriorityBadge } from "./ui";

interface ConceptCardProps {
  title: ReactNode;
  id?: string;
  priority?: Priority;
  tagline?: ReactNode;
  /** "What it is" — one concise definition. */
  whatItIs?: ReactNode;
  /** "Intuition" — plain language. */
  intuition?: ReactNode;
  /** "Example" — practical. */
  example?: ReactNode;
  /** "Interview Answer" — what to actually say out loud. */
  interviewAnswer?: ReactNode;
  /** "Common Mistake" */
  commonMistake?: ReactNode;
  /** "Key Takeaway" — one or two memorizable sentences. */
  keyTakeaway?: ReactNode;
  /** Free-form extra content rendered between the definition and the interview answer. */
  children?: ReactNode;
}

export function ConceptCard({
  title,
  id,
  priority,
  tagline,
  whatItIs,
  intuition,
  example,
  interviewAnswer,
  commonMistake,
  keyTakeaway,
  children,
}: ConceptCardProps) {
  return (
    <article id={id} className="card anchor-offset overflow-hidden">
      <div className="flex flex-wrap items-center gap-2.5 border-b border-line bg-surface-2 px-4 py-2.5 sm:px-5">
        <h3 className="text-[1.02rem] font-bold text-ink">{title}</h3>
        {priority && <PriorityBadge priority={priority} />}
        {tagline && <span className="text-xs text-faint">{tagline}</span>}
      </div>

      <div className="@container space-y-3 p-4 sm:p-5">
        {whatItIs && <DefinitionCard>{whatItIs}</DefinitionCard>}

        {(intuition || example) && (
          <div
            className={`grid grid-cols-1 gap-3 ${intuition && example ? "@2xl:grid-cols-2" : ""}`}
          >
            {intuition && <Callout variant="intuition">{intuition}</Callout>}
            {example && <Callout variant="example">{example}</Callout>}
          </div>
        )}

        {children}

        {interviewAnswer && (
          <Callout variant="interview">
            <span className="text-ink">{interviewAnswer}</span>
          </Callout>
        )}

        {commonMistake && <Callout variant="mistake">{commonMistake}</Callout>}

        {keyTakeaway && (
          <Callout variant="memorize" title="Key Takeaway">
            <span className="font-medium text-ink">{keyTakeaway}</span>
          </Callout>
        )}
      </div>
    </article>
  );
}
