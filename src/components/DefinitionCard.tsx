import type { ReactNode } from "react";
import { Callout } from "./Callout";

/**
 * The "what it is" block: one concise definition, visually distinct from
 * intuition, example and interview-answer content.
 */
export function DefinitionCard({
  term,
  children,
  className,
}: {
  /** Overrides the default "Definition" label — e.g. "The mechanism". */
  term?: string;
  children: ReactNode;
  className?: string;
}) {
  return (
    <Callout variant="definition" title={term} className={className}>
      {children}
    </Callout>
  );
}
