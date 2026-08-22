import { Fragment, type ReactNode } from "react";

/**
 * Minimal inline formatter so the content data files can stay plain strings.
 * Supports **bold**, `code`, and _italic_ inline; blank-line paragraphs and
 * "- " bullet lines at block level. Deliberately tiny — no HTML is injected.
 */

const INLINE = /(\*\*[^*]+\*\*|`[^`]+`|_[^_]+_)/g;

function inline(text: string, keyBase: string): ReactNode[] {
  return text.split(INLINE).filter(Boolean).map((part, i) => {
    const key = `${keyBase}-${i}`;
    if (part.startsWith("**") && part.endsWith("**")) {
      return (
        <strong key={key} className="font-semibold text-ink">
          {part.slice(2, -2)}
        </strong>
      );
    }
    if (part.startsWith("`") && part.endsWith("`")) {
      return (
        <code key={key} className="inline-code">
          {part.slice(1, -1)}
        </code>
      );
    }
    if (part.startsWith("_") && part.endsWith("_")) {
      return (
        <em key={key} className="italic">
          {part.slice(1, -1)}
        </em>
      );
    }
    return <Fragment key={key}>{part}</Fragment>;
  });
}

export function RichText({ text, className = "" }: { text: string; className?: string }) {
  const blocks = text.trim().split(/\n{2,}/);

  return (
    <div className={`space-y-2 ${className}`}>
      {blocks.map((block, bi) => {
        const lines = block.split("\n");
        const isList = lines.every((l) => l.trim().startsWith("- "));

        if (isList) {
          return (
            <ul key={bi} className="bullet-list">
              {lines.map((l, li) => (
                <li key={li}>{inline(l.trim().slice(2), `${bi}-${li}`)}</li>
              ))}
            </ul>
          );
        }

        return (
          <p key={bi} className="leading-relaxed">
            {lines.map((l, li) => (
              <Fragment key={li}>
                {li > 0 && <br />}
                {inline(l, `${bi}-${li}`)}
              </Fragment>
            ))}
          </p>
        );
      })}
    </div>
  );
}
