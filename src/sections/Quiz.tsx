import { Callout } from "../components/Callout";
import { QuizEngine } from "../components/QuizCard";
import { QUIZ } from "../data/quiz";

export function Quiz() {
  return (
    <>
      <Callout variant="tip" title="How to run this">
        <ul className="bullet-list">
          <li>
            <strong>Say the answer out loud before you reveal.</strong> Recognising an answer is not the same as
            producing one, and only production is tested in an interview.
          </li>
          <li>
            Grade yourself honestly. "I sort of knew it" is a miss — the whole point of the round is to find your
            gaps while it's still cheap.
          </li>
          <li>
            After a round, use <strong>Drill the misses</strong> to loop just the ones you got wrong, then re-read
            those sections.
          </li>
          <li>
            <strong>90%+</strong> means you're ready. <strong>Below 70%</strong> means go back through the
            red-priority sections before doing anything else.
          </li>
        </ul>
      </Callout>

      <QuizEngine items={QUIZ} />

      <p className="text-center text-[12px] text-faint">
        {QUIZ.length} questions across {new Set(QUIZ.map((q) => q.topic)).size} topics. Your best score is saved
        locally.
      </p>
    </>
  );
}
