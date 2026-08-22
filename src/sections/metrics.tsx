import { Callout } from "../components/Callout";
import { DefinitionCard } from "../components/DefinitionCard";
import { ConceptCard } from "../components/ConceptCard";
import { InterviewQuestion } from "../components/InterviewQuestion";
import {
  Chip,
  Code,
  ComparisonTable,
  Formula,
  FormulaCard,
  Grid,
  SubHeading,
} from "../components/ui";
import { ThresholdLab } from "../components/viz/ThresholdLab";

export function Metrics() {
  return (
    <>
      <Callout variant="tip" title="This is the section that decides the interview">
        Metrics questions are where interviewers separate people who have trained a model from people who have
        <em> shipped</em> one. Almost every question here has the same correct shape: name the metric, say what it
        optimises, and then tie it to the cost of the two error types in that specific business context.
      </Callout>

      {/* ---------------- Confusion matrix ---------------- */}
      <SubHeading id="confusion-matrix" priority="must">The Confusion Matrix</SubHeading>
      <ConceptCard
        title="Everything in classification comes from these four numbers"
        whatItIs="A 2×2 table of predicted class against actual class. Every classification metric is an arithmetic combination of its cells."
        intuition="Two ways to be right and two ways to be wrong. Which of the two wrongs hurts more is a business question, not a maths question."
      >
        <div className="grid gap-3 lg:grid-cols-[minmax(0,1fr)_minmax(0,1fr)]">
          <div className="thin-scroll overflow-x-auto">
            <table className="w-full border-collapse text-center text-[0.85rem]">
              <thead>
                <tr>
                  <th className="p-2" />
                  <th className="border-b border-line p-2 text-[11px] font-bold uppercase tracking-[0.07em] text-faint">
                    Predicted Positive
                  </th>
                  <th className="border-b border-line p-2 text-[11px] font-bold uppercase tracking-[0.07em] text-faint">
                    Predicted Negative
                  </th>
                </tr>
              </thead>
              <tbody>
                <tr>
                  <th className="border-r border-line p-2 text-right text-[11px] font-bold uppercase tracking-[0.07em] text-faint">
                    Actual Positive
                  </th>
                  <td className="p-1.5">
                    <MatrixCell code="TP" name="True Positive" note="caught it" tone="good" />
                  </td>
                  <td className="p-1.5">
                    <MatrixCell code="FN" name="False Negative" note="missed it · Type II" tone="bad" />
                  </td>
                </tr>
                <tr>
                  <th className="border-r border-line p-2 text-right text-[11px] font-bold uppercase tracking-[0.07em] text-faint">
                    Actual Negative
                  </th>
                  <td className="p-1.5">
                    <MatrixCell code="FP" name="False Positive" note="false alarm · Type I" tone="bad" />
                  </td>
                  <td className="p-1.5">
                    <MatrixCell code="TN" name="True Negative" note="correctly ignored" tone="good" />
                  </td>
                </tr>
              </tbody>
            </table>
          </div>
          <Callout variant="memorize" title="Never mix these up">
            <ul className="bullet-list">
              <li>
                The <strong>second word is what the model predicted</strong>. "False positive" = predicted positive,
                and that prediction was false.
              </li>
              <li>
                <strong>Precision</strong> divides by what you <em>predicted</em> positive (the column).{" "}
                <strong>Recall</strong> divides by what <em>actually is</em> positive (the row).
              </li>
              <li>
                FN = Type II error = a miss. FP = Type I error = a false alarm.
              </li>
            </ul>
          </Callout>
        </div>
      </ConceptCard>

      <ThresholdLab />

      {/* ---------------- Core metrics ---------------- */}
      <SubHeading id="classification-metrics" priority="must">Classification Metrics</SubHeading>

      <Grid cols={2}>
        <FormulaCard
          name="Accuracy"
          formula="(TP + TN) / (TP + TN + FP + FN)"
          reads="Of all predictions, how many were right?"
        >
          <p>
            Fine when classes are roughly balanced <em>and</em> both error types cost about the same. Badly
            misleading otherwise.
          </p>
          <p className="rounded-md bg-rose-50 px-2 py-1.5 text-[0.85rem] text-rose-700 dark:bg-rose-500/10 dark:text-rose-300">
            At 1% fraud, "always predict legitimate" scores <strong>99% accuracy</strong> and catches nothing. Always
            compare accuracy against the majority-class baseline.
          </p>
        </FormulaCard>

        <FormulaCard
          name="Precision"
          formula="TP / (TP + FP)"
          reads="Of everything I flagged, how much was real?"
          tone="accent"
        >
          <p>
            Optimise precision when a <strong>false alarm is expensive</strong>: it wastes a human's time, annoys a
            customer, or triggers an action that's hard to undo.
          </p>
          <p>
            <strong>Spam filter.</strong> A legitimate email in the junk folder is far worse than a spam message in
            the inbox — a missed invoice costs more than a nuisance. So a spam filter runs at high precision.
          </p>
        </FormulaCard>

        <FormulaCard
          name="Recall (Sensitivity, TPR)"
          formula="TP / (TP + FN)"
          reads="Of everything that was real, how much did I catch?"
          tone="accent"
        >
          <p>
            Optimise recall when a <strong>miss is expensive</strong>, and especially when a cheap second stage can
            filter the false alarms.
          </p>
          <p>
            <strong>Cancer screening.</strong> A missed tumour can be fatal; a false positive leads to a follow-up
            test. So screening runs at high recall and accepts the extra follow-ups.
          </p>
        </FormulaCard>

        <FormulaCard
          name="F1 Score"
          formula="2 · (Precision × Recall) / (Precision + Recall)"
          reads="The harmonic mean of precision and recall."
        >
          <p>
            Harmonic, not arithmetic — so it's dragged down by the weaker of the two. Precision 1.0 with recall 0.02
            gives F1 = 0.04, whereas a plain average would flatter it at 0.51.
          </p>
          <p>
            Use it as a single tie-breaking number when you need precision and recall to both be decent and have no
            strong reason to prefer one. If you <em>do</em> have a reason, F<sub>β</sub> is more honest: β &gt; 1
            weights recall, β &lt; 1 weights precision.
          </p>
        </FormulaCard>

        <FormulaCard
          name="Specificity (True Negative Rate)"
          formula="TN / (TN + FP)"
          reads="Of everything that was genuinely negative, how much did I correctly leave alone?"
        >
          <p>
            The mirror image of recall. Common in medicine, where a test is described by its sensitivity and
            specificity pair.
          </p>
          <p>
            <Code>FPR = 1 − Specificity</Code> — that's the x-axis of the ROC curve.
          </p>
        </FormulaCard>

        <FormulaCard
          name="Log Loss (Cross-Entropy)"
          formula="−(1/N) Σ [ y·log(p) + (1−y)·log(1−p) ]"
          reads="How badly calibrated and how confident were the wrong predictions?"
        >
          <p>
            Scores the <strong>probabilities</strong>, not the hard labels, and punishes confident mistakes
            brutally — predicting 0.99 for a negative costs far more than predicting 0.6.
          </p>
          <p>
            Worth mentioning whenever the downstream system uses the probability itself (expected value, ranking,
            bidding) rather than just a yes/no.
          </p>
        </FormulaCard>
      </Grid>

      {/* ---------------- Curves ---------------- */}
      <SubHeading priority="must">ROC and Precision–Recall curves</SubHeading>
      <Grid cols={2}>
        <ConceptCard
          title="ROC Curve & ROC-AUC"
          whatItIs={
            <>
              Sweep the decision threshold from 1 to 0 and plot <strong>TPR (recall)</strong> against{" "}
              <strong>FPR (1 − specificity)</strong>. The area under that curve is ROC-AUC.
            </>
          }
          intuition="AUC has a beautifully concrete meaning: it is the probability that a randomly chosen positive gets a higher score than a randomly chosen negative. It measures ranking quality, independent of any threshold."
        >
          <Formula>{`AUC = 1.0   perfect ranking
AUC = 0.9   strong
AUC = 0.8   useful
AUC = 0.7   modest
AUC = 0.5   coin flip — no signal
AUC < 0.5   your ranking is inverted (check the label!)`}</Formula>
          <Callout variant="mistake">
            ROC-AUC is <strong>optimistic on heavily imbalanced data</strong>. FPR has a huge denominator (all the
            negatives), so thousands of false positives barely move the x-axis. A model can look excellent by AUC and
            still hand your analysts an alert queue that's 95% noise.
          </Callout>
        </ConceptCard>

        <ConceptCard
          title="PR Curve & PR-AUC"
          whatItIs="Precision plotted against recall as the threshold sweeps. The area under it — usually reported as average precision — summarises performance on the positive class alone."
          intuition="PR-AUC ignores true negatives entirely. When 99% of your data is negative, that's exactly what you want: it refuses to give the model credit for the easy part of the problem."
        >
          <Formula>{`Baseline (no-skill) for ROC-AUC  =  0.5   (always)
Baseline (no-skill) for PR-AUC   =  the positive rate

At 1% positives, PR-AUC = 0.30 is a strong model.
At 50% positives, PR-AUC = 0.30 is worse than random.`}</Formula>
          <Callout variant="memorize">
            Always report the positive rate alongside PR-AUC. Unlike ROC-AUC, the number is meaningless without it.
          </Callout>
        </ConceptCard>
      </Grid>

      <SubHeading priority="must">Which metric, when</SubHeading>
      <ComparisonTable
        headers={["Scenario", "Preferred metric", "Why"]}
        rows={[
          ["Balanced classes, symmetric costs", <strong key="a">Accuracy</strong>, "Simple, interpretable, and nothing is hidden by the class ratio."],
          ["Missing a positive is costly", <strong key="b">Recall</strong>, "Disease screening, safety alerts, security triage. A miss is the expensive error."],
          ["A false alarm is costly", <strong key="c">Precision</strong>, "Spam, auto-blocking, anything that interrupts a human or reverses a transaction."],
          ["Need one balanced number", <strong key="d">F1</strong>, "Punishes the weaker of the two, so you can't game it by collapsing to one extreme."],
          ["Strong class imbalance", <strong key="e">PR-AUC</strong>, "Ignores the flood of true negatives that inflates ROC-AUC."],
          ["Comparing rankers, threshold undecided", <strong key="f">ROC-AUC</strong>, "Threshold-independent measure of how well the model orders cases."],
          ["Probabilities feed a downstream calculation", <strong key="g">Log loss / Brier + calibration curve</strong>, "You need the probability to mean what it says, not just to rank correctly."],
          ["Fixed review capacity (e.g. 100 alerts/day)", <strong key="h">Precision@K / Recall@K</strong>, "Matches the actual operating constraint better than any global metric."],
          ["Regression", <strong key="i">MAE or RMSE</strong>, "Pick based on how much you care about large errors specifically."],
        ]}
      />

      <Callout variant="interview" title="The fraud question — how to answer without falling in the trap">
        <p className="mb-2">
          <strong>"For fraud detection, should you optimise precision or recall?"</strong> The wrong answer is
          "recall, because missing fraud is bad." That's a rule, and the interviewer is testing whether you reach
          for rules or for reasoning.
        </p>
        <p className="mb-2">
          <strong>Say this instead:</strong> "It depends on the cost of a false positive versus a false negative,
          and on what happens after the model fires. If a flag means the transaction is auto-declined, a false
          positive is a real customer blocked at checkout — that's churn and support cost, so precision matters a
          lot. If a flag means it lands in a review queue, a false positive costs a few minutes of an analyst's
          time, so I'd push recall much harder — subject to the queue's capacity. In practice I'd put a dollar
          figure on each error type, pick the threshold that minimises expected cost, and use PR-AUC to compare
          models because the classes are heavily imbalanced."
        </p>
        <p>
          That answer names the tradeoff, ties it to the operational design, and gives a decision procedure. It
          works for medical screening, content moderation and credit risk with the nouns swapped.
        </p>
      </Callout>

      <Grid cols={2}>
        <DefinitionCard term="Multiclass averaging — know the three">
          <ul className="bullet-list">
            <li>
              <strong>Macro</strong> — compute the metric per class, then average unweighted. Every class counts
              equally, so rare classes matter as much as common ones.
            </li>
            <li>
              <strong>Weighted</strong> — average per-class metrics weighted by class support. Closer to accuracy;
              rare classes get drowned out.
            </li>
            <li>
              <strong>Micro</strong> — pool all TP/FP/FN across classes then compute once. For single-label
              multiclass, micro-F1 equals accuracy.
            </li>
          </ul>
          <p className="mt-2">
            If you care about rare classes, <strong>macro</strong>. If you care about overall throughput,{" "}
            <strong>weighted</strong> or <strong>micro</strong>. Saying which one you chose and why is a strong signal.
          </p>
        </DefinitionCard>

        <DefinitionCard term="Calibration — the metric people forget">
          <p className="mb-2">
            A model is <strong>calibrated</strong> if, among cases it scores 0.7, roughly 70% really are positive.
            AUC can be perfect while calibration is terrible — AUC only cares about order.
          </p>
          <ul className="bullet-list">
            <li>Check with a reliability diagram, or the Brier score / log loss.</li>
            <li>Fix with Platt scaling (a logistic fit on the scores) or isotonic regression, fitted on held-out data.</li>
            <li>
              Matters whenever the probability is <em>used</em>: expected-value pricing, thresholding on business
              cost, ad bidding, risk scoring.
            </li>
          </ul>
        </DefinitionCard>
      </Grid>

      {/* ---------------- Regression ---------------- */}
      <SubHeading id="regression-metrics" priority="must">Regression Metrics</SubHeading>
      <Grid cols={2}>
        <FormulaCard name="MAE — Mean Absolute Error" formula="MAE = (1/n) Σ |yᵢ − ŷᵢ|" reads="On average, how far off am I?">
          <p>
            Same units as the target, so it's directly interpretable: "we're off by $8,400 on average." Robust to
            outliers — every error counts in proportion to its size, no more.
          </p>
          <p>
            The model that minimises MAE predicts the <strong>conditional median</strong>.
          </p>
        </FormulaCard>

        <FormulaCard name="MSE — Mean Squared Error" formula="MSE = (1/n) Σ (yᵢ − ŷᵢ)²" reads="Average squared error — big misses count much more.">
          <p>
            Squaring means a 10-unit error contributes 100× what a 1-unit error does. Great when large errors are
            disproportionately bad; terrible when your data has outliers you don't want to chase.
          </p>
          <p>
            Units are squared (dollars²), so it's hard to interpret directly — but it's smooth and differentiable,
            which is why it's the default training loss.
          </p>
        </FormulaCard>

        <FormulaCard name="RMSE — Root Mean Squared Error" formula="RMSE = √MSE" reads="MSE brought back into the target's own units.">
          <p>
            Keeps MSE's heavy penalty on large errors but is readable again. Always ≥ MAE; the{" "}
            <strong>gap between RMSE and MAE tells you how skewed your errors are</strong> — if RMSE is much bigger,
            a few big misses dominate.
          </p>
          <p>
            The model that minimises RMSE/MSE predicts the <strong>conditional mean</strong>.
          </p>
        </FormulaCard>

        <FormulaCard name="R² — Coefficient of Determination" formula="R² = 1 − (SS_res / SS_tot)" reads="What fraction of the variance in y does the model explain?">
          <p>
            1.0 is perfect. 0 means you did no better than always predicting the mean.{" "}
            <strong>Negative R² is possible</strong> and means you did worse than that — common on a test set when
            the model is badly overfitted.
          </p>
          <p>
            Unitless, so it's comparable across problems — but it's also inflated by adding any feature at all, which
            is why <strong>adjusted R²</strong> exists.
          </p>
        </FormulaCard>
      </Grid>

      <ComparisonTable
        headers={["Metric", "Units", "Outlier sensitivity", "Optimal prediction", "Reach for it when"]}
        rows={[
          ["MAE", "Same as y", "Low — robust", "Conditional median", "Errors are roughly equally bad per unit; the data has outliers you don't want to chase."],
          ["MSE", "y²", "High", "Conditional mean", "You're training. Smooth and differentiable."],
          ["RMSE", "Same as y", "High", "Conditional mean", "You're reporting, and big misses genuinely matter more."],
          ["R²", "None", "Medium", "—", "Explaining to a non-technical audience how much signal you captured."],
          ["MAPE", "%", "High near zero", "—", "Errors should be judged relative to size — but it explodes when y ≈ 0 and punishes over-prediction asymmetrically."],
          ["Huber loss", "Same as y", "Tunable", "Between mean and median", "You want MSE's smoothness near zero and MAE's robustness in the tails."],
        ]}
      />

      <Callout variant="memorize" title="The MAE vs MSE answer in one breath">
        MSE squares the errors, so it penalises large errors disproportionately and is pulled toward outliers; MAE
        treats every unit of error equally and is robust. Choose MSE/RMSE when one big miss is much worse than
        several small ones — say, an ETA that's two hours out. Choose MAE when a few extreme rows are noise you
        don't want the model to chase.
      </Callout>

      <SubHeading>Interview questions</SubHeading>
      <div className="space-y-2">
        <InterviewQuestion
          qa={{
            id: "m1",
            q: "Explain precision and recall to a non-technical stakeholder.",
            short: "Precision: when we raise a flag, how often are we right? Recall: of everything we should have flagged, how much did we actually catch?",
            strong:
              "I'd use their own numbers. 'Last month we flagged 200 transactions. 140 of them really were fraud — that's 75% precision, so one in four of our alerts wastes an analyst's time. Over the same month there were 350 fraudulent transactions in total, and we caught 140 — that's 40% recall, so we're missing three out of five. If you want us to catch more, we can lower the bar, but the alert queue will grow and a bigger share of it will be false alarms. Which of those two costs more to you?' Framing it as a dial they control, with a real tradeoff, is much more useful than defining the terms.",
          }}
        />
        <InterviewQuestion
          qa={{
            id: "m2",
            q: "When is ROC-AUC misleading, and what would you use instead?",
            short: "On heavily imbalanced data. FPR barely moves when negatives dominate, so ROC-AUC flatters the model. Use PR-AUC.",
            strong:
              "ROC-AUC's x-axis is FPR = FP / (FP + TN). With 99% negatives, TN is enormous, so even a large absolute number of false positives produces a tiny FPR. The curve hugs the top-left and AUC looks great, while the actual alert queue is mostly noise. PR-AUC uses precision, whose denominator is TP + FP — only the things you flagged — so it responds sharply to false positives. Concretely: 1,000 positives and 999,000 negatives, 900 TP and 9,000 FP gives ROC-AUC around 0.95 but precision of just 9%. One caveat when reporting PR-AUC: its no-skill baseline is the positive rate, not 0.5, so you always have to state the base rate with it.",
          }}
        />
        <InterviewQuestion
          qa={{
            id: "m3",
            q: "Your model has 95% accuracy. Is it good?",
            short: "Unanswerable without the class balance and the cost of each error type.",
            strong:
              "My first question would be the base rate. If 95% of the data is one class, then 95% accuracy is exactly what predicting the majority class gets you, and the model has learned nothing. Even with balanced classes I'd want to know what the errors are — 5% errors concentrated entirely in the class we care about is a very different product than 5% spread evenly. So I'd ask for the confusion matrix, the majority-class baseline, and what the two error types cost. Accuracy is a summary that hides exactly the information that matters.",
          }}
        />
        <InterviewQuestion
          qa={{
            id: "m4",
            q: "Why is F1 a harmonic mean rather than an arithmetic one?",
            short: "So a model can't score well by maxing one metric and abandoning the other. The harmonic mean is dominated by the smaller value.",
            strong:
              "Take precision 1.0 and recall 0.02 — a model that flags one single case and happens to be right. The arithmetic mean is 0.51, which makes a useless model look middling. The harmonic mean is 2(1 × 0.02)/(1.02) = 0.039, which correctly says it's useless. The harmonic mean is always pulled toward the smaller number, so F1 only gets high when both are high. That's the whole design goal: it's a metric you can't game by collapsing to one extreme.",
          }}
        />
        <InterviewQuestion
          qa={{
            id: "m5",
            q: "What does R² = 0.85 actually mean, and can it be negative?",
            short: "The model explains 85% of the variance in the target. And yes — negative R² means you're doing worse than predicting the mean.",
            strong:
              "R² compares your model's residual sum of squares to the residual sum of squares of the constant model that always predicts the mean of y. 0.85 means you've eliminated 85% of that baseline error. On the training set of an OLS fit R² is bounded in [0, 1], but on held-out data there's nothing forcing your model to beat the mean — a badly overfitted model can and does score below zero. That's actually a useful diagnostic: negative test R² is a loud signal of overfitting or of a distribution shift between train and test.",
            deeper:
              "One trap: R² never decreases when you add a feature, even a column of random noise, because the extra parameter can only reduce training residuals. Adjusted R² penalises the parameter count for exactly this reason. And R² is not comparable across datasets with different target variance — the same model quality on a low-variance target yields a lower R².",
          }}
        />
      </div>

      <div className="flex flex-wrap gap-2">
        <Chip tone="accent">precision = flag quality</Chip>
        <Chip tone="accent">recall = coverage</Chip>
        <Chip tone="accent">AUC = ranking</Chip>
        <Chip tone="accent">log loss = calibration</Chip>
        <Chip tone="accent">RMSE = big errors hurt</Chip>
        <Chip tone="accent">MAE = robust</Chip>
      </div>
    </>
  );
}

function MatrixCell({
  code,
  name,
  note,
  tone,
}: {
  code: string;
  name: string;
  note: string;
  tone: "good" | "bad";
}) {
  const cls =
    tone === "good"
      ? "border-emerald-300 bg-emerald-50 text-emerald-700 dark:border-emerald-400/30 dark:bg-emerald-500/10 dark:text-emerald-300"
      : "border-rose-300 bg-rose-50 text-rose-700 dark:border-rose-400/30 dark:bg-rose-500/10 dark:text-rose-300";
  return (
    <div className={`rounded-lg border px-2 py-2 ${cls}`}>
      <div className="font-mono text-base font-extrabold">{code}</div>
      <div className="text-[11px] font-semibold">{name}</div>
      <div className="text-[10px] opacity-75">{note}</div>
    </div>
  );
}
