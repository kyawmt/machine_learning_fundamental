import type { ReactNode } from "react";
import { Callout } from "../components/Callout";
import { useCheckedSet } from "../hooks/useLocalStorage";

/* ------------------------------------------------------------------ */
/* Cheat sheet                                                         */
/* ------------------------------------------------------------------ */

function Block({
  title,
  accent = "indigo",
  children,
}: {
  title: string;
  accent?: "indigo" | "violet" | "emerald" | "amber" | "rose" | "sky";
  children: ReactNode;
}) {
  const bar = {
    indigo: "bg-indigo-500",
    violet: "bg-violet-500",
    emerald: "bg-emerald-500",
    amber: "bg-amber-500",
    rose: "bg-rose-500",
    sky: "bg-sky-500",
  }[accent];
  return (
    <div className="card overflow-hidden break-inside-avoid">
      <div className="flex items-center gap-2 border-b border-line bg-surface-2 px-3 py-1.5">
        <span className={`h-3 w-1 rounded-full ${bar}`} aria-hidden />
        <h4 className="text-[11.5px] font-bold uppercase tracking-[0.07em] text-ink">{title}</h4>
      </div>
      <div className="space-y-1 p-3 text-[0.83rem] leading-relaxed">{children}</div>
    </div>
  );
}

function Row({ k, v }: { k: ReactNode; v: ReactNode }) {
  return (
    <div className="flex gap-2">
      <span className="min-w-[100px] shrink-0 font-semibold text-ink">{k}</span>
      <span className="min-w-0 flex-1 break-words text-muted">{v}</span>
    </div>
  );
}

function Mono({ children }: { children: ReactNode }) {
  return <span className="font-mono text-[0.8rem] break-all text-muted">{children}</span>;
}

/** Print just this section, not the whole 25-section page. */
function printCheatSheet() {
  const cleanup = () => {
    document.body.classList.remove("print-cheat");
    window.removeEventListener("afterprint", cleanup);
  };
  window.addEventListener("afterprint", cleanup);
  document.body.classList.add("print-cheat");
  window.print();
  // Safari fires afterprint unreliably; belt and braces.
  window.setTimeout(cleanup, 1000);
}

export function CheatSheet() {
  return (
    <>
      <div className="no-print flex flex-wrap items-center justify-between gap-2 rounded-xl border border-line bg-surface-2 px-4 py-2.5">
        <p className="text-[0.87rem] text-muted">
          Everything worth having in short-term memory, on one page. Read it the morning of the interview.
        </p>
        <button
          type="button"
          onClick={printCheatSheet}
          className="rounded-lg border border-line bg-surface px-3 py-1.5 text-[12px] font-bold text-muted transition-colors hover:text-ink"
        >
          ⎙ Print / save as PDF
        </button>
      </div>

      <div className="grid grid-cols-1 gap-3 md:grid-cols-2 xl:grid-cols-3">
        <Block title="Core diagnosis" accent="indigo">
          <Row k="Overfitting" v={<>train <strong>good</strong>, validation <strong>bad</strong> → high variance</>} />
          <Row k="Underfitting" v={<>train <strong>bad</strong>, validation <strong>bad</strong> → high bias</>} />
          <Row k="Good fit" v="both good, gap small" />
          <Row k="Bias" v="model too simple — consistently wrong the same way" />
          <Row k="Variance" v="model too sensitive — changes a lot between samples" />
          <Row k="Total error" v={<Mono>bias² + variance + irreducible</Mono>} />
        </Block>

        <Block title="Classification metrics" accent="violet">
          <Row k="Accuracy" v={<Mono>(TP + TN) / Total</Mono>} />
          <Row k="Precision" v={<Mono>TP / (TP + FP)</Mono>} />
          <Row k="Recall / TPR" v={<Mono>TP / (TP + FN)</Mono>} />
          <Row k="Specificity" v={<Mono>TN / (TN + FP)</Mono>} />
          <Row k="FPR" v={<Mono>FP / (FP + TN) = 1 − specificity</Mono>} />
          <Row k="F1" v="harmonic mean of precision and recall" />
          <Row k="ROC-AUC" v="P(random positive ranked above random negative)" />
          <Row k="PR-AUC" v="baseline = positive rate; use when imbalanced" />
        </Block>

        <Block title="Regression metrics" accent="emerald">
          <Row k="MAE" v={<Mono>mean(|y − ŷ|)</Mono>} />
          <Row k="MSE" v={<Mono>mean((y − ŷ)²)</Mono>} />
          <Row k="RMSE" v={<Mono>√MSE</Mono>} />
          <Row k="R²" v={<Mono>1 − SSres / SStot</Mono>} />
          <Row k="MSE vs MAE" v="MSE punishes big errors more; MAE is robust to outliers" />
          <Row k="Optimal pred." v="MSE → conditional mean · MAE → conditional median" />
        </Block>

        <Block title="Regularization" accent="amber">
          <Row k="L1 (Lasso)" v={<><Mono>Loss + λΣ|w|</Mono> — sparse, feature selection</>} />
          <Row k="L2 (Ridge)" v={<><Mono>Loss + λΣw²</Mono> — shrinkage, keeps everything</>} />
          <Row k="Elastic Net" v="both — for wide, correlated data" />
          <Row k="λ ↑" v="bias ↑, variance ↓ (eventually underfits)" />
          <Row k="Also counts" v="dropout, early stopping, max_depth, subsample, augmentation" />
          <Row k="Prerequisite" v="scale your features first; don't penalise the intercept" />
        </Block>

        <Block title="Scaling" accent="sky">
          <Row
            k="Needs it"
            v="KNN · SVM · K-Means · PCA · neural networks · regularized linear models"
          />
          <Row k="Doesn't" v="decision tree · random forest · gradient boosted trees · Naive Bayes" />
          <Row k="Why not (trees)" v="splits depend only on value ordering, which rescaling preserves" />
          <Row k="Standardize" v={<Mono>z = (x − μ) / σ</Mono>} />
          <Row k="Min-max" v={<Mono>x' = (x − min) / (max − min)</Mono>} />
          <Row k="Rule" v={<><Mono>fit_transform</Mono> on train, <Mono>transform</Mono> elsewhere</>} />
        </Block>

        <Block title="Algorithms in one line" accent="rose">
          <Row k="Linear reg." v="regression; interpretable coefficients" />
          <Row k="Logistic reg." v="classification; linear in log-odds; calibrated probabilities" />
          <Row k="Decision tree" v="interpretable rules; overfits alone" />
          <Row k="Random forest" v="bagging; parallel deep trees; cuts variance" />
          <Row k="Gradient boosting" v="sequential error correction; cuts bias; tabular SOTA" />
          <Row k="KNN" v="distance-based; no training; costly inference" />
          <Row k="SVM" v="maximum margin; kernel trick; small high-dim data" />
          <Row k="Naive Bayes" v="probabilistic; independence assumption; fast text baseline" />
        </Block>

        <Block title="Key relationships" accent="indigo">
          <Row k="Complexity ↑" v="bias ↓ · variance ↑" />
          <Row k="Regularization ↑" v="bias ↑ · variance ↓" />
          <Row k="More data" v="variance ↓ · bias ≈ unchanged" />
          <Row k="Threshold ↑" v="precision ↑ (usually) · recall ↓ (always)" />
          <Row k="Threshold ↓" v="recall ↑ (always) · precision ↓ (usually)" />
          <Row k="K in KNN ↑" v="bias ↑ · variance ↓" />
          <Row k="Tree depth ↑" v="bias ↓ · variance ↑" />
          <Row k="GBM lr ↓" v="needs more trees; generalises better" />
        </Block>

        <Block title="Splits & cross-validation" accent="violet">
          <Row k="Train" v="fit parameters" />
          <Row k="Validation" v="choose model + hyperparameters + threshold" />
          <Row k="Test" v="one honest number, opened once" />
          <Row k="Ratios" v="80/10/10 · 70/15/15 · 80/20 with CV" />
          <Row k="Classification" v="StratifiedKFold" />
          <Row k="Repeated entities" v="GroupKFold" />
          <Row k="Time series" v="TimeSeriesSplit — never shuffle" />
          <Row k="CV + test?" v="Yes. CV replaces validation, not test" />
        </Block>

        <Block title="Leakage checklist" accent="rose">
          <Row k="Split first" v="before any fitted transformation" />
          <Row k="Pipeline it" v="scaler, imputer, encoder, PCA, selector, SMOTE" />
          <Row k="Timing" v="would this value exist, with this value, at prediction time?" />
          <Row k="Duplicates" v="deduplicate before splitting" />
          <Row k="Target encoding" v="out-of-fold only" />
          <Row k="Resampling" v="inside the training fold only" />
          <Row k="The tell" v="a metric that's implausibly good" />
        </Block>

        <Block title="Class imbalance" accent="amber">
          <Row k="1. Metric" v="PR-AUC, precision/recall at an operating point" />
          <Row k="2. Threshold" v="free, instant, often the whole fix" />
          <Row k="3. Class weights" v={<Mono>class_weight="balanced"</Mono>} />
          <Row k="4. Resample" v="SMOTE / under / over — training fold only" />
          <Row k="Warning" v="weights and resampling break calibration — recalibrate if needed" />
          <Row k="Never" v="resample the validation or test set" />
        </Block>

        <Block title="Gradient descent" accent="emerald">
          <Row k="Update" v={<Mono>θ ← θ − η∇J(θ)</Mono>} />
          <Row k="Why minus" v="the gradient points uphill; we're minimising" />
          <Row k="Batch" v="all rows — exact, smooth, slow" />
          <Row k="SGD" v="one row — noisy, fast steps" />
          <Row k="Mini-batch" v="32–512 — the practical default" />
          <Row k="η too high" v="oscillates, diverges, NaN" />
          <Row k="η too low" v="painfully slow, looks flat" />
          <Row k="Optimizers" v="SGD+momentum · RMSProp · Adam · AdamW" />
        </Block>

        <Block title="Encoding & missing data" accent="sky">
          <Row k="One-hot" v="unordered, low cardinality" />
          <Row k="Ordinal" v="genuinely ordered levels only" />
          <Row k="Target enc." v="high cardinality; out-of-fold + smoothing" />
          <Row k="Embeddings" v="very high cardinality inside a network" />
          <Row k="Never" v="arbitrary integers for unordered categories in a linear/distance model" />
          <Row k="Numeric NA" v="median + missing indicator" />
          <Row k="Categorical NA" v="explicit 'Missing' level" />
          <Row k="GBMs" v="handle missing natively — often just let them" />
        </Block>

        <Block title="Ensembles" accent="indigo">
          <Row k="Bagging" v="parallel · independent · reduces variance · Random Forest" />
          <Row k="Boosting" v="sequential · corrective · reduces bias · XGBoost" />
          <Row k="Stacking" v="meta-model over out-of-fold base predictions" />
          <Row k="Bootstrap" v="≈63.2% unique rows; ≈36.8% out-of-bag" />
          <Row k="Requirement" v="the members must make different mistakes" />
        </Block>

        <Block title="PCA & clustering" accent="violet">
          <Row k="PCA" v="orthogonal directions of maximum variance" />
          <Row k="Before PCA" v="standardize — variance depends on units" />
          <Row k="Choosing k" v="cumulative explained variance (e.g. 95%)" />
          <Row k="PCA vs t-SNE" v="PCA transforms new data; t-SNE is a plot" />
          <Row k="K-Means" v="assign → recompute centroids → repeat" />
          <Row k="Choosing K" v="elbow on inertia · silhouette score · domain" />
          <Row k="K-Means fails" v="non-spherical clusters, unequal density, unscaled features" />
        </Block>

        <Block title="Pipeline & production" accent="rose">
          <Row
            k="Order"
            v="frame → clean → split → features → preprocess → baseline → train → validate → tune → threshold → test once → deploy → monitor"
          />
          <Row k="Ship" v="the whole pipeline, not just the model" />
          <Row k="Data drift" v="the inputs shift" />
          <Row k="Concept drift" v="the input→target relationship shifts" />
          <Row k="Train/serve skew" v="features computed differently offline vs online" />
          <Row k="Retrain trigger" v="schedule + drift threshold + performance threshold" />
        </Block>
      </div>
    </>
  );
}

/* ------------------------------------------------------------------ */
/* Readiness check                                                     */
/* ------------------------------------------------------------------ */

const READINESS = [
  "I can explain overfitting and underfitting, and say how I'd diagnose each from two numbers.",
  "I can explain bias vs variance, and what happens to each as complexity increases.",
  "I can explain what data leakage is and name four concrete ways it happens.",
  "I understand train / validation / test splitting, and why the test set is opened only once.",
  "I can choose between stratified, grouped and time-based splits, and say why random splitting is sometimes wrong.",
  "I can write the confusion matrix and derive precision, recall, specificity and F1 from it.",
  "I can choose between precision and recall for a given business problem — and explain that it depends on error cost.",
  "I can explain when ROC-AUC is misleading and why PR-AUC is better on imbalanced data.",
  "I can explain MAE vs RMSE and when I'd prefer each.",
  "I can explain K-fold cross-validation and answer whether I still need a test set.",
  "I can explain L1 vs L2, including why L1 produces exact zeros.",
  "I can write the gradient descent update rule and explain the minus sign and the learning rate.",
  "I can compare random forest and gradient boosting on structure, tuning and overfitting risk.",
  "I can explain why feature scaling matters, and why tree models don't need it.",
  "I can name the encodings for categorical features and say why arbitrary integer encoding is a trap.",
  "I can list four ways to handle class imbalance, in the order I'd try them.",
  "I can explain PCA, why standardizing comes first, and what it costs me.",
  "I can describe a complete ML pipeline from raw data to monitoring in about 90 seconds.",
];

export function ReadinessCheck() {
  const checked = useCheckedSet("readiness");
  const total = READINESS.length;
  const done = READINESS.filter((_, i) => checked.has(`r${i}`)).length;
  const pct = Math.round((done / total) * 100);
  const allDone = done === total;

  return (
    <div className="mt-10">
      <div className="card overflow-hidden">
        <div className="border-b border-line bg-gradient-to-r from-indigo-500/12 to-violet-500/12 px-5 py-4">
          <h3 className="text-[1.3rem] font-extrabold text-ink">Interview Readiness Check</h3>
          <p className="mt-0.5 text-[0.9rem] text-muted">
            Tick only what you can explain <em>out loud, without notes</em>. This is the honest version of the
            progress bar.
          </p>
          <div className="mt-3 flex items-center gap-3">
            <div className="h-2 flex-1 overflow-hidden rounded-full bg-surface-3">
              <div
                className="h-full rounded-full bg-gradient-to-r from-indigo-500 to-violet-500 transition-[width] duration-500"
                style={{ width: `${pct}%` }}
              />
            </div>
            <span className="font-mono text-[13px] font-extrabold text-ink">
              {done}/{total}
            </span>
          </div>
        </div>

        <ul className="divide-y divide-[var(--border)]">
          {READINESS.map((text, i) => {
            const id = `r${i}`;
            const on = checked.has(id);
            return (
              <li key={id}>
                <button
                  type="button"
                  onClick={() => checked.toggle(id)}
                  aria-pressed={on}
                  className="flex w-full items-start gap-3 px-4 py-2.5 text-left transition-colors hover:bg-surface-2"
                >
                  <span
                    className={`mt-0.5 grid size-4 shrink-0 place-items-center rounded-[5px] border transition-colors ${
                      on
                        ? "border-emerald-500 bg-emerald-500 text-white dark:border-emerald-400 dark:bg-emerald-400 dark:text-slate-900"
                        : "border-line-strong text-transparent"
                    }`}
                    aria-hidden
                  >
                    <svg width="9" height="9" viewBox="0 0 24 24" fill="none">
                      <path
                        d="M4 12.5l5.2 5L20 6.5"
                        stroke="currentColor"
                        strokeWidth="3.4"
                        strokeLinecap="round"
                        strokeLinejoin="round"
                      />
                    </svg>
                  </span>
                  <span
                    className={`text-[0.9rem] leading-relaxed ${
                      on ? "text-faint line-through decoration-1" : "text-muted"
                    }`}
                  >
                    {text}
                  </span>
                </button>
              </li>
            );
          })}
        </ul>

        {done > 0 && !allDone && (
          <div className="border-t border-line bg-surface-2 px-4 py-2.5 text-[12.5px] text-faint">
            {total - done} to go. Anything still unticked is exactly what to re-read.
          </div>
        )}
      </div>

      <div
        className={`mt-5 overflow-hidden rounded-2xl border p-6 text-center transition-colors sm:p-8 ${
          allDone
            ? "border-emerald-300 bg-emerald-50/70 dark:border-emerald-400/35 dark:bg-emerald-500/10"
            : "border-line bg-surface"
        }`}
      >
        {allDone && (
          <div className="mb-2 text-2xl" aria-hidden>
            ✓
          </div>
        )}
        <p className="mx-auto max-w-2xl text-[1.05rem] font-bold leading-relaxed text-ink sm:text-[1.15rem]">
          If you can explain every red-priority topic without looking at the notes, you are ready for the Machine
          Learning fundamentals portion of an AI Engineer interview.
        </p>
        <p className="mx-auto mt-3 max-w-xl text-[0.9rem] leading-relaxed text-muted">
          {allDone
            ? "That's all eighteen. Do one last Rapid-Fire round for recall speed, skim the cheat sheet on the way in, and stop revising."
            : "And when an answer genuinely depends on context, say so — “it depends on the cost of false positives versus false negatives” is a stronger answer than a confident rule."}
        </p>
        <Callout variant="tip" className="mx-auto mt-5 max-w-2xl text-left" title="Three habits for the actual interview">
          <ul className="bullet-list">
            <li>
              <strong>Name the concept first, then explain.</strong> "That's a bias problem — both scores are low
              and close together, which means…" Interviewers are listening for the label.
            </li>
            <li>
              <strong>Reach for a concrete number or example.</strong> "At 1% positives, always-predict-negative
              scores 99%" lands far harder than "accuracy can be misleading."
            </li>
            <li>
              <strong>Say what you'd check, not just what you'd do.</strong> Diagnosis before treatment is the
              single clearest signal of someone who has actually debugged a model.
            </li>
          </ul>
        </Callout>
      </div>
    </div>
  );
}
